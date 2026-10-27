# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "torch_geometric",
#   "scikit-learn",
#   "torch",
# ]
# ///
# MAGIC %md
# MAGIC # Edge overload prediction with TGCN, TGN and TGAT
# MAGIC
# MAGIC This notebook predicts whether a call graph edge `(UM, DM, rpctype)` is overloaded in minute `t`.
# MAGIC It scores every edge that is active in minute `t` and uses only information up to minute `t - 1`.
# MAGIC It reads the output of `feature_engineering_pipeline_v2` and the raw call graph, which gives the call count of each edge per minute.
# MAGIC
# MAGIC **Label.** An edge is overloaded when its call count is above the `label_quantile` of that edge's own call counts in the
# MAGIC training window. Edges with fewer than `min_obs` training minutes use the quantile of their rpctype. An rpctype never seen in
# MAGIC training uses the overall training quantile. No validation or test data is used to set a threshold.
# MAGIC
# MAGIC **Scope.** Sections 10 to 13 score the edges active in minute `t`. Knowing that an edge is active is information from
# MAGIC minute `t`, so that task is "given that this edge is called now, is it overloaded". The export at the end scores every regular
# MAGIC edge before minute `t`, which is the prospective task, and reports both settings side by side.
# MAGIC The label is a high sampled call count, not a measured CPU shortage.
# MAGIC
# MAGIC **Tuning and training.** Each model has its own hyperparameters. All models get the same maximum number of epochs by default, and the hyperparameter table reports epochs run and training time. A random search on the
# MAGIC validation window chooses the hyperparameters, and training stops early when the validation PR-AUC stops improving.
# MAGIC The decision threshold is chosen on validation with the `threshold_objective` widget and then applied unchanged to the test window.
# MAGIC
# MAGIC **Evaluation.** Test metrics come with 95% intervals that treat each test minute as one unit. Each model is compared with the
# MAGIC persistence baseline ("overloaded now if it was overloaded one minute ago") and with the other models, with Holm adjusted p values.
# MAGIC
# MAGIC | Section | Content |
# MAGIC |---|---|
# MAGIC | 1 | Settings and hyperparameters |
# MAGIC | 2 | Load the feature tables |
# MAGIC | 3 | Call counts and labels |
# MAGIC | 4 | Build the per minute time steps |
# MAGIC | 5 | Models |
# MAGIC | 6 | Training with early stopping |
# MAGIC | 7 | Hyperparameter search |
# MAGIC | 8 | Evaluation and statistical tests |
# MAGIC | 9 | Chart functions |
# MAGIC | 10 to 12 | TGCN, TGN and TGAT: tune, train, evaluate, charts |
# MAGIC | 13 | Model comparison |
# MAGIC | 14 | Save results |

# COMMAND ----------

# MAGIC %md
# MAGIC **Storage.** This is the tail feature version. Everything it writes goes under `s3://research-s20426/tail_features`.
# MAGIC The folders of the previous approach (`tgnn_dataset2`, `tgnn_event_dataset`, `load_balancing_work`) are never read or overwritten here.
# MAGIC
# MAGIC | Folder | Written by | Contents |
# MAGIC |---|---|---|
# MAGIC | `tail_features/tgnn_dataset` | snapshot FE, snapshot CPU and memory models, edge classifier | node_index, snapshots, manifests, dataset_stats, CPU and memory forecasts and results, edge_overload_predictions, anomaly_results |
# MAGIC | `tail_features/tgnn_event_dataset` | event stream FE, event stream CPU and memory models | events, node_feat_timeline, dataset_stats, event forecasts and results |
# MAGIC | `tail_features/load_balancing_work` | load balancing | container_minutes and other working tables |
# MAGIC
# MAGIC Inputs stay shared: `etl_output_root` and `msrtqps_parquet/parquet_final`.

# COMMAND ----------

# MAGIC %pip install torch_geometric scikit-learn torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
from matplotlib.ticker import MaxNLocator
from pyspark.sql import functions as F
from scipy import stats as sstats
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings and hyperparameters

# COMMAND ----------

# Settings

dbutils.widgets.text("fe_output_root", "s3://research-s20426/tail_features/tgnn_dataset", "Feature engineering output folder")
dbutils.widgets.text("etl_output_root", "s3://research-s20426/etl_output_root", "ETL output folder (for call counts)")
dbutils.widgets.text("callgraph_dir", "callgraph_1", "Call graph folder inside the ETL output")
dbutils.widgets.text("artifact_dir", "", "Folder for checkpoints, charts and results (blank means a temporary folder)")
dbutils.widgets.text("label_quantile", "0.95", "Overload threshold quantile, fitted on training minutes per edge")
dbutils.widgets.text("min_obs", "20", "Training minutes needed for an edge's own threshold")
dbutils.widgets.dropdown("imbalance", "weighted_bce", ["weighted_bce", "sqrt_weighted_bce", "focal", "none"],
                         "Loss for the class imbalance")
dbutils.widgets.text("focal_gamma", "2.0", "Focal loss gamma (only for imbalance = focal)")
dbutils.widgets.text("focal_alpha", "0.75", "Focal loss weight of the overload class (only for imbalance = focal)")
dbutils.widgets.dropdown("threshold_objective", "balanced_accuracy", ["balanced_accuracy", "f1", "f2", "mcc"],
                         "Metric maximised on validation to choose the threshold")
dbutils.widgets.text("seeds", "42", "Training seeds separated by commas")
dbutils.widgets.text("tgcn_max_epochs", "15", "Maximum epochs for TGCN")
dbutils.widgets.text("tgn_max_epochs", "15", "Maximum epochs for TGN")
dbutils.widgets.text("tgat_max_epochs", "15", "Maximum epochs for TGAT")
dbutils.widgets.text("patience", "4", "Epochs without validation improvement before stopping")
dbutils.widgets.dropdown("run_tuning", "yes", ["yes", "no"], "Run the hyperparameter search")
dbutils.widgets.text("tuning_trials", "6", "Configurations tried per model")
dbutils.widgets.text("tgcn_tuning_epochs", "5", "Maximum epochs per TGCN tuning trial")
dbutils.widgets.text("tgn_tuning_epochs", "5", "Maximum epochs per TGN tuning trial")
dbutils.widgets.text("tgat_tuning_epochs", "5", "Maximum epochs per TGAT tuning trial")
dbutils.widgets.text("log_every", "100", "Print the running training loss every N minutes (0 turns it off)")

FE_ROOT = dbutils.widgets.get("fe_output_root")
PREVIOUS_APPROACH_ROOTS = {"s3://research-s20426/tgnn_dataset2", "s3://research-s20426/tgnn_event_dataset", "s3://research-s20426/load_balancing_work"}
assert FE_ROOT.rstrip("/") not in PREVIOUS_APPROACH_ROOTS, "This is the tail feature version. It must not write into the folders of the previous approach."
ETL_ROOT = dbutils.widgets.get("etl_output_root")
CALLGRAPH_DIR = dbutils.widgets.get("callgraph_dir")
ARTIFACT_DIR = dbutils.widgets.get("artifact_dir").strip()
LABEL_QUANTILE = float(dbutils.widgets.get("label_quantile"))
MIN_OBS = int(dbutils.widgets.get("min_obs"))
IMBALANCE = dbutils.widgets.get("imbalance")
FOCAL_GAMMA = float(dbutils.widgets.get("focal_gamma"))
FOCAL_ALPHA = float(dbutils.widgets.get("focal_alpha"))
THRESHOLD_OBJECTIVE = dbutils.widgets.get("threshold_objective")
SEEDS = [int(s) for s in dbutils.widgets.get("seeds").split(",") if s.strip()]
MAX_EPOCHS = {
    "TGCN": int(dbutils.widgets.get("tgcn_max_epochs")),
    "TGN": int(dbutils.widgets.get("tgn_max_epochs")),
    "TGAT": int(dbutils.widgets.get("tgat_max_epochs"))
}
PATIENCE = int(dbutils.widgets.get("patience"))
RUN_TUNING = dbutils.widgets.get("run_tuning") == "yes"
TUNING_TRIALS = int(dbutils.widgets.get("tuning_trials"))
TUNING_MAX_EPOCHS = {
    "TGCN": int(dbutils.widgets.get("tgcn_tuning_epochs")),
    "TGN": int(dbutils.widgets.get("tgn_tuning_epochs")),
    "TGAT": int(dbutils.widgets.get("tgat_tuning_epochs"))
}
LOG_EVERY = int(dbutils.widgets.get("log_every"))
assert SEEDS, "give at least one seed"

FILE_TAG = "overload"
OUT_DIR = ARTIFACT_DIR or tempfile.mkdtemp(prefix=f"{FILE_TAG}_")
os.makedirs(OUT_DIR, exist_ok=True)
if not ARTIFACT_DIR:
    print("artifact_dir is blank, so results go to a temporary folder that is deleted with the session.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"label: call count above the per edge train quantile {LABEL_QUANTILE}, device {DEVICE}")
print(f"loss: {IMBALANCE}, threshold chosen on validation by {THRESHOLD_OBJECTIVE}")
print(f"maximum epochs: {MAX_EPOCHS}, patience {PATIENCE}, seeds {SEEDS}")
print(f"tuning: {'on' if RUN_TUNING else 'off'}, {TUNING_TRIALS} trials per model")
print(f"maximum epochs per tuning trial: {TUNING_MAX_EPOCHS}")

# COMMAND ----------

# Hyperparameters
#
# DEFAULT_PARAMS is used when tuning is off. It is also the first configuration tried by the search.
# SEARCH_SPACE lists the values the random search can pick for each model.

DEFAULT_PARAMS = {
    "TGCN": {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.2, "tbptt": 4},
    "TGN": {
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.2,
        "time_dim": 16, "neighbor_size": 10
    },
    "TGAT": {
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.2,
        "time_dim": 16, "neighbor_size": 10
    }
}

SEARCH_SPACE = {
    "TGCN": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.2, 0.4],
        "tbptt": [2, 4, 8]
    },
    "TGN": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.2, 0.4],
        "time_dim": [8, 16, 32],
        "neighbor_size": [5, 10, 20]
    },
    "TGAT": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.2, 0.4],
        "time_dim": [8, 16, 32],
        "neighbor_size": [5, 10, 20]
    }
}

TUNED_PARAMS = {}      # filled by tune_model()
TUNING_RESULTS = {}    # one table of trials per model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the feature tables

# COMMAND ----------

# Load

stats_text = "\n".join(row.value for row in spark.read.text(f"{FE_ROOT}/dataset_stats").collect())
stats = json.loads(stats_text)

NUM_NODES = int(stats["num_nodes"])
NODE_FEATS = stats["node_feature_names"]
TRAIN_RANGE = stats["train_t_range"]
VAL_RANGE = stats["val_t_range"]
TEST_RANGE = stats["test_t_range"]
TRAIN_END = TRAIN_RANGE[1] + 1          # first minute after training, as in the FE notebook


def read_all_splits(kind):
    """Union of the train, val and test snapshot tables of one kind (nodes or edges)."""
    frames = [spark.read.parquet(f"{FE_ROOT}/snapshots/{split}/{kind}") for split in ("train", "val", "test")]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


node_index = spark.read.parquet(f"{FE_ROOT}/node_index")
edge_snapshots = read_all_splits("edges")
node_snapshots = read_all_splits("nodes")
print(f"nodes in graph: {NUM_NODES:,}, node features: {len(NODE_FEATS)}")
print(f"splits by minute: train {TRAIN_RANGE}, val {VAL_RANGE}, test {TEST_RANGE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Call counts and labels
# MAGIC The labels need the exact call count of each edge per minute, so the counts are taken from the raw call graph
# MAGIC and joined on the same keys. Each edge also gets its features from the previous minute.

# COMMAND ----------

# Labels

callgraph = spark.read.parquet(f"{ETL_ROOT}/{CALLGRAPH_DIR}").withColumn("t_idx", (F.col("timestamp") / 60000).cast("int"))
src_lookup = node_index.select(F.col("node_key").alias("UM"), F.col("node_id").alias("src_node_id"))
dst_lookup = node_index.select(F.col("node_key").alias("DM"), F.col("node_id").alias("dst_node_id"))

counts = (
    callgraph.groupBy("UM", "DM", "t_idx", "rpctype").agg(F.count("*").alias("call_count"))
    .join(src_lookup, "UM")
    .join(dst_lookup, "DM")
    .select("src_node_id", "dst_node_id", "t_idx", "rpctype", "call_count")
)
edges = edge_snapshots.join(counts, ["src_node_id", "dst_node_id", "t_idx", "rpctype"], "inner")
print(f"edge rows: {edge_snapshots.count():,}, with call counts: {edges.count():,} (these should match)")

# Thresholds use training minutes only.
train_edges = edges.filter(F.col("t_idx") < TRAIN_END)
quantile = f"percentile_approx(call_count, {LABEL_QUANTILE})"
edge_threshold = train_edges.groupBy("src_node_id", "dst_node_id", "rpctype").agg(
    F.count("*").alias("train_minutes"), F.expr(quantile).alias("edge_threshold"))
type_threshold = train_edges.groupBy("rpctype").agg(F.expr(quantile).alias("type_threshold"))
GLOBAL_THRESHOLD = float(train_edges.agg(F.expr(quantile)).first()[0])
log_count = train_edges.select(F.log1p("call_count").alias("value")).agg(F.mean("value"), F.stddev("value")).first()
LOG_COUNT_MEAN, LOG_COUNT_STD = float(log_count[0]), float(log_count[1] or 1.0)

edges = (
    edges.join(edge_threshold, ["src_node_id", "dst_node_id", "rpctype"], "left")
    .join(type_threshold, "rpctype", "left")
    .withColumn("threshold", F.when(F.col("train_minutes") >= MIN_OBS, F.col("edge_threshold"))
                .otherwise(F.coalesce(F.col("type_threshold"), F.lit(GLOBAL_THRESHOLD))))
    .withColumn("label", (F.col("call_count") > F.col("threshold")).cast("float"))
    .withColumn("log_count_z", (F.log1p("call_count") - F.lit(LOG_COUNT_MEAN)) / F.lit(LOG_COUNT_STD))
)

# Features of the same edge one minute earlier. These are known before minute t.
current = edges.select("src_node_id", "dst_node_id", "rpctype", "t_idx", "rpctype_idx",
                       "rt_mean_rs", "rt_abs_mean_rs", "rt_abs_p95_log_rs", "rt_abs_std_log_rs", "log_count_z", "label")
previous = current.select(
    "src_node_id", "dst_node_id", "rpctype", (F.col("t_idx") + 1).alias("t_idx"),
    F.col("rt_mean_rs").alias("prev_rt"), F.col("rt_abs_mean_rs").alias("prev_rt_abs"),
    F.col("rt_abs_p95_log_rs").alias("prev_rt_p95"), F.col("rt_abs_std_log_rs").alias("prev_rt_std"),
    F.col("log_count_z").alias("prev_count"), F.col("label").alias("prev_label")
)
table = (
    current.join(previous, ["src_node_id", "dst_node_id", "rpctype", "t_idx"], "left")
    .withColumn("has_prev", F.col("prev_count").isNotNull().cast("float"))
    .fillna(0.0, subset=["prev_rt", "prev_rt_abs", "prev_rt_p95", "prev_rt_std", "prev_count", "prev_label"])
)

edge_table = table.toPandas().sort_values("t_idx", kind="stable").reset_index(drop=True)
node_table = node_snapshots.select("node_id", "t_idx", *NODE_FEATS).toPandas()
node_table = node_table.sort_values("t_idx", kind="stable").reset_index(drop=True)
assert np.isin(edge_table.label.unique(), [0.0, 1.0]).all(), "labels must be 0 or 1"


def in_range(minutes, value_range):
    return (minutes >= value_range[0]) & (minutes <= value_range[1])


POSITIVE_RATE = {split: float(edge_table.label[in_range(edge_table.t_idx, rng)].mean())
                 for split, rng in (("train", TRAIN_RANGE), ("val", VAL_RANGE), ("test", TEST_RANGE))}
print("overload rate by split:", {split: round(rate, 4) for split, rate in POSITIVE_RATE.items()})
print(f"edge rows: {len(edge_table):,}, node rows: {len(node_table):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build the per minute time steps
# MAGIC
# MAGIC Step `t` holds the edges of minute `t`. The model first predicts their labels from the previous minute features and its state,
# MAGIC then ingests minute `t`. Inputs replace NaN by 0 and are clipped to plus or minus 10.

# COMMAND ----------

# Arrays and time steps

CLIP = 10.0
N_RPC = int(min(16, np.nan_to_num(edge_table.rpctype_idx.to_numpy()).max() + 1))
rpc = np.clip(np.nan_to_num(edge_table.rpctype_idx.to_numpy()).astype(int), 0, N_RPC - 1)
rpc_onehot = np.eye(N_RPC, dtype=np.float32)[rpc]

# What an edge looks like in minute t. Used to update the model after predicting.
current_features = edge_table[["rt_mean_rs", "rt_abs_mean_rs", "rt_abs_p95_log_rs", "rt_abs_std_log_rs", "log_count_z"]].to_numpy(np.float32)
current_features = np.clip(np.nan_to_num(current_features), -CLIP, CLIP)
EDGE_MSG = np.concatenate([current_features, rpc_onehot], axis=1)
# What was known about the edge before minute t. Used to predict.
previous_features = edge_table[["prev_rt", "prev_rt_abs", "prev_rt_p95", "prev_rt_std", "prev_count"]].to_numpy(np.float32)
previous_features = np.clip(np.nan_to_num(previous_features), -CLIP, CLIP)
TARGET_FEATURES = np.concatenate([previous_features, edge_table[["has_prev"]].to_numpy(np.float32), rpc_onehot], axis=1)

MSG_DIM = EDGE_MSG.shape[1]
TARGET_DIM = TARGET_FEATURES.shape[1]
EDGE_SRC = edge_table.src_node_id.to_numpy(np.int64)
EDGE_DST = edge_table.dst_node_id.to_numpy(np.int64)
LABELS = edge_table.label.to_numpy(np.float32)
PREV_LABELS = edge_table.prev_label.to_numpy(np.float32)
NODE_INPUTS = np.clip(np.nan_to_num(node_table[NODE_FEATS].to_numpy(np.float32)), -CLIP, CLIP)
NODE_DIM = NODE_INPUTS.shape[1]
NODE_IDS = node_table.node_id.to_numpy(np.int64)
TOTAL_EVENTS = len(edge_table)


def row_ranges(minutes):
    """Map each minute to the (start, end) rows it occupies in an array sorted by minute."""
    unique, starts = np.unique(minutes, return_index=True)
    ends = np.append(starts[1:], len(minutes))
    return {int(m): (int(s), int(e)) for m, s, e in zip(unique, starts, ends)}


edge_rows = row_ranges(edge_table.t_idx.to_numpy())
node_rows = row_ranges(node_table.t_idx.to_numpy())
ALL_MINUTES = sorted(set(edge_rows) | set(node_rows))

STEPS = {}
for t in ALL_MINUTES:
    edge_start, edge_end = edge_rows.get(t, (0, 0))
    node_start, node_end = node_rows.get(t, (0, 0))
    STEPS[t] = {
        "t": t,
        "nid": torch.from_numpy(NODE_IDS[node_start:node_end]),
        "x": torch.from_numpy(NODE_INPUTS[node_start:node_end]),
        "src": torch.from_numpy(EDGE_SRC[edge_start:edge_end]),
        "dst": torch.from_numpy(EDGE_DST[edge_start:edge_end]),
        "msg": torch.from_numpy(EDGE_MSG[edge_start:edge_end]),
        "target": {
            "t": t,
            "src": torch.from_numpy(EDGE_SRC[edge_start:edge_end]),
            "dst": torch.from_numpy(EDGE_DST[edge_start:edge_end]),
            "features": torch.from_numpy(TARGET_FEATURES[edge_start:edge_end]),
            "y": torch.from_numpy(LABELS[edge_start:edge_end]),
            "prev_label": torch.from_numpy(PREV_LABELS[edge_start:edge_end]),
            "row": torch.arange(edge_start, edge_end, dtype=torch.long)
        }
    }

TRAIN_STEPS = [t for t in ALL_MINUTES if TRAIN_RANGE[0] <= t <= TRAIN_RANGE[1]]
VAL_STEPS = [t for t in ALL_MINUTES if VAL_RANGE[0] <= t <= VAL_RANGE[1]]
TEST_STEPS = [t for t in ALL_MINUTES if TEST_RANGE[0] <= t <= TEST_RANGE[1]]
ignored = [t for t in ALL_MINUTES if t > TEST_RANGE[1]]
if ignored:
    print(f"edge minutes after the test range are ignored: {ignored} (no node data, usually a partial last minute)")


def to_device(obj):
    if isinstance(obj, dict):
        return {key: to_device(value) for key, value in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(DEVICE, non_blocking=True)
    return obj


train_labels = LABELS[in_range(edge_table.t_idx.to_numpy(), TRAIN_RANGE)]
POS_WEIGHT = torch.tensor((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0), device=DEVICE)
print(f"steps: train {len(TRAIN_STEPS)}, val {len(VAL_STEPS)}, test {len(TEST_STEPS)}")
print(f"training overload rate {train_labels.mean():.2%}, negative to positive ratio {POS_WEIGHT.item():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Models
# MAGIC
# MAGIC All three models share the edge classifier head and the node state. Only the encoder differs.
# MAGIC
# MAGIC | Model | Over time | Over the graph |
# MAGIC |---|---|---|
# MAGIC | TGCN | GRU hidden state per node, carried from minute to minute | Two GCN layers on each snapshot |
# MAGIC | TGN | Memory per node, updated by every edge | One attention layer over the latest edges of each node |
# MAGIC | TGAT | No memory, cosine encoding of how long ago an edge was seen | Two attention layers over a two hop neighbourhood |

# COMMAND ----------

# Shared model parts


class EdgeHead(nn.Module):
    """Maps the embeddings of both endpoints, the previous minute edge features and the endpoints' last observed
    node features to an overload logit."""

    def __init__(self, embedding_dim, dropout):
        super().__init__()
        input_dim = 2 * embedding_dim + TARGET_DIM + 2 * NODE_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, src_embedding, dst_embedding, edge_features, src_x, dst_x):
        features = torch.cat([src_embedding, dst_embedding, edge_features, src_x, dst_x], dim=-1)
        return self.net(features).squeeze(-1)


class NodeStateModel(nn.Module):
    """Keeps the last observed features of every node. Subclasses add the encoder."""

    tbptt = 1

    def __init__(self):
        super().__init__()
        self.register_buffer("last_x", torch.zeros(NUM_NODES, NODE_DIM), persistent=False)

    def reset_state(self):
        self.last_x.zero_()

    def detach_state(self):
        pass

    def remember_nodes(self, step):
        if step["nid"].numel() > 0:
            self.last_x[step["nid"]] = step["x"]

    def classify(self, src_embedding, dst_embedding, target):
        src_x = self.last_x[target["src"]]
        dst_x = self.last_x[target["dst"]]
        return self.head(src_embedding, dst_embedding, target["features"], src_x, dst_x)


class RecentNeighborLoader(LastNeighborLoader):
    """Keeps exactly the `size` most recent interactions of every node.
    PyG's LastNeighborLoader keeps an arbitrary subset when a node has more than `size` interactions in one insert call."""

    def insert(self, src, dst):
        n = src.numel()
        if n == 0:
            return
        first_id = self.cur_e_id
        self.cur_e_id += n

        # Each interaction is stored twice, once for each endpoint, and sorted by node and then by arrival.
        position = torch.arange(n, device=src.device).repeat(2)
        nodes = torch.cat([dst, src])
        neighbors = torch.cat([src, dst])
        order = torch.argsort(nodes * n + position)
        nodes, neighbors, position = nodes[order], neighbors[order], position[order]

        node_ids, group, counts = torch.unique_consecutive(nodes, return_inverse=True, return_counts=True)
        rank = torch.arange(nodes.numel(), device=src.device) - (torch.cumsum(counts, 0) - counts)[group]
        n_dropped = (counts[group] - self.size).clamp(min=0)
        keep = rank >= n_dropped
        slot = group[keep] * self.size + (rank - n_dropped)[keep]

        new_e_id = position.new_full((node_ids.numel() * self.size,), -1)
        new_e_id[slot] = position[keep] + first_id
        new_neighbors = neighbors.new_zeros(node_ids.numel() * self.size)
        new_neighbors[slot] = neighbors[keep]

        # Merge with what each node already had and keep the newest `size`.
        all_e_id = torch.cat([self.e_id[node_ids], new_e_id.view(-1, self.size)], dim=-1)
        all_neighbors = torch.cat([self.neighbors[node_ids], new_neighbors.view(-1, self.size)], dim=-1)
        top_e_id, top_index = all_e_id.topk(self.size, dim=-1)
        self.e_id[node_ids] = top_e_id
        self.neighbors[node_ids] = torch.gather(all_neighbors, 1, top_index)


class EdgeLog:
    """Time and message of every edge ingested so far. The position in the log is the edge id used by the loader."""

    def __init__(self):
        self.time = torch.zeros(max(TOTAL_EVENTS, 1), dtype=torch.long, device=DEVICE)
        self.msg = torch.zeros(max(TOTAL_EVENTS, 1), MSG_DIM, device=DEVICE)
        self.size = 0

    def reset(self):
        self.size = 0

    def add(self, time_seconds, msg):
        end = self.size + time_seconds.numel()
        self.time[self.size:end] = time_seconds
        self.msg[self.size:end] = msg
        self.size = end


class CosineTimeEncoder(nn.Module):
    """TGAT time encoding: cos(w * t + b) with learnable frequencies."""

    def __init__(self, dim):
        super().__init__()
        self.out_channels = dim
        self.w = nn.Parameter(torch.from_numpy(1.0 / 10 ** np.linspace(0, 6, dim)).float())
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        return torch.cos(t.view(-1, 1) * self.w + self.b)

# COMMAND ----------

# TGCN


class TGCNModel(NodeStateModel):
    """Two GCN layers on each snapshot feed a GRU cell per node."""

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        self.tbptt = params["tbptt"]
        self.hidden_dim = hidden
        self.gcn1 = GCNConv(NODE_DIM, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = EdgeHead(hidden, params["dropout"])
        self.register_buffer("h", torch.zeros(NUM_NODES, hidden), persistent=False)

    def reset_state(self):
        super().reset_state()
        self.h = torch.zeros(NUM_NODES, self.hidden_dim, device=self.last_x.device)

    def detach_state(self):
        self.h = self.h.detach()

    def predict(self, target):
        return self.classify(self.h[target["src"]], self.h[target["dst"]], target)

    def update(self, step):
        if step["nid"].numel() > 0:
            x = torch.zeros(NUM_NODES, NODE_DIM, device=step["x"].device)
            x[step["nid"]] = step["x"]
            edges = torch.stack([step["src"], step["dst"]])
            edges = torch.cat([edges, edges.flip(0)], dim=1)
            z = TF.relu(self.gcn1(x, edges))
            z = self.gcn2(z, edges)
            new_h = self.gru(z[step["nid"]], self.h[step["nid"]])
            self.h = self.h.index_copy(0, step["nid"], new_h)
        self.remember_nodes(step)

# COMMAND ----------

# TGN


class TemporalAttention(nn.Module):
    """One graph attention layer whose edge features are a time encoding and the edge message."""

    def __init__(self, dim, time_encoder, dropout):
        super().__init__()
        self.time_encoder = time_encoder
        self.conv = TransformerConv(dim, dim // 2, heads=2, dropout=dropout, edge_dim=MSG_DIM + time_encoder.out_channels)

    def forward(self, x, last_update, edge_index, edge_time, edge_msg):
        relative_time = (last_update[edge_index[0]] - edge_time).to(x.dtype)
        edge_attr = torch.cat([self.time_encoder(relative_time), edge_msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGNModel(NodeStateModel):
    """Per node memory updated by every edge, followed by one attention layer over recent neighbours."""

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        time_dim = params["time_dim"]
        self.memory = TGNMemory(
            NUM_NODES, MSG_DIM, hidden, time_dim,
            message_module=IdentityMessage(MSG_DIM, hidden, time_dim),
            aggregator_module=LastAggregator()
        )
        self.attention = TemporalAttention(hidden, self.memory.time_enc, params["dropout"])
        self.head = EdgeHead(hidden, params["dropout"])
        self.loader = RecentNeighborLoader(NUM_NODES, size=params["neighbor_size"], device=DEVICE)
        self.log = EdgeLog()
        self.local_index = torch.empty(NUM_NODES, dtype=torch.long, device=DEVICE)

    def reset_state(self):
        super().reset_state()
        self.memory.reset_state()
        self.loader.reset_state()
        self.log.reset()

    def detach_state(self):
        self.memory.detach()

    def predict(self, target):
        query = torch.cat([target["src"], target["dst"]]).unique()
        node_ids, edge_index, edge_ids = self.loader(query)
        self.local_index[node_ids] = torch.arange(node_ids.size(0), device=node_ids.device)
        memory, last_update = self.memory(node_ids)
        embedding = self.attention(memory, last_update, edge_index, self.log.time[edge_ids], self.log.msg[edge_ids])
        return self.classify(embedding[self.local_index[target["src"]]], embedding[self.local_index[target["dst"]]], target)

    def update(self, step):
        if step["src"].numel() > 0:
            seconds = torch.full((step["src"].numel(),), step["t"] * 60, dtype=torch.long, device=step["src"].device)
            self.memory.update_state(step["src"], step["dst"], seconds, step["msg"])
            self.loader.insert(step["src"], step["dst"])
            self.log.add(seconds, step["msg"])
        self.remember_nodes(step)

# COMMAND ----------

# TGAT


class TGATModel(NodeStateModel):
    """Two attention layers over a two hop neighbourhood, with a cosine encoding of how long ago each edge was seen.
    Layer 1 updates every neighbour from its own neighbours. Layer 2 updates the endpoints from their neighbours."""

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        self.time_encoder = CosineTimeEncoder(params["time_dim"])
        self.input = nn.Linear(NODE_DIM, hidden)
        edge_dim = MSG_DIM + params["time_dim"]
        self.layer1 = TransformerConv(hidden, hidden // 2, heads=2, dropout=params["dropout"], edge_dim=edge_dim)
        self.layer2 = TransformerConv(hidden, hidden // 2, heads=2, dropout=params["dropout"], edge_dim=edge_dim)
        self.head = EdgeHead(hidden, params["dropout"])
        self.loader = RecentNeighborLoader(NUM_NODES, size=params["neighbor_size"], device=DEVICE)
        self.log = EdgeLog()
        self.local_index = torch.empty(NUM_NODES, dtype=torch.long, device=DEVICE)

    def reset_state(self):
        super().reset_state()
        self.loader.reset_state()
        self.log.reset()

    def edge_features(self, edge_ids, target_minute):
        minutes_ago = (target_minute * 60 - self.log.time[edge_ids]).float() / 60.0
        return torch.cat([self.time_encoder(minutes_ago), self.log.msg[edge_ids]], dim=-1)

    def embed(self, query_nodes, target_minute):
        hop1_nodes, hop1_edges, hop1_ids = self.loader(query_nodes)
        hop1_edges_global = hop1_nodes[hop1_edges]          # the loader relabels on every call, so keep global ids
        hop2_nodes, hop2_edges, hop2_ids = self.loader(hop1_nodes)
        self.local_index[hop2_nodes] = torch.arange(hop2_nodes.size(0), device=hop2_nodes.device)
        h = self.input(self.last_x[hop2_nodes])
        h = TF.relu(self.layer1(h, hop2_edges, self.edge_features(hop2_ids, target_minute)))
        return self.layer2(h, self.local_index[hop1_edges_global], self.edge_features(hop1_ids, target_minute))

    def predict(self, target):
        embedding = self.embed(torch.cat([target["src"], target["dst"]]).unique(), target["t"])
        return self.classify(embedding[self.local_index[target["src"]]], embedding[self.local_index[target["dst"]]], target)

    def update(self, step):
        if step["src"].numel() > 0:
            seconds = torch.full((step["src"].numel(),), step["t"] * 60, dtype=torch.long, device=step["src"].device)
            self.loader.insert(step["src"], step["dst"])
            self.log.add(seconds, step["msg"])
        self.remember_nodes(step)

# COMMAND ----------

# Model registry

MODEL_CLASSES = {"TGCN": TGCNModel, "TGN": TGNModel, "TGAT": TGATModel}


def build_model(name, params):
    return MODEL_CLASSES[name](params).to(DEVICE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Training with early stopping
# MAGIC
# MAGIC Every epoch walks through the training minutes in order and then continues into the validation minutes.
# MAGIC The weights of the epoch with the highest validation PR-AUC are kept.

# COMMAND ----------

# Loss and one pass over the time steps


def make_loss(strategy):
    """weighted_bce weights overloads by the negative to positive ratio of the training labels. sqrt_weighted_bce uses
    the square root of that ratio. focal down weights easy rows. none is plain binary cross entropy."""
    if strategy == "none":
        return lambda logits, y: TF.binary_cross_entropy_with_logits(logits, y)
    if strategy == "weighted_bce":
        return lambda logits, y: TF.binary_cross_entropy_with_logits(logits, y, pos_weight=POS_WEIGHT)
    if strategy == "sqrt_weighted_bce":
        return lambda logits, y: TF.binary_cross_entropy_with_logits(logits, y, pos_weight=POS_WEIGHT.sqrt())
    if strategy == "focal":
        def focal_loss(logits, y):
            bce = TF.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p_true = torch.exp(-bce)
            alpha = FOCAL_ALPHA * y + (1 - FOCAL_ALPHA) * (1 - y)
            return (alpha * (1 - p_true) ** FOCAL_GAMMA * bce).mean()
        return focal_loss
    raise ValueError(f"unknown imbalance strategy {strategy}")


LOSS_FN = make_loss(IMBALANCE)


def optimizer_step(model, optimizer, loss):
    if loss is None:
        return
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()


def run_sequence(model, steps, optimizer=None, collect=False, log_every=0, tag=""):
    """Predict and then ingest each step in order. Trains when an optimizer is given.
    Returns the mean loss. When collect is True it also returns the label, probability,
    persistence label and minute of every row."""
    training = optimizer is not None
    collected = {key: [] for key in ("y", "prob", "persistence", "minute", "row")}
    loss_sum = 0.0
    n_rows = 0
    pending_loss = None

    with torch.set_grad_enabled(training):
        for k, t in enumerate(steps, start=1):
            step = to_device(STEPS[t])
            target = step["target"]
            if target["y"].numel() > 0:
                logits = model.predict(target)
                loss = LOSS_FN(logits, target["y"])
                n = target["y"].numel()
                loss_sum += loss.item() * n
                n_rows += n
                if training:
                    pending_loss = loss if pending_loss is None else pending_loss + loss
                if collect:
                    collected["y"].append(target["y"].cpu())
                    collected["prob"].append(torch.sigmoid(logits).detach().cpu())
                    collected["persistence"].append(target["prev_label"].cpu())
                    collected["minute"].append(torch.full((n,), int(t), dtype=torch.long))
                    collected["row"].append(target["row"].cpu())

            model.update(step)

            if training and k % model.tbptt == 0:
                optimizer_step(model, optimizer, pending_loss)
                pending_loss = None
                model.detach_state()
            if training and log_every and k % log_every == 0:
                print(f"    {tag} minute {k}/{len(steps)}, running train loss {loss_sum / max(n_rows, 1):.4f}")

        if training and pending_loss is not None:
            optimizer_step(model, optimizer, pending_loss)
            model.detach_state()

    result = {"loss": loss_sum / max(n_rows, 1), "rows": n_rows}
    if collect and collected["y"]:
        for key, parts in collected.items():
            result[key] = torch.cat(parts).numpy()
    return result

# COMMAND ----------

# Training one model with early stopping


def safe_pr_auc(y, prob):
    return float(average_precision_score(y, prob)) if 0 < np.mean(y) < 1 else float("nan")


def safe_roc_auc(y, prob):
    return float(roc_auc_score(y, prob)) if 0 < np.mean(y) < 1 else float("nan")


def epoch_summary(epoch, train, val):
    return {
        "epoch": epoch,
        "train_loss": train["loss"],
        "val_loss": val["loss"],
        "train_pr_auc": safe_pr_auc(train["y"], train["prob"]),
        "val_pr_auc": safe_pr_auc(val["y"], val["prob"]),
        "train_roc_auc": safe_roc_auc(train["y"], train["prob"]),
        "val_roc_auc": safe_roc_auc(val["y"], val["prob"])
    }


def copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_run(name, params, max_epochs, seed, verbose=True):
    """Train one model with one seed. Stops after PATIENCE epochs without a higher validation PR-AUC.
    Returns the best weights, the best epoch and the history of every epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    history = []
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        model.reset_state()
        train = run_sequence(model, TRAIN_STEPS, optimizer=optimizer, collect=True,
                             log_every=LOG_EVERY if verbose else 0, tag=f"epoch {epoch}")
        model.eval()
        val = run_sequence(model, VAL_STEPS, collect=True)     # continues from the end of training, no reset
        summary = epoch_summary(epoch, train, val)
        summary["seconds"] = time.time() - start
        history.append(summary)

        if verbose:
            print(f"epoch {epoch:02d}/{max_epochs}"
                  f" | train loss {summary['train_loss']:.4f}"
                  f" | val loss {summary['val_loss']:.4f}"
                  f" | PR-AUC train {summary['train_pr_auc']:.4f} val {summary['val_pr_auc']:.4f}"
                  f" | {time.time() - start:.0f}s")

        if summary["val_pr_auc"] > best_score:
            best_score = summary["val_pr_auc"]
            best_epoch = epoch
            best_state = copy_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                if verbose:
                    print(f"early stop: no improvement for {PATIENCE} epochs, best epoch {best_epoch}")
                break

    if best_state is None:
        print("warning: validation PR-AUC was undefined in every epoch, keeping the last epoch")
        print("this happens when the validation window has no overloads")
        best_score = float("nan")
        best_epoch = len(history)
        best_state = copy_state(model)

    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": name, "seed": seed, "params": params, "state": best_state, "best_epoch": best_epoch,
            "best_val_pr_auc": best_score, "history": history,
            "epochs_run": len(history), "train_seconds": float(sum(row["seconds"] for row in history))}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hyperparameter search
# MAGIC
# MAGIC The search tries the default configuration first and then random draws from `SEARCH_SPACE`, without repeats.
# MAGIC Each trial trains with the first seed for up to the model's tuning epochs, with early stopping.
# MAGIC The configuration with the highest validation PR-AUC wins. The test window is not used.
# MAGIC Because the winner is picked on validation, its validation score is optimistic. Only test results are unbiased.

# COMMAND ----------

# Hyperparameter search


def draw_configurations(name, n_trials, seed):
    """Default configuration plus random distinct draws from the search space."""
    rng = np.random.default_rng(seed)
    space = SEARCH_SPACE[name]
    configurations = [dict(DEFAULT_PARAMS[name])]
    seen = {tuple(sorted(configurations[0].items()))}
    n_possible = int(np.prod([len(values) for values in space.values()]))
    while len(configurations) < min(n_trials, n_possible):
        params = dict(DEFAULT_PARAMS[name])
        for key, values in space.items():
            params[key] = values[int(rng.integers(len(values)))]
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            configurations.append(params)
    return configurations


def tune_model(name):
    print(f"\n==== hyperparameter search for {name}: {TUNING_TRIALS} trials ====")
    configurations = draw_configurations(name, TUNING_TRIALS, SEEDS[0])
    rows = []
    for trial, params in enumerate(configurations, start=1):
        start = time.time()
        run = train_one_run(name, params, TUNING_MAX_EPOCHS[name], SEEDS[0], verbose=False)
        rows.append({"trial": trial, **params, "best_epoch": run["best_epoch"], "val_pr_auc": run["best_val_pr_auc"]})
        print(f"trial {trial}: val PR-AUC {run['best_val_pr_auc']:.4f} at epoch {run['best_epoch']}"
              f" | {time.time() - start:.0f}s | {params}")
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False, kind="stable", na_position="last")
    table = table.reset_index(drop=True)
    best_trial = int(table.loc[0, "trial"])
    TUNED_PARAMS[name] = configurations[best_trial - 1]
    TUNING_RESULTS[name] = table
    print(f"best configuration for {name}: {TUNED_PARAMS[name]}")
    return table


def params_for(name):
    return TUNED_PARAMS.get(name, DEFAULT_PARAMS[name])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Evaluation and statistical tests
# MAGIC
# MAGIC The final model is trained once per seed with its tuned hyperparameters and its own maximum epochs. Each seed gets its own
# MAGIC threshold from validation. The first seed is used for the charts and the tests. Further seeds show how much results move between runs.
# MAGIC
# MAGIC **How the intervals work.** Every metric can be recomputed with one test minute left out. These leave one out values (jackknife)
# MAGIC show how much each minute moves the metric. Their long run variance is estimated with the equal weighted cosine method of Lazarus,
# MAGIC Lewis, Stock and Watson (2018). This gives the standard error. Intervals and p values use a t distribution. Intervals for metrics
# MAGIC between 0 and 1 are built on the logit scale and MCC on the atanh scale, so they stay inside the possible range. The threshold is
# MAGIC treated as fixed because it was chosen on validation. In simulations with 108 minutes the intervals covered the true value
# MAGIC 93 to 96% of the time and the 5% test for a difference rejected 5.2% of the time when two models were equally good.

# COMMAND ----------

# Point metrics and the decision threshold

THRESHOLD_OBJECTIVES = ("balanced_accuracy", "f1", "f2", "mcc")


def compute_metrics(y, prob, threshold):
    """Metrics at one threshold (predict overload when prob >= threshold). Accuracy is shown for reference only."""
    y = (np.asarray(y) > 0.5).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0,
        "accuracy": (tp + tn) / max(len(y), 1),
        "pr_auc": safe_pr_auc(y, prob), "roc_auc": safe_roc_auc(y, prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "rows": int(len(y)), "positive_rate": float(y.mean())
    }


def pick_threshold(y, prob, objective):
    """Threshold that maximises `objective`, checked at every distinct score in one sorted sweep."""
    y = (np.asarray(y) > 0.5).astype(int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-prob, kind="stable")
    scores, labels = prob[order], y[order]
    tp = np.cumsum(labels).astype(float)
    fp = np.cumsum(1 - labels).astype(float)
    last_of_group = np.r_[scores[1:] != scores[:-1], True]
    tp, fp, candidates = tp[last_of_group], fp[last_of_group], scores[last_of_group]
    positives, negatives = float(labels.sum()), float(len(labels) - labels.sum())
    fn, tn = positives - tp, negatives - fp
    recall = tp / max(positives, 1.0)
    specificity = tn / max(negatives, 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    if objective == "f1":
        value = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    elif objective == "f2":
        value = 5 * precision * recall / np.maximum(4 * precision + recall, 1e-12)
    elif objective == "balanced_accuracy":
        value = (recall + specificity) / 2
    else:
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        value = np.where(denominator > 0, (tp * tn - fp * fn) / np.maximum(denominator, 1e-12), 0.0)
    best = int(np.argmax(value))
    return float(candidates[best]), float(value[best])

# COMMAND ----------

# Statistical inference

METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "mcc")
TESTED_METRICS = ("pr_auc", "roc_auc", "f1", "balanced_accuracy")


class RankedScores:
    """Scores sorted once, so every metric can be recomputed for any weighting of the minutes with cumulative sums."""

    def __init__(self, y, prob, minute_index):
        prob = np.asarray(prob, dtype=float)
        order = np.argsort(-prob, kind="stable")
        self.y = (np.asarray(y) > 0.5).astype(float)[order]
        self.prob = prob[order]
        self.minute = np.asarray(minute_index)[order]
        self.last_of_group = np.r_[self.prob[1:] != self.prob[:-1], True]


def weighted_metrics(ranked, minute_weights, threshold):
    """All metrics with each row weighted by the weight of its minute. With all weights equal to 1 this equals
    compute_metrics and sklearn's average_precision_score and roc_auc_score."""
    w = minute_weights[ranked.minute]
    w_pos = w * ranked.y
    w_neg = w - w_pos
    tp = np.cumsum(w_pos)[ranked.last_of_group]
    fp = np.cumsum(w_neg)[ranked.last_of_group]
    positives, negatives = float(tp[-1]), float(fp[-1])
    out = {"pr_auc": float("nan"), "roc_auc": float("nan")}
    if positives > 0:
        recall_curve = tp / positives
        precision_curve = tp / np.maximum(tp + fp, 1e-300)
        out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve))
    if positives > 0 and negatives > 0:
        tpr = np.r_[0.0, tp / positives]
        fpr = np.r_[0.0, fp / negatives]
        out["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    predicted = ranked.prob >= threshold
    TP = float(w_pos[predicted].sum())
    FP = float(w_neg[predicted].sum())
    FN, TN = positives - TP, negatives - FP
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / positives if positives > 0 else 0.0
    specificity = TN / negatives if negatives > 0 else 0.0
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    out["balanced_accuracy"] = (recall + specificity) / 2
    out["mcc"] = (TP * TN - FP * FN) / denominator if denominator > 0 else 0.0
    return out


def ewc_terms(n_minutes):
    """Number of cosine terms recommended by Lazarus, Lewis, Stock and Watson (2018)."""
    return max(2, int(math.floor(0.4 * n_minutes ** (2 / 3))))


def ewc_variance(values, n_terms):
    """Equal weighted cosine estimate of the long run variance of a per minute series."""
    centred = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centred)
    t = np.arange(1, n + 1)
    projections = [math.sqrt(2 / n) * np.dot(centred, np.cos(math.pi * j * (t - 0.5) / n)) for j in range(1, n_terms + 1)]
    return float(np.mean(np.square(projections)))


def jackknife_all_metrics(ranked, threshold, n_minutes):
    """Full sample value and leave one minute out pseudo values of every metric, in one pass over the minutes."""
    weights = np.ones(n_minutes)
    full = weighted_metrics(ranked, weights, threshold)
    pseudo = {metric: np.empty(n_minutes) for metric in METRICS}
    for t in range(n_minutes):
        weights[t] = 0.0
        without_t = weighted_metrics(ranked, weights, threshold)
        weights[t] = 1.0
        for metric in METRICS:
            pseudo[metric][t] = n_minutes * full[metric] - (n_minutes - 1) * without_t[metric]
    return full, pseudo


def summarise(estimate, pseudo, scale="identity"):
    """Estimate, standard error, 95% interval and two sided p value against zero from pseudo values.
    scale = "logit" builds the interval on the logit scale (metrics in [0, 1]) and "atanh" on the atanh scale (MCC)."""
    pseudo = np.asarray(pseudo, dtype=float)
    n_minutes = len(pseudo)
    nan = float("nan")
    if n_minutes < 8 or not np.isfinite(estimate) or not np.all(np.isfinite(pseudo)):
        return {"est": estimate, "se": nan, "low": nan, "high": nan, "p": nan}
    dof = ewc_terms(n_minutes)
    std_error = math.sqrt(max(ewc_variance(pseudo, dof), 0.0) / n_minutes)
    if std_error == 0.0:
        return {"est": estimate, "se": 0.0, "low": estimate, "high": estimate, "p": nan}
    q = float(sstats.t.ppf(0.975, dof))
    p_value = float(2 * sstats.t.sf(abs(estimate / std_error), dof))
    if scale == "logit" and 0 < estimate < 1:
        centre = math.log(estimate / (1 - estimate))
        spread = q * std_error / (estimate * (1 - estimate))
        low, high = 1 / (1 + math.exp(-(centre - spread))), 1 / (1 + math.exp(-(centre + spread)))
    elif scale == "atanh" and -1 < estimate < 1:
        centre = math.atanh(estimate)
        spread = q * std_error / (1 - estimate ** 2)
        low, high = math.tanh(centre - spread), math.tanh(centre + spread)
    else:
        low, high = estimate - q * std_error, estimate + q * std_error
        if scale == "logit":
            low, high = max(low, 0.0), min(high, 1.0)
        if scale == "atanh":
            low, high = max(low, -1.0), min(high, 1.0)
    return {"est": estimate, "se": std_error, "low": low, "high": high, "p": p_value}


def holm_adjust(p_values):
    """Holm step down adjustment for a family of p values. NaN values are left as NaN."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    order = valid[np.argsort(p[valid], kind="stable")]
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(order) - rank) * p[index]))
        adjusted[index] = running_max
    return adjusted


def compare_scorers(y, minutes, scorers):
    """Intervals for every scorer and for the difference of every pair, all on the same test rows.
    scorers maps a label to (probabilities, threshold)."""
    unique_minutes, minute_index = np.unique(minutes, return_inverse=True)
    n_minutes = len(unique_minutes)
    jackknifed = {label: jackknife_all_metrics(RankedScores(y, prob, minute_index), threshold, n_minutes)
                  for label, (prob, threshold) in scorers.items()}

    single = {}
    for label, (full, pseudo) in jackknifed.items():
        single[label] = {}
        for metric in METRICS:
            scale = "atanh" if metric == "mcc" else "logit"
            single[label][metric] = summarise(full[metric], pseudo[metric], scale)

    differences = {}
    labels = list(scorers)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            full_a, pseudo_a = jackknifed[a]
            full_b, pseudo_b = jackknifed[b]
            differences[(a, b)] = {metric: summarise(full_a[metric] - full_b[metric], pseudo_a[metric] - pseudo_b[metric])
                                   for metric in TESTED_METRICS}
    return single, differences, n_minutes


def interval_text(summary, digits=4):
    return f"{summary['est']:+.{digits}f} [{summary['low']:+.{digits}f}, {summary['high']:+.{digits}f}]"

# COMMAND ----------

# Final training and evaluation

TRAINED = {}      # model name -> list of runs, one per seed
RESULTS = {}      # model name -> evaluation of the first seed plus the spread over seeds


def train_model(name):
    """Train one model per seed with its tuned (or default) hyperparameters and its own maximum epochs."""
    params = params_for(name)
    print(f"\n==== training {name} | max epochs {MAX_EPOCHS[name]} | patience {PATIENCE} | seeds {SEEDS} ====")
    print(f"hyperparameters: {params}")
    runs = []
    for seed in SEEDS:
        print(f"seed {seed}")
        run = train_one_run(name, params, MAX_EPOCHS[name], seed)
        run["tuned"] = name in TUNED_PARAMS
        path = os.path.join(OUT_DIR, f"{FILE_TAG}_{name}_seed{seed}.pt")
        torch.save(run["state"], path)
        print(f"best epoch {run['best_epoch']}, checkpoint saved to {path}")
        runs.append(run)
    TRAINED[name] = runs
    return runs


def predict_all_splits(run):
    """Load the best weights and run train, val and test in one continuous pass."""
    model = build_model(run["name"], run["params"])
    model.load_state_dict(run["state"])
    model.eval()
    model.reset_state()
    results = {
        "train": run_sequence(model, TRAIN_STEPS, collect=True),
        "val": run_sequence(model, VAL_STEPS, collect=True),
        "test": run_sequence(model, TEST_STEPS, collect=True)
    }
    del model
    return results


def evaluate_model(name):
    assert name in TRAINED, f"run train_model('{name}') first"
    print(f"\n==== evaluating {name} ====")
    seed_rows = []
    first = None
    for run in TRAINED[name]:
        results = predict_all_splits(run)
        threshold, objective_value = pick_threshold(results["val"]["y"], results["val"]["prob"], THRESHOLD_OBJECTIVE)
        metrics = {split: compute_metrics(result["y"], result["prob"], threshold) for split, result in results.items()}
        seed_rows.append({"seed": run["seed"], "best_epoch": run["best_epoch"], "threshold": threshold,
                          "test_pr_auc": metrics["test"]["pr_auc"], "test_roc_auc": metrics["test"]["roc_auc"],
                          "test_f1": metrics["test"]["f1"], "test_balanced_accuracy": metrics["test"]["balanced_accuracy"]})
        if first is None:
            first = (run, results, threshold, objective_value, metrics)

    run, results, threshold, objective_value, metrics = first
    test = results["test"]
    single, differences, n_minutes = compare_scorers(test["y"], test["minute"], {"model": (test["prob"], threshold),
                                                                                  "persistence": (test["persistence"], 0.5)})
    versus_persistence = differences[("model", "persistence")]
    adjusted = holm_adjust([versus_persistence[metric]["p"] for metric in TESTED_METRICS])
    for metric, p_holm in zip(TESTED_METRICS, adjusted):
        versus_persistence[metric]["p_holm"] = float(p_holm)

    RESULTS[name] = {
        "run": run,
        "threshold": threshold,
        "objective_value": objective_value,
        "metrics": metrics,
        "val": results["val"],
        "test": test,
        "inference": {"minutes": n_minutes, "model": single["model"], "persistence": single["persistence"],
                      "versus_persistence": versus_persistence},
        "seeds": pd.DataFrame(seed_rows)
    }
    print_evaluation(name)
    return RESULTS[name]


def print_evaluation(name):
    result = RESULTS[name]
    run, metrics, inference = result["run"], result["metrics"], result["inference"]
    print(f"seed {run['seed']}, best epoch {run['best_epoch']}, threshold {result['threshold']:.4f} "
          f"(maximises validation {THRESHOLD_OBJECTIVE} = {result['objective_value']:.4f})")
    columns = ["precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "pr_auc", "roc_auc", "positive_rate"]
    print(pd.DataFrame(metrics).T[columns].round(4).to_string())

    print(f"\ntest window: {inference['minutes']} minutes, t({ewc_terms(inference['minutes'])}) intervals")
    rows = []
    for metric in METRICS:
        rows.append({"metric": metric, "model": interval_text(inference["model"][metric]),
                     "persistence": interval_text(inference["persistence"][metric])})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nmodel minus persistence (positive means the model is better, p_holm adjusts for the four metrics):")
    rows = [{"metric": metric, "difference": interval_text(d), "p": round(d["p"], 4), "p_holm": round(d["p_holm"], 4)}
            for metric, d in inference["versus_persistence"].items()]
    print(pd.DataFrame(rows).to_string(index=False))

    if len(result["seeds"]) > 1:
        seeds = result["seeds"]
        print(f"\nspread over {len(seeds)} seeds:")
        print(f"  test PR-AUC {seeds.test_pr_auc.mean():.4f} +/- {seeds.test_pr_auc.std(ddof=1):.4f}")
        balanced = seeds.test_balanced_accuracy
        print(f"  test balanced accuracy {balanced.mean():.4f} +/- {balanced.std(ddof=1):.4f}")
        print(seeds.round(4).to_string(index=False))


def print_diagnosis(name):
    """Conclusions from the tests, followed by rules of thumb that are not tests."""
    result = RESULTS[name]
    for metric, label in (("pr_auc", "PR-AUC"), ("balanced_accuracy", "Balanced accuracy")):
        d = result["inference"]["versus_persistence"][metric]
        if not np.isfinite(d["p"]):
            print(f"{label}: identical to persistence.")
        elif d["p_holm"] < 0.05:
            direction = "higher" if d["est"] > 0 else "lower"
            print(f"{label} is {direction} than persistence: {interval_text(d)}, Holm p = {d['p_holm']:.3g}.")
        else:
            print(f"{label} is not distinguishable from persistence: {interval_text(d)}, Holm p = {d['p_holm']:.3g}.")

    notes = []
    metrics = result["metrics"]
    if result["run"]["best_epoch"] == MAX_EPOCHS[name]:
        notes.append("The best epoch is the last allowed epoch, so a higher maximum may help.")
    if metrics["train"]["balanced_accuracy"] - metrics["test"]["balanced_accuracy"] > 0.05:
        notes.append("Train balanced accuracy is more than 0.05 above test. "
                     "This can be overfitting or drift between periods.")
    if metrics["test"]["recall"] < 0.5:
        notes.append(f"Only {metrics['test']['recall']:.0%} of real overloads are caught at this threshold. "
                     "Try threshold_objective f2.")
    print("Rules of thumb (not tests): " + (" ".join(notes) if notes else "none triggered."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Chart functions
# MAGIC Each function draws one chart. Each chart is called from its own cell in the model sections.
# MAGIC The confusion matrix shows the count and the share of the actual class in every cell, so each row adds up to 100%.

# COMMAND ----------

# Charts for one model

COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c", "model": "#1f77b4"}


def show(fig, name, tag):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{FILE_TAG}_{name}_{tag}.png"), dpi=130, bbox_inches="tight")
    display(fig)
    plt.close(fig)


def integer_epochs(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)


def plot_tuning(name):
    """Best validation PR-AUC of every tuning trial."""
    if name not in TUNING_RESULTS:
        print(f"no tuning results for {name} (run_tuning is off or the tuning cell was skipped)")
        return
    table = TUNING_RESULTS[name].sort_values("trial")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.trial.astype(str), table.val_pr_auc, color=COLORS["model"])
    best = TUNING_RESULTS[name].loc[0]
    ax.bar(str(best.trial), best.val_pr_auc, color=COLORS["test"], label="selected")
    ax.axhline(POSITIVE_RATE["val"], ls="dotted", c="grey", label="chance (validation overload rate)")
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation PR-AUC")
    ax.set_title(f"{name}: hyperparameter search")
    ax.legend()
    show(fig, name, "tuning")


def plot_loss_curves(name):
    history = pd.DataFrame(RESULTS[name]["run"]["history"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.epoch, history.train_loss, marker="o", color=COLORS["train"], label="train loss")
    ax.plot(history.epoch, history.val_loss, marker="o", color=COLORS["val"], label="validation loss")
    ax.axvline(RESULTS[name]["run"]["best_epoch"], ls="dashed", c="grey", label="selected epoch")
    ax.set_ylabel(f"loss ({IMBALANCE})")
    ax.set_title(f"{name}: train and validation loss")
    ax.legend()
    integer_epochs(ax)
    show(fig, name, "loss_curves")


def plot_pr_auc_curves(name):
    history = pd.DataFrame(RESULTS[name]["run"]["history"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.epoch, history.train_pr_auc, marker="o", color=COLORS["train"], label="train PR-AUC")
    ax.plot(history.epoch, history.val_pr_auc, marker="o", color=COLORS["val"], label="validation PR-AUC")
    ax.axhline(POSITIVE_RATE["val"], ls="dotted", c="grey", label="chance (validation overload rate)")
    ax.axvline(RESULTS[name]["run"]["best_epoch"], ls="dashed", c="grey", alpha=0.6)
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"{name}: train and validation PR-AUC")
    ax.legend()
    integer_epochs(ax)
    show(fig, name, "pr_auc_curves")


def plot_split_metrics(name):
    """Precision, recall, F1 and balanced accuracy on each split at the validation threshold."""
    metrics = RESULTS[name]["metrics"]
    names = ["precision", "recall", "f1", "balanced_accuracy"]
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, split in enumerate(("train", "val", "test")):
        values = [metrics[split][m] for m in names]
        bars = ax.bar(np.arange(len(names)) + i * width, values, width, color=COLORS[split], label=split)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(names)) + width)
    ax.set_xticklabels(["Precision", "Recall", "F1", "Balanced accuracy"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}: metrics by split at threshold {RESULTS[name]['threshold']:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    show(fig, name, "split_metrics")


def plot_confusion_matrix(name):
    """One test confusion matrix. Each cell shows the count and the share of the actual class."""
    m = RESULTS[name]["metrics"]["test"]
    counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    shares = counts / np.maximum(row_totals, 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(counts[i, j]):,}\n{shares[i, j]:.1%}", ha="center", va="center", fontsize=13,
                    color="white" if shares[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted normal", "Predicted overload"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Actual normal\n(n={int(row_totals[0, 0]):,})", f"Actual overload\n(n={int(row_totals[1, 0]):,})"])
    ax.set_title(f"{name}: test confusion matrix\nEach row adds up to 100%")
    ax.set_xlabel(f"recall {m['recall']:.1%} | specificity {m['specificity']:.1%} | precision {m['precision']:.1%} | "
                  f"balanced accuracy {m['balanced_accuracy']:.1%}", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of the actual class")
    show(fig, name, "confusion_matrix")


def plot_pr_curve(name):
    test = RESULTS[name]["test"]
    m = RESULTS[name]["metrics"]["test"]
    precision, recall, _ = precision_recall_curve(test["y"], test["prob"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(recall, precision, label=f"{name} (PR-AUC {m['pr_auc']:.3f})")
    ax.axhline(m["positive_rate"], ls="dotted", c="grey", label=f"chance ({m['positive_rate']:.3f})")
    ax.scatter([m["recall"]], [m["precision"]], c="red", zorder=5, label="chosen threshold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{name}: test precision and recall")
    ax.legend()
    ax.grid(alpha=0.25)
    show(fig, name, "pr_curve")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. TGCN

# COMMAND ----------

# 10a. TGCN hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGCN")

# COMMAND ----------

# 10b. TGCN training
train_model("TGCN")

# COMMAND ----------

# 10c. TGCN evaluation
evaluate_model("TGCN")
print_diagnosis("TGCN")

# COMMAND ----------

# 10d. TGCN chart: hyperparameter search
plot_tuning("TGCN")

# COMMAND ----------

# 10e. TGCN chart: train and validation loss
plot_loss_curves("TGCN")

# COMMAND ----------

# 10f. TGCN chart: train and validation PR-AUC
plot_pr_auc_curves("TGCN")

# COMMAND ----------

# 10g. TGCN chart: metrics by split
plot_split_metrics("TGCN")

# COMMAND ----------

# 10h. TGCN chart: confusion matrix
plot_confusion_matrix("TGCN")

# COMMAND ----------

# 10i. TGCN chart: precision and recall curve
plot_pr_curve("TGCN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. TGN

# COMMAND ----------

# 11a. TGN hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGN")

# COMMAND ----------

# 11b. TGN training
train_model("TGN")

# COMMAND ----------

# 11c. TGN evaluation
evaluate_model("TGN")
print_diagnosis("TGN")

# COMMAND ----------

# 11d. TGN chart: hyperparameter search
plot_tuning("TGN")

# COMMAND ----------

# 11e. TGN chart: train and validation loss
plot_loss_curves("TGN")

# COMMAND ----------

# 11f. TGN chart: train and validation PR-AUC
plot_pr_auc_curves("TGN")

# COMMAND ----------

# 11g. TGN chart: metrics by split
plot_split_metrics("TGN")

# COMMAND ----------

# 11h. TGN chart: confusion matrix
plot_confusion_matrix("TGN")

# COMMAND ----------

# 11i. TGN chart: precision and recall curve
plot_pr_curve("TGN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. TGAT

# COMMAND ----------

# 12a. TGAT hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGAT")

# COMMAND ----------

# 12b. TGAT training
train_model("TGAT")

# COMMAND ----------

# 12c. TGAT evaluation
evaluate_model("TGAT")
print_diagnosis("TGAT")

# COMMAND ----------

# 12d. TGAT chart: hyperparameter search
plot_tuning("TGAT")

# COMMAND ----------

# 12e. TGAT chart: train and validation loss
plot_loss_curves("TGAT")

# COMMAND ----------

# 12f. TGAT chart: train and validation PR-AUC
plot_pr_auc_curves("TGAT")

# COMMAND ----------

# 12g. TGAT chart: metrics by split
plot_split_metrics("TGAT")

# COMMAND ----------

# 12h. TGAT chart: confusion matrix
plot_confusion_matrix("TGAT")

# COMMAND ----------

# 12i. TGAT chart: precision and recall curve
plot_pr_curve("TGAT")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Model comparison on the test window
# MAGIC Persistence gives a 0 or 1 score, so its PR-AUC and ROC-AUC describe a single operating point.
# MAGIC Run this after every model you want to compare has been evaluated.

# COMMAND ----------

# Comparison tables


def comparison_tables():
    """Intervals for persistence and every model, and every pairwise difference, all on the same test rows."""
    first = next(iter(RESULTS.values()))
    for result in RESULTS.values():
        same_minutes = np.array_equal(result["test"]["minute"], first["test"]["minute"])
        same_labels = np.array_equal(result["test"]["y"], first["test"]["y"])
        assert same_minutes and same_labels, "the models were scored on different rows"
    scorers = {"Persistence": (first["test"]["persistence"], 0.5)}
    for name, result in RESULTS.items():
        scorers[name] = (result["test"]["prob"], result["threshold"])
    single, differences, n_minutes = compare_scorers(first["test"]["y"], first["test"]["minute"], scorers)

    rows = []
    for label, (_, threshold) in scorers.items():
        row = {"model": label, "threshold": threshold}
        for metric in METRICS:
            summary = single[label][metric]
            row[metric] = summary["est"]
            row[f"{metric}_low"] = summary["low"]
            row[f"{metric}_high"] = summary["high"]
        rows.append(row)
    table = pd.DataFrame(rows)

    pair_rows = []
    for (a, b), by_metric in differences.items():
        for metric, summary in by_metric.items():
            pair_rows.append({"a": a, "b": b, "metric": metric, "difference": summary["est"],
                              "low": summary["low"], "high": summary["high"], "p": summary["p"]})
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = np.nan
    for metric in TESTED_METRICS:
        selected = pairs.metric == metric
        pairs.loc[selected, "p_holm"] = holm_adjust(pairs.loc[selected, "p"].to_numpy())
    return table, pairs, n_minutes


def hyperparameter_table():
    rows = []
    for name in RESULTS:
        run = RESULTS[name]["run"]
        rows.append({"model": name, "tuned": run["tuned"], "max_epochs": MAX_EPOCHS[name], "best_epoch": run["best_epoch"],
                     "epochs_run": run.get("epochs_run"), "train_seconds": round(run.get("train_seconds", float("nan")), 1),
                     "threshold": RESULTS[name]["threshold"], **run["params"]})
    return pd.DataFrame(rows)


COMPARISON, PAIRS, TEST_MINUTES = comparison_tables()
HYPERPARAMETERS = hyperparameter_table()
SEED_SPREAD = pd.concat([result["seeds"].assign(model=name) for name, result in RESULTS.items()], ignore_index=True)

shown_columns = ["model", "threshold"]
for metric in ("pr_auc", "roc_auc", "f1", "balanced_accuracy", "precision", "recall"):
    shown_columns += [metric, f"{metric}_low", f"{metric}_high"]
print(f"Test window: {TEST_MINUTES} minutes, t({ewc_terms(TEST_MINUTES)}) intervals")
print(COMPARISON[shown_columns].round(4).to_string(index=False))
print("\nPairwise differences (a minus b, positive means a is better, Holm adjusted within each metric):")
print(PAIRS.round(4).to_string(index=False))
print("\nHyperparameters, selected epochs and thresholds:")
print(HYPERPARAMETERS.to_string(index=False))
if len(SEEDS) > 1:
    print("\nSpread over seeds:")
    spread = SEED_SPREAD.groupby("model")[["test_pr_auc", "test_balanced_accuracy", "test_f1"]].agg(["mean", "std"])
    print(spread.round(4).to_string())

# COMMAND ----------

# Comparison chart: test metrics with 95% intervals

shown_metrics = ["pr_auc", "roc_auc", "f1", "balanced_accuracy"]
width = 0.8 / len(COMPARISON)
fig, ax = plt.subplots(figsize=(10, 5))
for i, row in COMPARISON.reset_index(drop=True).iterrows():
    values = np.array([row[m] for m in shown_metrics])
    lows = np.array([row[f"{m}_low"] for m in shown_metrics])
    highs = np.array([row[f"{m}_high"] for m in shown_metrics])
    ax.bar(np.arange(len(shown_metrics)) + i * width, values, width, yerr=np.vstack([values - lows, highs - values]),
           capsize=3, label=row["model"])
ax.set_xticks(np.arange(len(shown_metrics)) + width * (len(COMPARISON) - 1) / 2)
ax.set_xticklabels(["PR-AUC", "ROC-AUC", "F1", "Balanced accuracy"])
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
ax.set_title("Test metrics with 95% intervals")
show(fig, "comparison", "test_metrics")

# COMMAND ----------

# Comparison chart: validation PR-AUC per epoch for every model

fig, ax = plt.subplots(figsize=(8, 4.8))
for name, result in RESULTS.items():
    history = pd.DataFrame(result["run"]["history"])
    line, = ax.plot(history.epoch, history.val_pr_auc, marker="o", label=f"{name} validation")
    ax.plot(history.epoch, history.train_pr_auc, ls="dashed", alpha=0.6, color=line.get_color(), label=f"{name} train")
ax.set_ylabel("PR-AUC")
ax.set_title("PR-AUC per epoch, first seed (dashed lines are training)")
ax.legend(fontsize=8)
integer_epochs(ax)
show(fig, "comparison", "pr_auc_by_epoch")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Edge predictions for the load balancing notebook
# MAGIC
# MAGIC The evaluation above scores only the edges that appear in a minute. Whether an edge appears in the 0.5% call graph sample
# MAGIC depends on that same minute's traffic, so a controller cannot know it in advance. For the export, every regular edge
# MAGIC (seen in at least `min_obs` training minutes) is scored at every test minute from what was known one minute earlier:
# MAGIC its features in the previous minute if it appeared, otherwise zeros with `has_prev = 0`. An edge that does not appear
# MAGIC in the minute gets label 0. The model was trained on present edges only, so its threshold is chosen again on the
# MAGIC validation minutes of this same population (every regular edge, present or not). `objective_value` is that validation
# MAGIC score, and the load balancing notebook picks the classifier with it. No test minute is used for either choice.

# COMMAND ----------

# Edge prediction export

CLASSIFIER_HORIZON = 1
train_rows = edge_table[edge_table.t_idx <= TRAIN_RANGE[1]]
known_edges = (
    train_rows.groupby(["src_node_id", "dst_node_id", "rpctype"])
    .agg(train_minutes=("t_idx", "size"), rpctype_idx=("rpctype_idx", "first"))
    .reset_index()
)
known_edges = known_edges[known_edges.train_minutes >= MIN_OBS].reset_index(drop=True)
known_position = {key: position for position, key in enumerate(zip(known_edges.src_node_id, known_edges.dst_node_id, known_edges.rpctype))}
row_position = np.array([known_position.get(key, -1) for key in zip(edge_table.src_node_id, edge_table.dst_node_id, edge_table.rpctype)])
known_rpc = np.eye(N_RPC, dtype=np.float32)[np.clip(np.nan_to_num(known_edges.rpctype_idx.to_numpy()).astype(int), 0, N_RPC - 1)]
known_src = torch.from_numpy(known_edges.src_node_id.to_numpy(np.int64))
known_dst = torch.from_numpy(known_edges.dst_node_id.to_numpy(np.int64))
print(f"regular edges scored at every test minute: {len(known_edges):,}")


def known_edge_target(t):
    """Target for every regular edge at minute t, built only from minute t - 1. Labels come from minute t."""
    previous_features = np.zeros((len(known_edges), current_features.shape[1]), dtype=np.float32)
    has_previous = np.zeros(len(known_edges), dtype=np.float32)
    start, end = edge_rows.get(t - 1, (0, 0))
    positions = row_position[start:end]
    keep = positions >= 0
    previous_features[positions[keep]] = current_features[start:end][keep]
    has_previous[positions[keep]] = 1.0
    labels = np.zeros(len(known_edges), dtype=np.float32)
    present = np.zeros(len(known_edges), dtype=np.int64)
    start, end = edge_rows.get(t, (0, 0))
    positions = row_position[start:end]
    keep = positions >= 0
    labels[positions[keep]] = LABELS[start:end][keep]
    present[positions[keep]] = 1
    features = np.concatenate([previous_features, has_previous[:, None], known_rpc], axis=1)
    target = {"t": t, "src": known_src, "dst": known_dst, "features": torch.from_numpy(features)}
    return target, labels, present


def score_known_edges(run):
    """Replay train, validation and test in order with the best weights and score every regular edge at each validation and test minute."""
    model = build_model(run["name"], run["params"])
    model.load_state_dict(run["state"])
    model.eval()
    model.reset_state()
    split_of_minute = {**{t: "val" for t in VAL_STEPS}, **{t: "test" for t in TEST_STEPS}}
    parts = []
    with torch.no_grad():
        for t in TRAIN_STEPS + VAL_STEPS + TEST_STEPS:
            if t in split_of_minute:
                target, labels, present = known_edge_target(t)
                probability = torch.sigmoid(model.predict(to_device(target))).cpu().numpy()
                parts.append(pd.DataFrame({
                    "src_node_id": known_edges.src_node_id.to_numpy(), "dst_node_id": known_edges.dst_node_id.to_numpy(),
                    "rpctype": known_edges.rpctype.to_numpy(), "t_idx": t, "probability": probability,
                    "label": labels, "present": present, "split": split_of_minute[t]
                }))
            model.update(to_device(STEPS[t]))
    del model
    return pd.concat(parts, ignore_index=True)


node_names = node_index.select(F.col("node_id").cast("long").alias("node_id"), F.col("node_key").alias("msname")).toPandas()
caller_names = node_names.rename(columns={"node_id": "src_node_id", "msname": "caller"})
callee_names = node_names.rename(columns={"node_id": "dst_node_id", "msname": "callee"})

prediction_parts = []
for name, result in RESULTS.items():
    scored = score_known_edges(result["run"])
    validation_rows = scored[scored.split == "val"]
    population_threshold, population_objective = pick_threshold(validation_rows.label, validation_rows.probability, THRESHOLD_OBJECTIVE)
    scored = scored.assign(
        model=name,
        horizon=CLASSIFIER_HORIZON,
        threshold=float(population_threshold),
        present_edge_threshold=float(result["threshold"]),
        objective=THRESHOLD_OBJECTIVE,
        objective_value=float(population_objective)
    )
    print(f"{name}: threshold on all regular edges {population_threshold:.4f} (on present edges {result['threshold']:.4f}), "
          f"validation {THRESHOLD_OBJECTIVE} {population_objective:.4f}")
    scored["predicted_overload"] = (scored.probability >= scored.threshold).astype(int)
    prediction_parts.append(scored)
EDGE_PREDICTIONS = (
    pd.concat(prediction_parts, ignore_index=True)
    .merge(caller_names, on="src_node_id", how="left")
    .merge(callee_names, on="dst_node_id", how="left")
)
assert EDGE_PREDICTIONS[["caller", "callee"]].notna().all().all(), "every edge must map to a service name"

EDGE_PREDICTION_PATH = f"{FE_ROOT}/edge_overload_predictions"
spark.createDataFrame(EDGE_PREDICTIONS).write.mode("overwrite").parquet(EDGE_PREDICTION_PATH)
print(f"{len(EDGE_PREDICTIONS):,} edge prediction rows written to {EDGE_PREDICTION_PATH}")
print(EDGE_PREDICTIONS.groupby(["model", "split"]).agg(rows=("label", "size"), present_rate=("present", "mean"), overload_rate=("label", "mean"),
                                            flagged_rate=("predicted_overload", "mean")).round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Two evaluation settings
# MAGIC
# MAGIC **Active edges** (section 13) are the edges called in minute `t`. Knowing that an edge is called is information from minute `t`,
# MAGIC so that setting is conditional classification. **Regular edges** (this cell) are every edge seen in at least `min_obs` training
# MAGIC minutes, scored before minute `t`. An edge that is not called gets label 0. This is the prospective setting that the load balancing
# MAGIC notebook uses. The models were trained on active edges, so the two settings are reported side by side and never pooled.
# MAGIC In both settings the label means a high sampled call count compared with the edge's own training history. It is not a measured CPU shortage.

# COMMAND ----------

# Two evaluation settings

key_columns = ["t_idx", "src_node_id", "dst_node_id", "rpctype"]
first_model = next(iter(RESULTS))
first_rows = EDGE_PREDICTIONS[EDGE_PREDICTIONS.model == first_model][key_columns + ["label"]]
previous_label = first_rows.assign(t_idx=first_rows.t_idx + 1).rename(columns={"label": "previous_label"})
reference = (
    EDGE_PREDICTIONS[(EDGE_PREDICTIONS.model == first_model) & (EDGE_PREDICTIONS.split == "test")]
    .sort_values(key_columns).reset_index(drop=True)
    .merge(previous_label, on=key_columns, how="left").fillna({"previous_label": 0.0})
)
prospective_scorers = {"Persistence": (reference.previous_label.to_numpy(), 0.5)}
for name in RESULTS:
    rows = EDGE_PREDICTIONS[(EDGE_PREDICTIONS.model == name) & (EDGE_PREDICTIONS.split == "test")].sort_values(key_columns).reset_index(drop=True)
    assert np.array_equal(rows[key_columns + ["label"]].to_numpy(), reference[key_columns + ["label"]].to_numpy()), "models must score the same rows"
    prospective_scorers[name] = (rows.probability.to_numpy(), float(rows.threshold.iloc[0]))
prospective_single, prospective_pairs, _ = compare_scorers(reference.label.to_numpy(), reference.t_idx.to_numpy(), prospective_scorers)

setting_rows = []
for _, row in COMPARISON.iterrows():
    entry = {"setting": "active edges", "model": row["model"], "rows": int(len(RESULTS[first_model]["test"]["y"])),
             "positive_rate": POSITIVE_RATE["test"]}
    for metric in ("pr_auc", "roc_auc", "f1", "balanced_accuracy", "precision", "recall"):
        entry[metric], entry[f"{metric}_low"], entry[f"{metric}_high"] = row[metric], row[f"{metric}_low"], row[f"{metric}_high"]
    setting_rows.append(entry)
for label, by_metric in prospective_single.items():
    entry = {"setting": "regular edges", "model": label, "rows": len(reference), "positive_rate": float(reference.label.mean())}
    for metric in ("pr_auc", "roc_auc", "f1", "balanced_accuracy", "precision", "recall"):
        entry[metric], entry[f"{metric}_low"], entry[f"{metric}_high"] = by_metric[metric]["est"], by_metric[metric]["low"], by_metric[metric]["high"]
    setting_rows.append(entry)
TWO_SETTINGS = pd.DataFrame(setting_rows)
print(TWO_SETTINGS[["setting", "model", "rows", "positive_rate", "pr_auc", "roc_auc", "f1", "balanced_accuracy", "precision", "recall"]]
      .round(4).to_string(index=False))
PROSPECTIVE_PAIRS = pd.DataFrame([{"a": a, "b": b, "metric": metric, "difference": summary["est"], "low": summary["low"],
                                   "high": summary["high"], "p": summary["p"]}
                                  for (a, b), by_metric in prospective_pairs.items() for metric, summary in by_metric.items()])
PROSPECTIVE_PAIRS["p_holm"] = np.nan
for metric in TESTED_METRICS:
    selected = PROSPECTIVE_PAIRS.metric == metric
    PROSPECTIVE_PAIRS.loc[selected, "p_holm"] = holm_adjust(PROSPECTIVE_PAIRS.loc[selected, "p"].to_numpy())
print("\nRegular edges, pairwise differences (a minus b, Holm adjusted within each metric):")
print(PROSPECTIVE_PAIRS.round(4).to_string(index=False))
print("\nPR-AUC depends on the positive rate, so compare it with the positive rate of the same setting, not across settings.")

fig, ax = plt.subplots(figsize=(9, 4.8))
labels = list(prospective_scorers)
positions = np.arange(len(labels))
for offset, setting in ((-0.2, "active edges"), (0.2, "regular edges")):
    part = TWO_SETTINGS[TWO_SETTINGS.setting == setting].set_index("model").reindex(labels)
    errors = np.vstack([part.pr_auc - part.pr_auc_low, part.pr_auc_high - part.pr_auc]).clip(min=0)
    ax.bar(positions + offset, part.pr_auc, 0.38, yerr=np.nan_to_num(errors), capsize=3, label=setting)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_xlabel("Scorer")
ax.set_ylabel("Test PR-AUC")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("PR-AUC by Evaluation Setting")
show(fig, "comparison", "pr_auc_by_setting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Save results

# COMMAND ----------

# Save

summary = {
    "settings": {
        "label_quantile": LABEL_QUANTILE,
        "min_obs": MIN_OBS,
        "imbalance": IMBALANCE,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "seeds": SEEDS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "tuning": RUN_TUNING,
        "tuning_trials": TUNING_TRIALS,
        "tuning_max_epochs": TUNING_MAX_EPOCHS
    },
    "positive_rate": POSITIVE_RATE,
    "hyperparameters": HYPERPARAMETERS.to_dict(orient="records"),
    "comparison": COMPARISON.to_dict(orient="records"),
    "pairwise_tests": PAIRS.to_dict(orient="records"),
    "two_settings": TWO_SETTINGS.to_dict(orient="records"),
    "regular_edge_pairwise_tests": PROSPECTIVE_PAIRS.to_dict(orient="records"),
    "models": {
        name: {
            "best_epoch": result["run"]["best_epoch"],
            "threshold": result["threshold"],
            "history": result["run"]["history"],
            "metrics": result["metrics"],
            "inference": result["inference"],
            "seeds": result["seeds"].to_dict(orient="records")
        }
        for name, result in RESULTS.items()
    }
}
summary_json = json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))

with open(os.path.join(OUT_DIR, f"{FILE_TAG}_results.json"), "w") as f:
    f.write(summary_json)
COMPARISON.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_comparison.csv"), index=False)
PAIRS.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_pairwise_tests.csv"), index=False)
TWO_SETTINGS.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_two_settings.csv"), index=False)
HYPERPARAMETERS.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_hyperparameters.csv"), index=False)
SEED_SPREAD.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_seeds.csv"), index=False)
for name, tuning_table in TUNING_RESULTS.items():
    tuning_table.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_{name}_tuning.csv"), index=False)
print(f"results saved to {OUT_DIR}")

try:
    spark.createDataFrame([(summary_json,)], ["json"]).coalesce(1).write.mode("overwrite").text(f"{FE_ROOT}/anomaly_results")
    print(f"also written to {FE_ROOT}/anomaly_results")
except Exception as error:
    print(f"could not write to {FE_ROOT}: {str(error)[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC
# MAGIC **Epochs and tuning.** Set `tgcn_max_epochs`, `tgn_max_epochs` and `tgat_max_epochs` separately.
# MAGIC Early stopping ends a run when the validation PR-AUC has not improved for `patience` epochs. The best epoch is kept.
# MAGIC Each tuning trial runs for at most the model's own tuning epochs (`tgcn_tuning_epochs` and so on).
# MAGIC Short trials can favour configurations that learn fast, so raise the tuning epochs if the selected learning rate looks too high.
# MAGIC Edit `SEARCH_SPACE` to change what the search can pick. Set `run_tuning` to `no` to use `DEFAULT_PARAMS`.
# MAGIC
# MAGIC **Class imbalance.** It is handled by the loss (`imbalance`), by the threshold chosen on validation (`threshold_objective`) and by
# MAGIC metrics that are not dominated by the normal class: recall, specificity, balanced accuracy, MCC and PR-AUC. A random scorer has a
# MAGIC PR-AUC equal to the overload rate, not 0.5. `balanced_accuracy` and `f2` catch more overloads at the cost of more false alarms.
# MAGIC
# MAGIC **What the tests cover.** The intervals describe uncertainty from the finite test window. They do not include training randomness,
# MAGIC which the `seeds` widget measures, or the choice of hyperparameters, epoch and threshold, which are fixed on validation first.
# MAGIC Read `p_holm` in the pairwise table.
# MAGIC
# MAGIC **Label.** The label is statistical: a call count above the edge's own training quantile. An edge whose training counts are almost
# MAGIC constant gets a threshold equal to that constant, so any increase counts as an overload. For a capacity based definition change only
# MAGIC the `label` column in section 3.
# MAGIC
# MAGIC **Neighbour lists.** TGN and TGAT keep exactly the latest `neighbor_size` edges of each node through `RecentNeighborLoader`.
# MAGIC
# MAGIC **Memory.** The driver holds the labelled edge table (about 6.2 million rows). Use a high memory driver and a GPU for TGN and TGAT.
# MAGIC TGCN backpropagates through `tbptt` minutes. TGN backpropagates through one minute because the PyG memory is updated in place.
# MAGIC TGAT carries no state between minutes.