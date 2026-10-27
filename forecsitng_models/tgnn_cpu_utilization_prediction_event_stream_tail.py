# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "torch",
#   "torch_geometric",
# ]
# ///
# MAGIC %md
# MAGIC # CPU utilization forecasting with TGCN, TGN, TGAT, DCRNN and GraphMixer from the event stream
# MAGIC
# MAGIC This notebook predicts the CPU utilization of each microservice a few minutes ahead.
# MAGIC It reads the output of `event_stream_fe_pipeline`. The table `events` has one row per call with its exact time in milliseconds.
# MAGIC The table `node_feat_timeline` has the per minute service features, which are the same as in the snapshot pipeline.
# MAGIC Node ids come from the same `node_index`, so the results can be compared directly with the snapshot notebook.
# MAGIC The target, the split, the baselines, the tuning and the tests are the same. Only the graph input differs.
# MAGIC
# MAGIC **Prediction setup.** A target minute is predicted only from minutes that end `horizon` minutes earlier.
# MAGIC Each model predicts the last observed value plus a learned correction, so an untrained model equals persistence.
# MAGIC
# MAGIC **How the models use the events.** TGCN builds one graph per minute from the unique caller and callee pairs of that minute.
# MAGIC TGN updates the memory of both services at every call, in time order. TGAT attends over the latest calls of each service
# MAGIC and of its neighbours, with an encoding of how long ago each call happened.
# MAGIC DCRNN builds one weighted directed graph per minute and diffuses along and against the call direction inside a GRU.
# MAGIC GraphMixer mixes the latest calls of each service with an MLP-Mixer and adds the mean features of its recent neighbours.
# MAGIC
# MAGIC **Size control.** Raw call streams are large. Spark first keeps the calls that touch a service with resource data
# MAGIC (`event_filter`) and then keeps a random sample of at most `max_events_per_minute` calls per minute.
# MAGIC Section 2 prints how many calls were kept.
# MAGIC
# MAGIC **Tuning and evaluation.** Each model has its own hyperparameters. All models get the same maximum epochs by default, and the hyperparameter table reports epochs run and training time.
# MAGIC A random search on validation picks the hyperparameters and early stopping picks the epoch.
# MAGIC Test metrics come with 95% intervals that treat each test minute as one unit.
# MAGIC
# MAGIC | Section | Content |
# MAGIC |---|---|
# MAGIC | 1 | Settings and hyperparameters |
# MAGIC | 2 | Load the feature tables and the calls |
# MAGIC | 3 | Build the per minute time steps |
# MAGIC | 4 | Models |
# MAGIC | 5 | Training with early stopping |
# MAGIC | 6 | Hyperparameter search |
# MAGIC | 7 | Evaluation and statistical tests |
# MAGIC | 8 | Chart functions |
# MAGIC | 9 to 13 | TGCN, TGN, TGAT, DCRNN and GraphMixer: tune, train, evaluate, charts |
# MAGIC | 14 | Model comparison |
# MAGIC | 15 | Save results |

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

# MAGIC %pip install  torch torch_geometric

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
from scipy import stats as sstats
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings and hyperparameters

# COMMAND ----------

# Settings

# dbutils.widgets.text("fe_output_root", "s3://research-s20426/tgnn_event_dataset", "Event stream FE output folder")
# dbutils.widgets.text("artifact_dir", "", "Folder for checkpoints, charts and results (blank means a temporary folder)")
# dbutils.widgets.text("horizon", "1", "Minutes ahead to predict")
# dbutils.widgets.dropdown("loss_type", "huber", ["huber", "mse", "mae"], "Training loss")
# dbutils.widgets.text("huber_beta", "1.0", "Huber beta in scaled units")
# dbutils.widgets.text("target_median", "0.1126", "Train median of the raw target (printed by the snapshot FE)")
# dbutils.widgets.text("target_iqr", "0.1139", "Train IQR of the raw target (printed by the snapshot FE)")
# dbutils.widgets.text("unit_scale", "100", "Multiplier from the raw value to percent")
# dbutils.widgets.dropdown("event_filter", "touches_resource_node", ["touches_resource_node", "all"], "Calls to load")
# dbutils.widgets.text("max_events_per_minute", "20000", "Random cap on calls per minute after filtering (0 keeps all)")
# dbutils.widgets.text("event_batch_size", "2000", "Calls per TGN and TGAT update inside a minute")
# dbutils.widgets.text("cycle_period", "0", "Period of the time of day encoding in minutes (0 uses the span of the data)")
# dbutils.widgets.text("seeds", "42", "Training seeds separated by commas")
# dbutils.widgets.text("tgcn_max_epochs", "30", "Maximum epochs for TGCN")
# dbutils.widgets.text("tgn_max_epochs", "15", "Maximum epochs for TGN")
# dbutils.widgets.text("tgat_max_epochs", "15", "Maximum epochs for TGAT")
# dbutils.widgets.text("patience", "4", "Epochs without validation improvement before stopping")
# dbutils.widgets.dropdown("run_tuning", "yes", ["yes", "no"], "Run the hyperparameter search")
# dbutils.widgets.text("tuning_trials", "6", "Configurations tried per model")
# dbutils.widgets.text("tgcn_tuning_epochs", "10", "Maximum epochs per TGCN tuning trial")
# dbutils.widgets.text("tgn_tuning_epochs", "5", "Maximum epochs per TGN tuning trial")
# dbutils.widgets.text("tgat_tuning_epochs", "5", "Maximum epochs per TGAT tuning trial")
# dbutils.widgets.text("log_every", "100", "Print the running training loss every N minutes (0 turns it off)")


dbutils.widgets.text("fe_output_root", "s3://research-s20426/tail_features/tgnn_event_dataset", "Event stream FE output folder")
dbutils.widgets.text("node_index_root", "s3://research-s20426/tail_features/tgnn_dataset", "Snapshot FE output folder that holds node_index")
dbutils.widgets.text("artifact_dir", "", "Folder for checkpoints, charts and results (blank means a temporary folder)")
dbutils.widgets.text("horizon", "1", "Minutes ahead to predict")
dbutils.widgets.text("calibration_share", "0.4", "Share of the validation window held out for calibration")
dbutils.widgets.dropdown("loss_type", "huber", ["huber", "mse", "mae"], "Training loss")
dbutils.widgets.text("huber_beta", "1.0", "Huber beta in scaled units")
dbutils.widgets.text("target_median", "0.1126", "Train median of the raw target (printed by the snapshot FE)")
dbutils.widgets.text("target_iqr", "0.1139", "Train IQR of the raw target (printed by the snapshot FE)")
dbutils.widgets.text("unit_scale", "100", "Multiplier from the raw value to percent")
dbutils.widgets.dropdown("event_filter", "touches_resource_node", ["touches_resource_node", "all"], "Calls to load")
dbutils.widgets.text("max_events_per_minute", "20000", "Random cap on calls per minute after filtering (0 keeps all)")
dbutils.widgets.text("event_batch_size", "2000", "Calls per TGN and TGAT update inside a minute")
dbutils.widgets.text("cycle_period", "0", "Period of the time of day encoding in minutes (0 uses the span of the data)")
dbutils.widgets.text("seeds", "42", "Training seeds separated by commas")
dbutils.widgets.text("tgcn_max_epochs", "10", "Maximum epochs for TGCN")
dbutils.widgets.text("tgn_max_epochs", "10", "Maximum epochs for TGN")
dbutils.widgets.text("tgat_max_epochs", "10", "Maximum epochs for TGAT")
dbutils.widgets.text("dcrnn_max_epochs", "10", "Maximum epochs for DCRNN")
dbutils.widgets.text("graphmixer_max_epochs", "10", "Maximum epochs for GraphMixer")
dbutils.widgets.text("patience", "4", "Epochs without validation improvement before stopping")
dbutils.widgets.dropdown("run_tuning", "no", ["yes", "no"], "Run the hyperparameter search")
dbutils.widgets.text("tuning_trials", "5", "Configurations tried per model")
dbutils.widgets.text("tgcn_tuning_epochs", "5", "Maximum epochs per TGCN tuning trial")
dbutils.widgets.text("tgn_tuning_epochs", "5", "Maximum epochs per TGN tuning trial")
dbutils.widgets.text("tgat_tuning_epochs", "5", "Maximum epochs per TGAT tuning trial")
dbutils.widgets.text("dcrnn_tuning_epochs", "5", "Maximum epochs per tuning trial for DCRNN")
dbutils.widgets.text("graphmixer_tuning_epochs", "5", "Maximum epochs per tuning trial for GraphMixer")
dbutils.widgets.text("log_every", "100", "Print the running training loss every N minutes (0 turns it off)")

FE_ROOT = dbutils.widgets.get("fe_output_root")
PREVIOUS_APPROACH_ROOTS = {"s3://research-s20426/tgnn_dataset2", "s3://research-s20426/tgnn_event_dataset", "s3://research-s20426/load_balancing_work"}
assert FE_ROOT.rstrip("/") not in PREVIOUS_APPROACH_ROOTS, "This is the tail feature version. It must not write into the folders of the previous approach."
NODE_INDEX_ROOT = dbutils.widgets.get("node_index_root")
ARTIFACT_DIR = dbutils.widgets.get("artifact_dir").strip()
HORIZON = int(dbutils.widgets.get("horizon"))
CALIBRATION_SHARE = float(dbutils.widgets.get("calibration_share"))
LOSS_TYPE = dbutils.widgets.get("loss_type")
HUBER_BETA = float(dbutils.widgets.get("huber_beta"))
TARGET_MEDIAN = float(dbutils.widgets.get("target_median"))
TARGET_IQR = float(dbutils.widgets.get("target_iqr"))
UNIT_SCALE = float(dbutils.widgets.get("unit_scale"))
EVENT_FILTER = dbutils.widgets.get("event_filter")
MAX_EVENTS_PER_MINUTE = int(dbutils.widgets.get("max_events_per_minute"))
EVENT_BATCH = int(dbutils.widgets.get("event_batch_size"))
CYCLE_PERIOD_SETTING = int(dbutils.widgets.get("cycle_period"))
SEEDS = [int(s) for s in dbutils.widgets.get("seeds").split(",") if s.strip()]
MAX_EPOCHS = {
    "TGCN": int(dbutils.widgets.get("tgcn_max_epochs")),
    "TGN": int(dbutils.widgets.get("tgn_max_epochs")),
    "TGAT": int(dbutils.widgets.get("tgat_max_epochs")),
    "DCRNN": int(dbutils.widgets.get("dcrnn_max_epochs")),
    "GraphMixer": int(dbutils.widgets.get("graphmixer_max_epochs"))
}
PATIENCE = int(dbutils.widgets.get("patience"))
RUN_TUNING = dbutils.widgets.get("run_tuning") == "yes"
TUNING_TRIALS = int(dbutils.widgets.get("tuning_trials"))
TUNING_MAX_EPOCHS = {
    "TGCN": int(dbutils.widgets.get("tgcn_tuning_epochs")),
    "TGN": int(dbutils.widgets.get("tgn_tuning_epochs")),
    "TGAT": int(dbutils.widgets.get("tgat_tuning_epochs")),
    "DCRNN": int(dbutils.widgets.get("dcrnn_tuning_epochs")),
    "GraphMixer": int(dbutils.widgets.get("graphmixer_tuning_epochs"))
}
LOG_EVERY = int(dbutils.widgets.get("log_every"))

assert HORIZON >= 1, "horizon must be at least 1"
assert EVENT_BATCH >= 1, "event_batch_size must be at least 1"
assert SEEDS, "give at least one seed"

# Everything that differs between the CPU and the memory notebook is in this block.
TARGET_NAME = "CPU utilization"
TARGET_COL = "cpu_rs"
ROLL_COL = "cpu_rolling_mean"
FILE_TAG = "cpu_event"
PIPELINE = "event_stream"
EXTRA_SETTINGS = {
    "event_filter": EVENT_FILTER,
    "max_events_per_minute": MAX_EVENTS_PER_MINUTE,
    "event_batch_size": EVENT_BATCH
}

OUT_DIR = ARTIFACT_DIR or tempfile.mkdtemp(prefix=f"{FILE_TAG}_")
os.makedirs(OUT_DIR, exist_ok=True)
if not ARTIFACT_DIR:
    print("artifact_dir is blank, so results go to a temporary folder that is deleted with the session.")

UNIT_LABEL = "pp" if UNIT_SCALE == 100 else "units"
PP_PER_UNIT = TARGET_IQR * UNIT_SCALE        # one scaled unit expressed in percentage points
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"target: {TARGET_NAME} ({TARGET_COL}), horizon {HORIZON} min, loss {LOSS_TYPE}, device {DEVICE}")
print(f"calls: filter {EVENT_FILTER}, cap {MAX_EVENTS_PER_MINUTE or 'none'} per minute, update batch {EVENT_BATCH}")
print(f"maximum epochs: {MAX_EPOCHS}, patience {PATIENCE}, seeds {SEEDS}")
print(f"tuning: {'on' if RUN_TUNING else 'off'}, {TUNING_TRIALS} trials per model")
print(f"maximum epochs per tuning trial: {TUNING_MAX_EPOCHS}")

# COMMAND ----------

# Hyperparameters
#
# DEFAULT_PARAMS is used when tuning is off. It is also the first configuration tried by the search.
# SEARCH_SPACE lists the values the random search can pick for each model.

DEFAULT_PARAMS = {
    "TGCN": {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "tbptt": 4},
    "TGN": {
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1,
        "time_dim": 16, "neighbor_size": 10
    },
    "TGAT": {
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1,
        "time_dim": 16, "neighbor_size": 10
    },
    "DCRNN": {
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1,
        "tbptt": 4, "diffusion_steps": 3, "num_layers": 2
    },
    "GraphMixer": {
        "hidden_dim": 100, "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1,
        "time_dim": 100, "recent_links": 20, "node_window_calls": 2000
    }
}

SEARCH_SPACE = {
    "TGCN": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.1, 0.3],
        "tbptt": [2, 4, 8]
    },
    "TGN": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.1, 0.3],
        "time_dim": [8, 16, 32],
        "neighbor_size": [5, 10, 20]
    },
    "TGAT": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.1, 0.3],
        "time_dim": [8, 16, 32],
        "neighbor_size": [5, 10, 20]
    },
    "DCRNN": {
        "hidden_dim": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.1, 0.3],
        "tbptt": [2, 4],
        "diffusion_steps": [2, 3, 4],
        "num_layers": [1, 2]
    },
    "GraphMixer": {
        "hidden_dim": [50, 100, 200],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "dropout": [0.0, 0.1, 0.3],
        "recent_links": [10, 20, 30]
    }
}

TUNED_PARAMS = {}      # filled by tune_model()
TUNING_RESULTS = {}    # one table of trials per model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the feature tables and the calls
# MAGIC The node timeline is small and is collected as it is. The call table is filtered and sampled in Spark before it is collected.

# COMMAND ----------

# Load

from pyspark.sql import functions as F

stats_text = "\n".join(row.value for row in spark.read.text(f"{FE_ROOT}/dataset_stats").collect())
stats = json.loads(stats_text)

# The FE notebook stores the training median and IQR of the target. They convert scaled values back to utilisation,
# so they are taken from the same dataset rather than typed in by hand.
stored_scaler = stats.get("target_scalers", {}).get(TARGET_COL)
if stored_scaler:
    if abs(stored_scaler["median"] - TARGET_MEDIAN) > 1e-9 or abs(stored_scaler["iqr"] - TARGET_IQR) > 1e-9:
        print(f"target_median and target_iqr come from dataset_stats: {stored_scaler['median']:.6g} and {stored_scaler['iqr']:.6g} "
              f"(the widgets said {TARGET_MEDIAN} and {TARGET_IQR})")
    TARGET_MEDIAN = float(stored_scaler["median"])
    TARGET_IQR = float(stored_scaler["iqr"])
    PP_PER_UNIT = TARGET_IQR * UNIT_SCALE
else:
    print("dataset_stats has no target_scalers. Rerun the FE notebook, or check that the target_median and target_iqr widgets match it.")
assert "t_idx_boundaries" in stats, "dataset_stats has no t_idx_boundaries. Is fe_output_root the event stream FE output?"

NUM_NODES = int(stats["num_nodes"])
NODE_FEATS = stats["node_feature_names"]
BOUNDS = stats["t_idx_boundaries"]
RAW_EVENT_COUNTS = stats.get("event_counts", {})

assert TARGET_COL in NODE_FEATS and ROLL_COL in NODE_FEATS, "target or rolling mean column is missing from the node features"
TGT_IDX = NODE_FEATS.index(TARGET_COL)
ROLL_IDX = NODE_FEATS.index(ROLL_COL)


def read_all_splits(kind):
    """Union of the train, val and test tables of one kind (events or node_feat_timeline)."""
    frames = [spark.read.parquet(f"{FE_ROOT}/{kind}/{split}") for split in ("train", "val", "test")]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)
    return combined


node_table = read_all_splits("node_feat_timeline").select("node_id", "t_idx", *NODE_FEATS).toPandas()
node_table = node_table.sort_values(["t_idx", "node_id"], kind="stable").reset_index(drop=True)
RESOURCE_NODES = sorted(int(node) for node in node_table.node_id.unique())

events = read_all_splits("events").select(
    F.col("src_node_id").cast("int"),
    F.col("dst_node_id").cast("int"),
    F.col("t").cast("long"),
    F.col("t_idx").cast("int"),
    F.col("rpctype_idx").cast("int"),
    F.col("rt_rs").cast("float"),
    F.col("rt_abs_log_rs").cast("float")
)
full_events = events
if EVENT_FILTER == "touches_resource_node":
    events = events.filter(F.col("src_node_id").isin(RESOURCE_NODES) | F.col("dst_node_id").isin(RESOURCE_NODES))
filtered_events = events
if MAX_EVENTS_PER_MINUTE > 0:
    per_minute = events.groupBy("t_idx").count()
    # Keep each call with probability cap / count, so about `cap` calls remain in each minute.
    events = events.join(per_minute, "t_idx").filter(F.rand(42) * F.col("count") < MAX_EVENTS_PER_MINUTE).drop("count")

event_table = events.toPandas()
event_table["rpctype_idx"] = event_table.rpctype_idx.fillna(0)
event_table = event_table.sort_values(["t_idx", "t"], kind="stable").reset_index(drop=True)
assert np.all(np.diff(event_table.t.to_numpy()) >= 0), "calls are not in time order"
largest_node_id = int(max(node_table.node_id.max(), event_table.src_node_id.max(), event_table.dst_node_id.max()))
assert node_table.node_id.min() >= 0 and largest_node_id < NUM_NODES, \
    f"node id {largest_node_id} does not fit num_nodes {NUM_NODES}. Rerun the FE notebook so node ids run from 0 to N - 1."

# Same boundaries as the snapshot pipeline. The test range ends at the last minute with node data,
# because the call table can contain one more partial minute.
MIN_T = int(min(node_table.t_idx.min(), event_table.t_idx.min()))
MAX_T = int(max(node_table.t_idx.max(), event_table.t_idx.max()))
TRAIN_RANGE = [MIN_T, int(BOUNDS["train_end"]) - 1]
VAL_RANGE = [int(BOUNDS["train_end"]), int(BOUNDS["val_end"]) - 1]
TEST_RANGE = [int(BOUNDS["val_end"]), min(int(BOUNDS["test_end"]), int(node_table.t_idx.max()))]
CYCLE_PERIOD = CYCLE_PERIOD_SETTING if CYCLE_PERIOD_SETTING > 0 else int(BOUNDS["test_end"]) - MIN_T

print(f"nodes in graph: {NUM_NODES:,}, services with resource data: {len(RESOURCE_NODES):,}")
print(f"splits by minute: train {TRAIN_RANGE}, val {VAL_RANGE}, test {TEST_RANGE}, cycle period {CYCLE_PERIOD} min")
for split, (low, high) in (("train", TRAIN_RANGE), ("val", VAL_RANGE), ("test", TEST_RANGE)):
    kept = int(((event_table.t_idx >= low) & (event_table.t_idx <= high)).sum())
    raw = RAW_EVENT_COUNTS.get(split)
    print(f"  {split}: kept {kept:,} calls" + (f" of {raw:,} ({kept / raw:.1%})" if raw else ""))
calls_per_minute = event_table.groupby("t_idx").size()
print(f"calls per minute: median {int(calls_per_minute.median()):,}, max {int(calls_per_minute.max()):,}")
print(f"collected call table: {event_table.memory_usage(deep=True).sum() / 1e9:.2f} GB on the driver")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does the sample keep the shape of the traffic?
# MAGIC
# MAGIC The filter and the cap remove calls before the models see them. This check compares the kept calls with all calls of the
# MAGIC same minute: how many of the busiest edges survive, whether edges keep their rank, whether the busiest callers keep their out degree rank,
# MAGIC and whether busy minutes still look busy.

# COMMAND ----------

# Sampling check

from pyspark.sql import Window
from scipy import stats as sampling_stats
import warnings
warnings.filterwarnings("ignore", message="An input array is constant")

TOP_EDGES = 50
edge_keys = ["t_idx", "src_node_id", "dst_node_id"]
full_edge_counts = full_events.groupBy(*edge_keys).agg(F.count("*").alias("all_calls"))
busiest_edges = (
    full_edge_counts.withColumn("rank", F.row_number().over(Window.partitionBy("t_idx").orderBy(F.desc("all_calls"))))
    .filter(F.col("rank") <= 10 * TOP_EDGES).toPandas()
)
full_out_degree = (
    full_edge_counts.groupBy("t_idx", "src_node_id").agg(F.count("*").alias("all_degree"))
    .withColumn("rank", F.row_number().over(Window.partitionBy("t_idx").orderBy(F.desc("all_degree"))))
    .filter(F.col("rank") <= 10 * TOP_EDGES).drop("rank").toPandas()
)
full_minute_calls = full_events.groupBy("t_idx").count().toPandas().set_index("t_idx")["count"].sort_index()

kept_edge_counts = event_table.groupby(edge_keys).size().rename("kept_calls").reset_index()
kept_out_degree = kept_edge_counts.groupby(["t_idx", "src_node_id"]).size().rename("kept_degree").reset_index()
edge_view = busiest_edges.merge(kept_edge_counts, on=edge_keys, how="left").fillna({"kept_calls": 0})
degree_view = full_out_degree.merge(kept_out_degree, on=["t_idx", "src_node_id"], how="left").fillna({"kept_degree": 0})

check_rows = []
for minute, edges_in_minute in edge_view.groupby("t_idx"):
    top_all = set(edges_in_minute.nsmallest(TOP_EDGES, "rank")[["src_node_id", "dst_node_id"]].itertuples(index=False))
    top_kept = set(edges_in_minute.nlargest(TOP_EDGES, "kept_calls")[["src_node_id", "dst_node_id"]].itertuples(index=False))
    degrees = degree_view[degree_view.t_idx == minute]
    check_rows.append({
        "t_idx": minute,
        "top_edges_kept": len(top_all & top_kept) / max(len(top_all), 1),
        "edge_rank_spearman": sampling_stats.spearmanr(edges_in_minute.all_calls, edges_in_minute.kept_calls).correlation,
        "degree_rank_spearman": sampling_stats.spearmanr(degrees.all_degree, degrees.kept_degree).correlation if len(degrees) > 2 else np.nan
    })
SAMPLING_CHECK = pd.DataFrame(check_rows)
kept_minute_calls = event_table.groupby("t_idx").size().reindex(full_minute_calls.index, fill_value=0)
volume_spearman = sampling_stats.spearmanr(full_minute_calls, kept_minute_calls).correlation
print(SAMPLING_CHECK[["top_edges_kept", "edge_rank_spearman", "degree_rank_spearman"]].describe(percentiles=[0.1, 0.5]).round(3).to_string())
print(f"\nShare of all calls kept: {kept_minute_calls.sum() / full_minute_calls.sum():.1%}")
print(f"In a typical minute {100 * SAMPLING_CHECK.top_edges_kept.median():.0f}% of the {TOP_EDGES} busiest edges are still among the {TOP_EDGES} busiest after sampling.")
print(f"Edge rank correlation, median {SAMPLING_CHECK.edge_rank_spearman.median():.2f}. Caller out degree rank correlation, median {SAMPLING_CHECK.degree_rank_spearman.median():.2f}.")
print(f"Rank correlation of calls per minute, all calls against kept calls: {volume_spearman:.2f}.")
if MAX_EVENTS_PER_MINUTE > 0 and volume_spearman < 0.5:
    print("The cap flattens the traffic volume: busy minutes no longer have more calls than quiet ones. "
          "Volume information still reaches the models through the MCR node features.")
EXTRA_SETTINGS["sampling_check"] = {
    "kept_share": float(kept_minute_calls.sum() / full_minute_calls.sum()),
    "top_edges_kept_median": float(SAMPLING_CHECK.top_edges_kept.median()),
    "edge_rank_spearman_median": float(SAMPLING_CHECK.edge_rank_spearman.median()),
    "degree_rank_spearman_median": float(SAMPLING_CHECK.degree_rank_spearman.median()),
    "minute_volume_spearman": float(volume_spearman)
}
SAMPLING_CHECK.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_sampling_check.csv"), index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the per minute time steps
# MAGIC
# MAGIC Step `p` holds the calls of minute `p` in time order, which update the model after it has predicted.
# MAGIC It also holds the targets for minute `tau = p + horizon - 1`, which are predicted before minute `p` is ingested.
# MAGIC Model inputs replace NaN by 0 and are clipped to plus or minus 10 IQR. Targets and baselines are never clipped,
# MAGIC and rows with a NaN target are dropped.

# COMMAND ----------

# Arrays

assert "event_table" in globals(), "this cell frees the call table after converting it, so run section 2 again first"

CLIP = 10.0

# Calls: endpoints, exact time in seconds, two scaled response times and the rpctype.
N_RPC = int(min(16, event_table.rpctype_idx.max() + 1)) if len(event_table) else 1
EVENT_SRC = event_table.src_node_id.to_numpy(np.int32)
EVENT_DST = event_table.dst_node_id.to_numpy(np.int32)
EVENT_SECONDS = (event_table.t.to_numpy(np.int64) // 1000).astype(np.int32)
EVENT_RT = np.clip(np.nan_to_num(event_table[["rt_rs", "rt_abs_log_rs"]].to_numpy(np.float32)), -CLIP, CLIP)
EVENT_RPC = np.clip(event_table.rpctype_idx.to_numpy(), 0, N_RPC - 1).astype(np.uint8)
MSG_DIM = 2 + N_RPC
TOTAL_EVENTS = len(event_table)
EXTRA_SETTINGS["total_events"] = TOTAL_EVENTS


def row_ranges(minutes):
    """Map each minute to the (start, end) rows it occupies in an array sorted by minute."""
    unique, starts = np.unique(minutes, return_index=True)
    ends = np.append(starts[1:], len(minutes))
    return {int(m): (int(s), int(e)) for m, s, e in zip(unique, starts, ends)}


event_rows = row_ranges(event_table.t_idx.to_numpy())
del event_table                                   # the compact arrays above replace the pandas table
event_bytes = sum(array.nbytes for array in (EVENT_SRC, EVENT_DST, EVENT_SECONDS, EVENT_RT, EVENT_RPC))
print(f"calls: {TOTAL_EVENTS:,} in {event_bytes / 1e9:.2f} GB of arrays")
print(f"each TGN, TGAT or GraphMixer model keeps a {TOTAL_EVENTS * 13 / 1e9:.2f} GB call log on {DEVICE}")

# Node inputs, target and rolling mean baseline.
X_RAW = node_table[NODE_FEATS].to_numpy(np.float32)
NODE_INPUTS = np.clip(np.nan_to_num(X_RAW), -CLIP, CLIP)
NODE_DIM = NODE_INPUTS.shape[1]
NODE_IDS = node_table.node_id.to_numpy(np.int64)
Y_RAW = np.ascontiguousarray(X_RAW[:, TGT_IDX])
ROLL_RAW = np.ascontiguousarray(X_RAW[:, ROLL_IDX])
Y_OK = np.isfinite(Y_RAW)

finite_targets = Y_RAW[Y_OK]
print(f"NaN targets dropped: {int((~Y_OK).sum()):,}")
print(f"targets beyond {CLIP} IQR, kept unclipped: {int((np.abs(finite_targets) > CLIP).sum()):,}")

# A large share of one identical value usually means imputed rows. The FE imputes per instance before
# averaging to the service, so only minutes where every instance was missing show up here. It is a lower bound.
values, counts = np.unique(np.round(finite_targets, 6), return_counts=True)
MODE_SHARE = float(counts.max() / max(len(finite_targets), 1))
print(f"most frequent target value covers {MODE_SHARE:.2%} of rows (a lower bound on fully imputed service minutes)")

# COMMAND ----------

# Time steps

node_rows = row_ranges(node_table.t_idx.to_numpy())

STEPS = {}
for p in range(MIN_T, MAX_T + 1):
    tau = p + HORIZON - 1
    node_start, node_end = node_rows.get(p, (0, 0))
    event_start, event_end = event_rows.get(p, (0, 0))
    target_start, target_end = node_rows.get(tau, (0, 0))
    keep = Y_OK[target_start:target_end]
    angle = 2 * np.pi * tau / CYCLE_PERIOD
    time_of_cycle = np.tile(np.array([[np.sin(angle), np.cos(angle)]], dtype=np.float32), (int(keep.sum()), 1))
    STEPS[p] = {
        "t": p,
        "nid": torch.from_numpy(NODE_IDS[node_start:node_end]),
        "x": torch.from_numpy(NODE_INPUTS[node_start:node_end]),
        "y_obs": torch.from_numpy(Y_RAW[node_start:node_end]),
        "roll_obs": torch.from_numpy(ROLL_RAW[node_start:node_end]),
        "src": torch.from_numpy(EVENT_SRC[event_start:event_end]),
        "dst": torch.from_numpy(EVENT_DST[event_start:event_end]),
        "seconds": torch.from_numpy(EVENT_SECONDS[event_start:event_end]),
        "rt": torch.from_numpy(EVENT_RT[event_start:event_end]),
        "rpc": torch.from_numpy(EVENT_RPC[event_start:event_end]),
        "target": {
            "t": tau,
            "nid": torch.from_numpy(NODE_IDS[target_start:target_end][keep]),
            "y": torch.from_numpy(Y_RAW[target_start:target_end][keep]),
            "time_of_cycle": torch.from_numpy(time_of_cycle)
        }
    }


def steps_for(target_range):
    """Steps whose target minute falls inside target_range (inclusive)."""
    low, high = target_range
    return [p for p in range(MIN_T, MAX_T + 1) if low <= p + HORIZON - 1 <= high]


TRAIN_STEPS = steps_for(TRAIN_RANGE)
VAL_STEPS = steps_for(VAL_RANGE)
TEST_STEPS = steps_for(TEST_RANGE)

# The end of the validation window is held out as a calibration window. Tuning and early stopping only see the first part,
# so forecast errors on the calibration minutes are honest and can set safety margins downstream.
assert 0 < CALIBRATION_SHARE < 1, "calibration_share must be between 0 and 1"
calibration_count = max(1, int(round(len(VAL_STEPS) * CALIBRATION_SHARE)))
CAL_STEPS = VAL_STEPS[len(VAL_STEPS) - calibration_count:]
VAL_STEPS = VAL_STEPS[:len(VAL_STEPS) - calibration_count]
print(f"validation minutes for tuning and early stopping: {len(VAL_STEPS)}, held out calibration minutes: {len(CAL_STEPS)}")


def to_device(obj):
    if isinstance(obj, dict):
        return {key: to_device(value) for key, value in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(DEVICE, non_blocking=True)
    return obj


def count_targets(steps):
    return sum(len(STEPS[p]["target"]["y"]) for p in steps)


print(f"steps: train {len(TRAIN_STEPS)}, val {len(VAL_STEPS)}, test {len(TEST_STEPS)}")
print(f"target rows: train {count_targets(TRAIN_STEPS):,}, "
      f"val {count_targets(VAL_STEPS):,}, test {count_targets(TEST_STEPS):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Models
# MAGIC
# MAGIC All five models share the regression head and the node state. Only the encoder differs.
# MAGIC
# MAGIC | Model | Over time | Over the graph |
# MAGIC |---|---|---|
# MAGIC | TGCN | GRU hidden state per node, carried from minute to minute | Two GCN layers on the unique call pairs of each minute |
# MAGIC | TGN | Memory per node, updated at every call in time order | One attention layer over the latest calls of each node |
# MAGIC | TGAT | No memory, cosine encoding of how long ago a call happened | Two attention layers over a two hop neighbourhood |
# MAGIC | DCRNN | GRU hidden state per node with diffusion convolution gates | Random walks along and against the call direction, weighted by call counts |
# MAGIC | GraphMixer | No memory, fixed cosine encoding of how long ago a call happened | MLP-Mixer over the latest calls plus the mean of recent neighbours |

# COMMAND ----------

# Shared model parts


class RegressionHead(nn.Module):
    """Maps a node embedding and the node's last observed features to a correction of the last observed value.
    The last layer starts at zero, so an untrained model predicts exactly the persistence baseline."""

    def __init__(self, embedding_dim, dropout):
        super().__init__()
        input_dim = embedding_dim + NODE_DIM + 1 + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding, last_x, seen, time_of_cycle):
        features = torch.cat([embedding, last_x, seen.unsqueeze(-1), time_of_cycle], dim=-1)
        return self.net(features).squeeze(-1)


class NodeStateModel(nn.Module):
    """Keeps what each node looked like the last time it was observed. Subclasses add the encoder.
    Recurrent models set truncate_before_update so the last state update of a window is trained in that window."""

    tbptt = 1
    truncate_before_update = False

    def __init__(self):
        super().__init__()
        self.register_buffer("last_x", torch.zeros(NUM_NODES, NODE_DIM), persistent=False)
        self.register_buffer("last_y", torch.zeros(NUM_NODES), persistent=False)
        self.register_buffer("last_roll", torch.zeros(NUM_NODES), persistent=False)
        self.register_buffer("seen", torch.zeros(NUM_NODES), persistent=False)

    def reset_state(self):
        for buffer in (self.last_x, self.last_y, self.last_roll, self.seen):
            buffer.zero_()

    def detach_state(self):
        pass

    def remember_nodes(self, step):
        """Store the latest inputs, target and rolling mean of the nodes observed in this step."""
        if step["nid"].numel() == 0:
            return
        self.last_x[step["nid"]] = step["x"]
        finite = torch.isfinite(step["y_obs"])
        nodes = step["nid"][finite]
        y = step["y_obs"][finite]
        roll = step["roll_obs"][finite]
        self.last_y[nodes] = y
        self.last_roll[nodes] = torch.where(torch.isfinite(roll), roll, y)
        self.seen[nodes] = 1.0

    def has_history(self, target):
        """True for target rows whose node has an earlier observed value. Persistence is undefined otherwise."""
        return self.seen[target["nid"]] > 0

    def baselines(self, target):
        """Persistence and rolling mean predictions for the target rows."""
        return self.last_y[target["nid"]], self.last_roll[target["nid"]]

    def predict_from_embedding(self, embedding, target):
        nodes = target["nid"]
        correction = self.head(embedding, self.last_x[nodes], self.seen[nodes], target["time_of_cycle"])
        return self.last_y[nodes] + correction


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


def event_message(rt, rpc):
    """Message of a call: two scaled response times and a one hot rpctype."""
    return torch.cat([rt.float(), TF.one_hot(rpc.long().clamp(max=N_RPC - 1), N_RPC).float()], dim=-1)


def event_batches(step):
    """The calls of one minute in time order, in batches of EVENT_BATCH."""
    n = step["src"].numel()
    for start in range(0, n, EVENT_BATCH):
        end = min(start + EVENT_BATCH, n)
        yield (step["src"][start:end].long(), step["dst"][start:end].long(), step["seconds"][start:end],
               step["rt"][start:end], step["rpc"][start:end])


class EventLog:
    """Time and features of every call ingested so far, 13 bytes per call.
    The position of a call in the log is the id used by the neighbour loader."""

    def __init__(self):
        size = max(TOTAL_EVENTS, 1)
        self.seconds = torch.zeros(size, dtype=torch.int32, device=DEVICE)
        self.rt = torch.zeros(size, 2, dtype=torch.float32, device=DEVICE)
        self.rpc = torch.zeros(size, dtype=torch.uint8, device=DEVICE)
        self.size = 0

    def reset(self):
        self.size = 0

    def add(self, seconds, rt, rpc):
        end = self.size + seconds.numel()
        self.seconds[self.size:end] = seconds.to(torch.int32)
        self.rt[self.size:end] = rt.float()
        self.rpc[self.size:end] = rpc.to(torch.uint8)
        self.size = end

    def times(self, ids):
        return self.seconds[ids].long()

    def messages(self, ids):
        return event_message(self.rt[ids], self.rpc[ids])


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
    """Two GCN layers on the unique call pairs of each minute feed a GRU cell per node."""

    truncate_before_update = True

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        self.tbptt = params["tbptt"]
        self.hidden_dim = hidden
        self.gcn1 = GCNConv(NODE_DIM, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = RegressionHead(hidden, params["dropout"])
        self.register_buffer("h", torch.zeros(NUM_NODES, hidden), persistent=False)

    def reset_state(self):
        super().reset_state()
        self.h = torch.zeros(NUM_NODES, self.hidden_dim, device=self.last_x.device)

    def detach_state(self):
        self.h = self.h.detach()

    def predict(self, target):
        return self.predict_from_embedding(self.h[target["nid"]], target)

    def update(self, step):
        if step["nid"].numel() > 0:
            x = torch.zeros(NUM_NODES, NODE_DIM, device=step["x"].device)
            x[step["nid"]] = step["x"]
            if step["src"].numel() > 0:
                pairs = torch.unique(torch.stack([step["src"].long(), step["dst"].long()]), dim=1)
                edges = torch.cat([pairs, pairs.flip(0)], dim=1)
            else:
                edges = torch.zeros(2, 0, dtype=torch.long, device=step["x"].device)
            z = TF.relu(self.gcn1(x, edges))
            z = self.gcn2(z, edges)
            new_h = self.gru(z[step["nid"]], self.h[step["nid"]])
            self.h = self.h.index_copy(0, step["nid"], new_h)
        self.remember_nodes(step)

# COMMAND ----------

# TGN


class TemporalAttention(nn.Module):
    """One graph attention layer whose edge features are a time encoding and the call message."""

    def __init__(self, dim, time_encoder, dropout):
        super().__init__()
        self.time_encoder = time_encoder
        self.conv = TransformerConv(dim, dim // 2, heads=2, dropout=dropout, edge_dim=MSG_DIM + time_encoder.out_channels)

    def forward(self, x, last_update, edge_index, edge_time, edge_msg):
        relative_time = (last_update[edge_index[0]] - edge_time).to(x.dtype)
        edge_attr = torch.cat([self.time_encoder(relative_time), edge_msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGNModel(NodeStateModel):
    """Per node memory updated at every call in time order, followed by one attention layer over recent neighbours."""

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
        self.head = RegressionHead(hidden, params["dropout"])
        self.loader = RecentNeighborLoader(NUM_NODES, size=params["neighbor_size"], device=DEVICE)
        self.log = EventLog()
        self.local_index = torch.empty(NUM_NODES, dtype=torch.long, device=DEVICE)

    def reset_state(self):
        super().reset_state()
        self.memory.reset_state()
        self.loader.reset_state()
        self.log.reset()

    def detach_state(self):
        self.memory.detach()

    def predict(self, target):
        node_ids, edge_index, edge_ids = self.loader(target["nid"].unique())
        self.local_index[node_ids] = torch.arange(node_ids.size(0), device=node_ids.device)
        memory, last_update = self.memory(node_ids)
        embedding = self.attention(memory, last_update, edge_index, self.log.times(edge_ids), self.log.messages(edge_ids))
        return self.predict_from_embedding(embedding[self.local_index[target["nid"]]], target)

    def update(self, step):
        for src, dst, seconds, rt, rpc in event_batches(step):
            self.memory.update_state(src, dst, seconds.long(), event_message(rt, rpc))
            self.loader.insert(src, dst)
            self.log.add(seconds, rt, rpc)
        self.remember_nodes(step)

# COMMAND ----------

# TGAT


class TGATModel(NodeStateModel):
    """Two attention layers over a two hop neighbourhood, with a cosine encoding of how long ago each call happened.
    Layer 1 updates every neighbour from its own neighbours. Layer 2 updates the target nodes from their neighbours."""

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        self.time_encoder = CosineTimeEncoder(params["time_dim"])
        self.input = nn.Linear(NODE_DIM, hidden)
        edge_dim = MSG_DIM + params["time_dim"]
        self.layer1 = TransformerConv(hidden, hidden // 2, heads=2, dropout=params["dropout"], edge_dim=edge_dim)
        self.layer2 = TransformerConv(hidden, hidden // 2, heads=2, dropout=params["dropout"], edge_dim=edge_dim)
        self.head = RegressionHead(hidden, params["dropout"])
        self.loader = RecentNeighborLoader(NUM_NODES, size=params["neighbor_size"], device=DEVICE)
        self.log = EventLog()
        self.local_index = torch.empty(NUM_NODES, dtype=torch.long, device=DEVICE)

    def reset_state(self):
        super().reset_state()
        self.loader.reset_state()
        self.log.reset()

    def edge_features(self, edge_ids, target_minute):
        minutes_ago = (target_minute * 60 - self.log.times(edge_ids)).float() / 60.0
        return torch.cat([self.time_encoder(minutes_ago), self.log.messages(edge_ids)], dim=-1)

    def embed(self, query_nodes, target_minute):
        hop1_nodes, hop1_edges, hop1_ids = self.loader(query_nodes)
        hop1_edges_global = hop1_nodes[hop1_edges]          # the loader relabels on every call, so keep global ids
        hop2_nodes, hop2_edges, hop2_ids = self.loader(hop1_nodes)
        self.local_index[hop2_nodes] = torch.arange(hop2_nodes.size(0), device=hop2_nodes.device)
        h = self.input(self.last_x[hop2_nodes])
        h = TF.relu(self.layer1(h, hop2_edges, self.edge_features(hop2_ids, target_minute)))
        return self.layer2(h, self.local_index[hop1_edges_global], self.edge_features(hop1_ids, target_minute))

    def predict(self, target):
        embedding = self.embed(target["nid"].unique(), target["t"])
        return self.predict_from_embedding(embedding[self.local_index[target["nid"]]], target)

    def update(self, step):
        for src, dst, seconds, rt, rpc in event_batches(step):
            self.loader.insert(src, dst)
            self.log.add(seconds, rt, rpc)
        self.remember_nodes(step)

# COMMAND ----------

# DCRNN


def random_walk_step(values, source, target, weight, num_nodes):
    """One random walk step: each source node takes the weighted mean of `values` at the targets of its edges."""
    degree = torch.zeros(num_nodes, device=values.device).index_add_(0, source, weight)
    total = torch.zeros_like(values).index_add_(0, source, weight.unsqueeze(-1) * values[target])
    return total / degree.clamp(min=1e-12).unsqueeze(-1)


class DiffusionConv(nn.Module):
    """Diffusion convolution of Li et al. (2018): the sum over k = 0 to K - 1 of
    theta_k1 (D_O^-1 W)^k X + theta_k2 (D_I^-1 W^T)^k X. The two k = 0 terms are both X, so they share one weight."""

    def __init__(self, in_dim, out_dim, diffusion_steps):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.linear = nn.Linear((2 * diffusion_steps - 1) * in_dim, out_dim)

    def forward(self, x, graph):
        caller, callee, weight = graph
        terms = [x]
        for source, target in ((caller, callee), (callee, caller)):
            walked = x
            for _ in range(self.diffusion_steps - 1):
                walked = random_walk_step(walked, source, target, weight, x.size(0))
                terms.append(walked)
        return self.linear(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    """DCGRU cell of Li et al. (2018): a GRU whose gate and candidate transforms are diffusion convolutions."""

    def __init__(self, in_dim, hidden, diffusion_steps):
        super().__init__()
        self.gates = DiffusionConv(in_dim + hidden, 2 * hidden, diffusion_steps)
        self.candidate = DiffusionConv(in_dim + hidden, hidden, diffusion_steps)

    def forward(self, x, h, graph):
        reset, update = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1), graph)).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=-1), graph))
        return update * h + (1 - update) * candidate


class DCRNNModel(NodeStateModel):
    """Stacked DCGRU cells on the directed call graph of each minute.
    The edge weight W_ij is the number of calls from service i to service j in that minute."""

    truncate_before_update = True

    def __init__(self, params):
        super().__init__()
        hidden = params["hidden_dim"]
        self.tbptt = params["tbptt"]
        self.hidden_dim = hidden
        self.num_layers = params["num_layers"]
        input_dims = [NODE_DIM] + [hidden] * (self.num_layers - 1)
        self.cells = nn.ModuleList([DCGRUCell(dim, hidden, params["diffusion_steps"]) for dim in input_dims])
        self.head = RegressionHead(hidden, params["dropout"])
        self.register_buffer("h", torch.zeros(self.num_layers, NUM_NODES, hidden), persistent=False)

    def reset_state(self):
        super().reset_state()
        self.h = torch.zeros(self.num_layers, NUM_NODES, self.hidden_dim, device=self.last_x.device)

    def detach_state(self):
        self.h = self.h.detach()

    def predict(self, target):
        return self.predict_from_embedding(self.h[-1][target["nid"]], target)

    def minute_graph(self, step):
        device = self.h.device
        if step["src"].numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty, torch.zeros(0, device=device)
        pairs, counts = torch.unique(torch.stack([step["src"].long(), step["dst"].long()]), dim=1, return_counts=True)
        return pairs[0], pairs[1], counts.float()

    def update(self, step):
        x = torch.zeros(NUM_NODES, NODE_DIM, device=self.h.device)
        if step["nid"].numel() > 0:
            x[step["nid"]] = step["x"]
        graph = self.minute_graph(step)
        layer_input = x
        new_states = []
        for cell, h in zip(self.cells, self.h):
            layer_input = cell(layer_input, h, graph)
            new_states.append(layer_input)
        self.h = torch.stack(new_states)
        self.remember_nodes(step)

# COMMAND ----------

# GraphMixer


class FixedTimeEncoder(nn.Module):
    """GraphMixer time encoding cos(t w) with w_i = alpha^(-(i - 1) / beta) and alpha = beta = sqrt(d). It is not trained."""

    def __init__(self, dim):
        super().__init__()
        scale = math.sqrt(dim)
        frequencies = scale ** (-np.arange(dim) / scale)
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))

    def forward(self, t):
        return torch.cos(t.unsqueeze(-1) * self.frequencies)


class MixerBlock(nn.Module):
    """One MLP-Mixer layer as written in the GraphMixer paper.
    Token mixing works across the K calls and channel mixing works across the features. Both have a residual connection."""

    def __init__(self, n_tokens, dim, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(n_tokens, token_hidden), nn.GELU(), nn.Linear(token_hidden, n_tokens))
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, channel_hidden), nn.GELU(), nn.Linear(channel_hidden, dim))

    def forward(self, h):
        h = h + self.token_mlp(self.token_norm(h).transpose(1, 2)).transpose(1, 2)
        return h + self.channel_mlp(self.channel_norm(h))


class GraphMixerModel(NodeStateModel):
    """GraphMixer of Cong et al. (2023) with the link classifier replaced by the regression head.
    The link encoder mixes the K most recent calls of a service. The node encoder adds the mean features of the services
    met in the most recent `node_window_calls` calls."""

    def __init__(self, params):
        super().__init__()
        self.n_links = params["recent_links"]
        self.window_calls = params["node_window_calls"]
        link_dim = params["time_dim"] + MSG_DIM
        self.time_encoder = FixedTimeEncoder(params["time_dim"])
        self.mixer = MixerBlock(self.n_links, link_dim, max(1, self.n_links // 2), params["hidden_dim"])
        self.head = RegressionHead(NODE_DIM + link_dim, params["dropout"])
        self.loader = RecentNeighborLoader(NUM_NODES, size=self.n_links, device=DEVICE)
        self.log = EventLog()
        self.clear_window()

    def clear_window(self):
        self.window_src = torch.zeros(0, dtype=torch.long, device=DEVICE)
        self.window_dst = torch.zeros(0, dtype=torch.long, device=DEVICE)

    def reset_state(self):
        super().reset_state()
        self.loader.reset_state()
        self.log.reset()
        self.clear_window()

    def link_encoding(self, nodes, target_minute):
        call_ids = self.loader.e_id[nodes]
        present = (call_ids >= 0).unsqueeze(-1).float()
        call_ids = call_ids.clamp(min=0)
        seconds_ago = (target_minute * 60 - self.log.times(call_ids)).float()
        tokens = torch.cat([self.time_encoder(seconds_ago), self.log.messages(call_ids)], dim=-1) * present
        return self.mixer(tokens).mean(dim=1)

    def node_encoding(self, nodes):
        both_ways = torch.stack([torch.cat([self.window_src, self.window_dst]), torch.cat([self.window_dst, self.window_src])])
        node, neighbour = torch.unique(both_ways, dim=1)
        observed = self.seen[neighbour] > 0
        node, neighbour = node[observed], neighbour[observed]
        total = torch.zeros(NUM_NODES, NODE_DIM, device=DEVICE).index_add_(0, node, self.last_x[neighbour])
        count = torch.zeros(NUM_NODES, device=DEVICE).index_add_(0, node, torch.ones(node.numel(), device=DEVICE))
        neighbour_mean = total / count.clamp(min=1).unsqueeze(-1)
        return self.last_x[nodes] + neighbour_mean[nodes]

    def predict(self, target):
        nodes = target["nid"]
        embedding = torch.cat([self.node_encoding(nodes), self.link_encoding(nodes, target["t"])], dim=-1)
        return self.predict_from_embedding(embedding, target)

    def update(self, step):
        for src, dst, seconds, rt, rpc in event_batches(step):
            self.loader.insert(src, dst)
            self.log.add(seconds, rt, rpc)
            self.window_src = torch.cat([self.window_src, src])[-self.window_calls:]
            self.window_dst = torch.cat([self.window_dst, dst])[-self.window_calls:]
        self.remember_nodes(step)

# COMMAND ----------

# Model registry

MODEL_CLASSES = {
    "TGCN": TGCNModel,
    "TGN": TGNModel,
    "TGAT": TGATModel,
    "DCRNN": DCRNNModel,
    "GraphMixer": GraphMixerModel
}


def build_model(name, params):
    return MODEL_CLASSES[name](params).to(DEVICE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training with early stopping
# MAGIC
# MAGIC Every epoch walks through the training minutes in order and then continues into the validation minutes.
# MAGIC Rows of a service with no earlier observation are left out of the loss and the metrics.
# MAGIC The weights of the epoch with the lowest validation MAE are kept.
# MAGIC TGCN and DCRNN carry a hidden state and are trained with truncated backpropagation over `tbptt` minutes.
# MAGIC Each window ends before the next minute is ingested, so every state update inside the window receives gradient.
# MAGIC TGN keeps the order of the PyG TGN example, where the memory module is trained inside its own forward pass.

# COMMAND ----------

# Loss and one pass over the time steps


def make_loss(kind, beta):
    if kind == "huber":
        return lambda pred, y: TF.smooth_l1_loss(pred, y, beta=beta)
    if kind == "mse":
        return lambda pred, y: TF.mse_loss(pred, y)
    if kind == "mae":
        return lambda pred, y: TF.l1_loss(pred, y)
    raise ValueError(f"unknown loss {kind}")


LOSS_FN = make_loss(LOSS_TYPE, HUBER_BETA)


def optimizer_step(model, optimizer, loss):
    if loss is None:
        return
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()


def end_window(model, optimizer, loss):
    """Apply the summed loss of one truncated window, then cut the gradient path through the recurrent state."""
    optimizer_step(model, optimizer, loss)
    model.detach_state()


def select_rows(target, keep):
    """Keep only the target rows where `keep` is True."""
    return {key: value[keep] if torch.is_tensor(value) else value for key, value in target.items()}


def run_sequence(model, steps, optimizer=None, collect=False, log_every=0, tag=""):
    """Predict and then ingest each step in order. Trains when an optimizer is given.
    Returns the mean loss, the row counts and, when collect is True, the predictions with both baselines."""
    training = optimizer is not None
    collected = {key: [] for key in ("y", "pred", "persistence", "rolling", "nid", "minute")}
    loss_sum = 0.0
    n_rows = 0
    n_cold = 0
    pending_loss = None

    with torch.set_grad_enabled(training):
        for k, p in enumerate(steps, start=1):
            step = to_device(STEPS[p])
            target = step["target"]
            if target["y"].numel() > 0:
                keep = model.has_history(target)
                n_cold += int((~keep).sum())
                if bool(keep.any()):
                    target = select_rows(target, keep)
                    pred = model.predict(target)
                    loss = LOSS_FN(pred, target["y"])
                    n = target["y"].numel()
                    loss_sum += loss.item() * n
                    n_rows += n
                    if training:
                        pending_loss = loss if pending_loss is None else pending_loss + loss
                    if collect:
                        persistence, rolling = model.baselines(target)
                        collected["y"].append(target["y"].cpu())
                        collected["pred"].append(pred.detach().cpu())
                        collected["persistence"].append(persistence.cpu())
                        collected["rolling"].append(rolling.cpu())
                        collected["nid"].append(target["nid"].cpu())
                        collected["minute"].append(torch.full((n,), int(target["t"]), dtype=torch.long))

            window_ends = training and k % model.tbptt == 0
            if window_ends and model.truncate_before_update:
                end_window(model, optimizer, pending_loss)
                pending_loss = None
            model.update(step)
            if window_ends and not model.truncate_before_update:
                end_window(model, optimizer, pending_loss)
                pending_loss = None
            if training and log_every and k % log_every == 0:
                print(f"    {tag} minute {k}/{len(steps)}, running train loss {loss_sum / max(n_rows, 1):.4f}")

        if training and pending_loss is not None:
            optimizer_step(model, optimizer, pending_loss)
            model.detach_state()

    result = {"loss": loss_sum / max(n_rows, 1), "rows": n_rows, "cold_rows": n_cold}
    if collect and collected["y"]:
        for key, parts in collected.items():
            result[key] = torch.cat(parts).numpy()
    return result

# COMMAND ----------

# Training one model with early stopping


def mae_pp(y, pred):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float)))) * PP_PER_UNIT


def epoch_summary(epoch, train, val):
    return {
        "epoch": epoch,
        "train_loss": train["loss"],
        "val_loss": val["loss"],
        "train_mae_pp": mae_pp(train["y"], train["pred"]),
        "val_mae_pp": mae_pp(val["y"], val["pred"]),
        "train_persistence_mae_pp": mae_pp(train["y"], train["persistence"]),
        "val_persistence_mae_pp": mae_pp(val["y"], val["persistence"])
    }


def copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_run(name, params, max_epochs, seed, verbose=True):
    """Train one model with one seed. Stops after PATIENCE epochs without a lower validation MAE.
    Returns the best weights, the best epoch and the history of every epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    history = []
    best_score = math.inf
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
                  f" | val MAE {summary['val_mae_pp']:.3f} {UNIT_LABEL}"
                  f" (persistence {summary['val_persistence_mae_pp']:.3f})"
                  f" | {time.time() - start:.0f}s")

        if summary["val_mae_pp"] < best_score:
            best_score = summary["val_mae_pp"]
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
        print("warning: validation MAE was undefined in every epoch, keeping the last epoch")
        best_score = float("nan")
        best_epoch = len(history)
        best_state = copy_state(model)

    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": name, "seed": seed, "params": params, "state": best_state, "best_epoch": best_epoch,
            "best_val_mae_pp": best_score, "history": history,
            "epochs_run": len(history), "train_seconds": float(sum(row["seconds"] for row in history))}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Hyperparameter search
# MAGIC
# MAGIC The search tries the default configuration first and then random draws from `SEARCH_SPACE`, without repeats.
# MAGIC Each trial trains with the first seed for up to the model's tuning epochs, with early stopping.
# MAGIC The configuration with the lowest validation MAE wins. The test window is not used.
# MAGIC Because the winner is picked on validation, its validation MAE is optimistic. Only test results are unbiased.

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
        rows.append({"trial": trial, **params, "best_epoch": run["best_epoch"], "val_mae_pp": run["best_val_mae_pp"]})
        print(f"trial {trial}: val MAE {run['best_val_mae_pp']:.4f} {UNIT_LABEL} at epoch {run['best_epoch']}"
              f" | {time.time() - start:.0f}s | {params}")
    table = pd.DataFrame(rows).sort_values("val_mae_pp", kind="stable", na_position="last").reset_index(drop=True)
    best_trial = int(table.loc[0, "trial"])
    TUNED_PARAMS[name] = configurations[best_trial - 1]
    TUNING_RESULTS[name] = table
    print(f"best configuration for {name}: {TUNED_PARAMS[name]}")
    return table


def params_for(name):
    return TUNED_PARAMS.get(name, DEFAULT_PARAMS[name])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluation and statistical tests
# MAGIC
# MAGIC The final model is trained once per seed with its tuned hyperparameters and its own maximum epochs.
# MAGIC The first seed is used for the charts and the tests. Further seeds show how much results move between runs.
# MAGIC
# MAGIC **How the intervals work.** All metrics are built from per minute sums. A metric can therefore be recomputed
# MAGIC with one test minute left out. These leave one out values (jackknife) show how much each minute moves the metric.
# MAGIC Their long run variance is estimated with the equal weighted cosine method of Lazarus, Lewis, Stock and Watson (2018).
# MAGIC This gives the standard error. Intervals and p values use a t distribution. In simulations with 108 minutes the 5% test
# MAGIC rejected about 5% of the time for moderately autocorrelated errors and about 9% for strongly autocorrelated errors.

# COMMAND ----------

# Point metrics


def to_percent(scaled):
    """Scaled value back to utilisation in percent."""
    return (np.asarray(scaled, dtype=float) * TARGET_IQR + TARGET_MEDIAN) * UNIT_SCALE


def regression_metrics(y, pred):
    """MAE, RMSE, bias and R2. R2 is pooled over all services, so it mostly reflects differences between services."""
    y = np.asarray(y, dtype=float)
    error = np.asarray(pred, dtype=float) - y
    mse = float(np.mean(error ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae_pp": float(np.mean(np.abs(error))) * PP_PER_UNIT,
        "rmse_pp": math.sqrt(mse) * PP_PER_UNIT,
        "bias_pp": float(error.mean()) * PP_PER_UNIT,
        "r2": float(1 - np.sum(error ** 2) / total) if total > 0 else float("nan"),
        "mse": mse,
        "mse_pp2": mse * PP_PER_UNIT ** 2,
        "rows": int(len(y))
    }


def split_metrics(result):
    """Metrics of the model and both baselines on one split, with skill relative to persistence."""
    out = {
        "model": regression_metrics(result["y"], result["pred"]),
        "persistence": regression_metrics(result["y"], result["persistence"]),
        "rolling": regression_metrics(result["y"], result["rolling"])
    }
    reference = out["persistence"]["mse"]
    for metrics in out.values():
        metrics["skill"] = 1 - metrics["mse"] / reference if reference > 0 else float("nan")
    return out

# COMMAND ----------

# Statistical inference


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


def jackknife(statistic, n_minutes):
    """Full sample value and leave one minute out pseudo values of a statistic.
    `statistic(weights)` must accept one weight per minute, where 0 removes that minute."""
    weights = np.ones(n_minutes)
    full = float(statistic(weights))
    pseudo = np.empty(n_minutes)
    for t in range(n_minutes):
        weights[t] = 0.0
        pseudo[t] = n_minutes * full - (n_minutes - 1) * float(statistic(weights))
        weights[t] = 1.0
    return full, pseudo


def summarise_statistic(statistic, n_minutes, positive=False):
    """Estimate, standard error, 95% interval and two sided p value against zero.
    For a positive metric the interval is built on the log scale so it stays above zero."""
    estimate, pseudo = jackknife(statistic, n_minutes)
    dof = ewc_terms(n_minutes)
    std_error = math.sqrt(max(ewc_variance(pseudo, dof), 0.0) / n_minutes)
    if n_minutes < 8 or std_error == 0.0:
        return {"est": estimate, "se": std_error, "low": estimate, "high": estimate, "p": float("nan")}
    q = float(sstats.t.ppf(0.975, dof))
    if positive and estimate > 0:
        log_se = std_error / estimate
        low, high = estimate * math.exp(-q * log_se), estimate * math.exp(q * log_se)
    else:
        low, high = estimate - q * std_error, estimate + q * std_error
    p_value = float(2 * sstats.t.sf(abs(estimate / std_error), dof))
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


def minute_sums(result, series):
    """Per test minute: row count, sums of the target and its square, and sums of signed, absolute and squared errors."""
    minutes, index = np.unique(result["minute"], return_inverse=True)
    y = result["y"].astype(float)
    sums = {
        "rows": np.bincount(index).astype(float),
        "y": np.bincount(index, weights=y),
        "y_sq": np.bincount(index, weights=y ** 2)
    }
    for name, pred in series.items():
        error = np.asarray(pred, dtype=float) - y
        sums[f"{name}_err"] = np.bincount(index, weights=error)
        sums[f"{name}_abs"] = np.bincount(index, weights=np.abs(error))
        sums[f"{name}_sq"] = np.bincount(index, weights=error ** 2)
    return sums, len(minutes)


def mae_statistic(sums, name):
    return lambda w: (w @ sums[f"{name}_abs"]) / (w @ sums["rows"]) * PP_PER_UNIT


def rmse_statistic(sums, name):
    return lambda w: math.sqrt((w @ sums[f"{name}_sq"]) / (w @ sums["rows"])) * PP_PER_UNIT


def mse_statistic(sums, name):
    return lambda w: (w @ sums[f"{name}_sq"]) / (w @ sums["rows"]) * PP_PER_UNIT ** 2


def bias_statistic(sums, name):
    return lambda w: (w @ sums[f"{name}_err"]) / (w @ sums["rows"]) * PP_PER_UNIT


def r2_statistic(sums, name):
    """Pooled R2: one minus the squared error over the squared deviation from the mean of the kept minutes."""
    def r2(w):
        rows = w @ sums["rows"]
        total = (w @ sums["y_sq"]) - (w @ sums["y"]) ** 2 / rows
        return 1 - (w @ sums[f"{name}_sq"]) / total
    return r2


def skill_statistic(sums, name):
    return lambda w: 1 - (w @ sums[f"{name}_sq"]) / (w @ sums["persistence_sq"])


def mae_difference_statistic(sums, a, b):
    return lambda w: (w @ (sums[f"{a}_abs"] - sums[f"{b}_abs"])) / (w @ sums["rows"]) * PP_PER_UNIT


def mse_difference_statistic(sums, a, b):
    return lambda w: (w @ (sums[f"{a}_sq"] - sums[f"{b}_sq"])) / (w @ sums["rows"]) * PP_PER_UNIT ** 2


def test_window_inference(test):
    """Intervals for the model and both baselines, and tests of the model against each baseline."""
    series = {"model": test["pred"], "persistence": test["persistence"], "rolling": test["rolling"]}
    sums, n_minutes = minute_sums(test, series)
    out = {"minutes": n_minutes, "dof": ewc_terms(n_minutes)}
    for name in series:
        out[f"{name}_mae"] = summarise_statistic(mae_statistic(sums, name), n_minutes, positive=True)
        out[f"{name}_rmse"] = summarise_statistic(rmse_statistic(sums, name), n_minutes, positive=True)
        out[f"{name}_mse"] = summarise_statistic(mse_statistic(sums, name), n_minutes, positive=True)
        out[f"{name}_r2"] = summarise_statistic(r2_statistic(sums, name), n_minutes)
        out[f"{name}_bias"] = summarise_statistic(bias_statistic(sums, name), n_minutes)
    out["model_skill"] = summarise_statistic(skill_statistic(sums, "model"), n_minutes)
    out["rolling_skill"] = summarise_statistic(skill_statistic(sums, "rolling"), n_minutes)
    for baseline in ("persistence", "rolling"):
        out[f"mae_diff_{baseline}"] = summarise_statistic(mae_difference_statistic(sums, "model", baseline), n_minutes)
        out[f"mse_diff_{baseline}"] = summarise_statistic(mse_difference_statistic(sums, "model", baseline), n_minutes)
    return out


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
        "cal": run_sequence(model, CAL_STEPS, collect=True),
        "test": run_sequence(model, TEST_STEPS, collect=True)
    }
    del model
    return results


def evaluate_model(name):
    assert name in TRAINED, f"run train_model('{name}') first"
    print(f"\n==== evaluating {name} ====")
    seed_rows = []
    seed_results = []
    first = None
    for run in TRAINED[name]:
        results = predict_all_splits(run)
        seed_results.append({"seed": run["seed"], "val": results["val"], "cal": results["cal"], "test": results["test"]})
        metrics = {split: split_metrics(result) for split, result in results.items()}
        seed_rows.append({
            "seed": run["seed"],
            "best_epoch": run["best_epoch"],
            "test_mae_pp": metrics["test"]["model"]["mae_pp"],
            "test_skill": metrics["test"]["model"]["skill"]
        })
        if first is None:
            first = (run, results, metrics)

    run, results, metrics = first
    RESULTS[name] = {
        "run": run,
        "metrics": metrics,
        "val": results["val"],
        "cal": results["cal"],
        "test": results["test"],
        "seed_results": seed_results,
        "inference": test_window_inference(results["test"]),
        "cold_rows": {split: result["cold_rows"] for split, result in results.items()},
        "seeds": pd.DataFrame(seed_rows)
    }
    print_evaluation(name)
    return RESULTS[name]


def print_evaluation(name):
    result = RESULTS[name]
    run, metrics, inference = result["run"], result["metrics"], result["inference"]
    print(f"seed {run['seed']}, best epoch {run['best_epoch']}, cold start rows left out: {result['cold_rows']}")

    rows = []
    for split in ("train", "val", "test"):
        for series, m in metrics[split].items():
            rows.append({"split": split, "series": series, "mse_pp2": m["mse_pp2"], "rmse_pp": m["rmse_pp"],
                         "mae_pp": m["mae_pp"], "r2": m["r2"], "bias_pp": m["bias_pp"], "skill": m["skill"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print(f"\ntest window: {inference['minutes']} minutes, t({inference['dof']}) intervals")
    for series in ("model", "persistence", "rolling"):
        print(f"  {series:11s} MAE {interval_text(inference[f'{series}_mae'])} {UNIT_LABEL} | "
              f"RMSE {interval_text(inference[f'{series}_rmse'])} {UNIT_LABEL} | "
              f"R2 {interval_text(inference[f'{series}_r2'])}")
    skill = inference["model_skill"]
    print(f"  skill against persistence {interval_text(skill)}, p = {skill['p']:.4g}")
    for baseline in ("persistence", "rolling"):
        mae, mse = inference[f"mae_diff_{baseline}"], inference[f"mse_diff_{baseline}"]
        print(f"  model minus {baseline}: MAE {interval_text(mae)} (p = {mae['p']:.4g}), "
              f"MSE {interval_text(mse)} (p = {mse['p']:.4g}). Negative means the model is better.")

    if len(result["seeds"]) > 1:
        seeds = result["seeds"]
        print(f"\nspread over {len(seeds)} seeds:")
        print(f"  test MAE {seeds.test_mae_pp.mean():.4f} +/- {seeds.test_mae_pp.std(ddof=1):.4f} {UNIT_LABEL}")
        print(f"  test skill {seeds.test_skill.mean():+.4f} +/- {seeds.test_skill.std(ddof=1):.4f}")
        print(seeds.round(4).to_string(index=False))


def print_diagnosis(name):
    """Conclusions from the tests, followed by rules of thumb that are not tests."""
    result = RESULTS[name]
    skill = result["inference"]["model_skill"]
    rolling = result["inference"]["mae_diff_rolling"]

    if not np.isfinite(skill["p"]):
        verdict = "identical to persistence on the test window"
    elif skill["p"] < 0.05 and skill["low"] > 0:
        verdict = "better than persistence on MSE"
    elif skill["p"] < 0.05 and skill["high"] < 0:
        verdict = "worse than persistence on MSE"
    else:
        verdict = "not distinguishable from persistence on MSE at the 5% level"
    print(f"{name} is {verdict}: skill {interval_text(skill)}, p = {skill['p']:.3g}.")
    significant = np.isfinite(rolling["p"]) and rolling["p"] < 0.05
    print(f"Against the rolling mean the MAE difference is {interval_text(rolling)}, "
          f"p = {rolling['p']:.3g} ({'significant' if significant else 'not significant'} at 5%).")

    notes = []
    history = pd.DataFrame(result["run"]["history"])
    if result["run"]["best_epoch"] == MAX_EPOCHS[name]:
        notes.append("The best epoch is the last allowed epoch, so a higher maximum may help.")
    if result["metrics"]["test"]["model"]["mae_pp"] > 1.5 * result["metrics"]["train"]["model"]["mae_pp"]:
        notes.append("Test MAE is more than 1.5 times train MAE. This can be overfitting or drift between periods.")
    if len(history) > 1 and history.val_loss.iloc[-1] > 1.05 * history.val_loss.min():
        notes.append("Validation loss rose after its minimum. Early stopping kept the best epoch.")
    if MODE_SHARE > 0.01:
        notes.append(f"{MODE_SHARE:.1%} of targets share one value, which suggests imputed rows.")
    print("Rules of thumb (not tests): " + (" ".join(notes) if notes else "none triggered."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Chart functions
# MAGIC Each function draws one chart. Each chart is called from its own cell in the model sections.

# COMMAND ----------

# Charts for one model

COLORS = {
    "model": "#1f77b4",
    "persistence": "#7f7f7f",
    "rolling": "#bcbd22",
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c"
}


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
    """Best validation MAE of every tuning trial."""
    if name not in TUNING_RESULTS:
        print(f"no tuning results for {name} (run_tuning is off or the tuning cell was skipped)")
        return
    table = TUNING_RESULTS[name].sort_values("trial")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.trial.astype(str), table.val_mae_pp, color=COLORS["model"])
    best = TUNING_RESULTS[name].loc[0, "trial"]
    ax.bar(str(best), TUNING_RESULTS[name].loc[0, "val_mae_pp"], color=COLORS["test"], label="selected")
    ax.set_xlabel("trial")
    ax.set_ylabel(f"best val MAE ({UNIT_LABEL})")
    ax.set_title(f"{name}: hyperparameter search")
    ax.legend()
    show(fig, name, "tuning")


def plot_loss_curves(name):
    """Training and validation loss per epoch, with the selected epoch marked."""
    history = pd.DataFrame(RESULTS[name]["run"]["history"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.epoch, history.train_loss, marker="o", color=COLORS["train"], label="train loss")
    ax.plot(history.epoch, history.val_loss, marker="o", color=COLORS["val"], label="validation loss")
    ax.axvline(RESULTS[name]["run"]["best_epoch"], ls="dashed", c="grey", label="selected epoch")
    ax.set_ylabel(f"{LOSS_TYPE} loss")
    ax.set_title(f"{name}: train and validation loss")
    ax.legend()
    integer_epochs(ax)
    show(fig, name, "loss_curves")


def plot_mae_curves(name):
    """Training and validation MAE per epoch next to the persistence MAE."""
    history = pd.DataFrame(RESULTS[name]["run"]["history"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history.epoch, history.train_mae_pp, marker="o", color=COLORS["train"], label="train MAE")
    ax.plot(history.epoch, history.val_mae_pp, marker="o", color=COLORS["val"], label="validation MAE")
    ax.plot(history.epoch, history.val_persistence_mae_pp, ls="dotted", color=COLORS["persistence"],
            label="persistence on validation")
    ax.axvline(RESULTS[name]["run"]["best_epoch"], ls="dashed", c="grey", alpha=0.6)
    ax.set_ylabel(f"MAE ({UNIT_LABEL})")
    ax.set_title(f"{name}: train and validation MAE")
    ax.legend(fontsize=8)
    integer_epochs(ax)
    show(fig, name, "mae_curves")


def plot_split_mae(name):
    """MAE of the model and both baselines on each split."""
    metrics = RESULTS[name]["metrics"]
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, series in enumerate(("model", "persistence", "rolling")):
        values = [metrics[split][series]["mae_pp"] for split in ("train", "val", "test")]
        ax.bar(np.arange(3) + i * width, values, width, color=COLORS[series], label=series)
    ax.set_xticks(np.arange(3) + width)
    ax.set_xticklabels(["train", "val", "test"])
    ax.set_ylabel(f"MAE ({UNIT_LABEL})")
    ax.set_title(f"{name}: MAE by split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    show(fig, name, "split_mae")


def plot_pred_vs_actual(name):
    test = RESULTS[name]["test"]
    actual, predicted = to_percent(test["y"]), to_percent(test["pred"])
    low = min(np.percentile(actual, 0.5), np.percentile(predicted, 0.5))
    high = max(np.percentile(actual, 99.5), np.percentile(predicted, 99.5))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hexes = ax.hexbin(actual, predicted, gridsize=70, bins="log", cmap="Blues", mincnt=1, extent=(low, high, low, high))
    ax.plot([low, high], [low, high], color="red", ls="dashed", lw=1, label="perfect prediction")
    ax.set_xlabel(f"actual {TARGET_NAME} (%)")
    ax.set_ylabel(f"predicted {TARGET_NAME} (%)")
    ax.set_title(f"{name}: predicted against actual, test")
    fig.colorbar(hexes, ax=ax, label="count (log scale)")
    ax.legend(loc="upper left")
    show(fig, name, "pred_vs_actual")


def plot_residuals(name):
    test = RESULTS[name]["test"]
    errors = (test["pred"] - test["y"]) * PP_PER_UNIT
    low, high = np.percentile(errors, [1, 99])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.clip(errors, low, high), bins=80, color=COLORS["model"], alpha=0.85)
    ax.axvline(0, c="k", lw=1)
    ax.axvline(errors.mean(), c="r", ls="dashed", label=f"mean error {errors.mean():+.2f} {UNIT_LABEL}")
    ax.set_xlabel(f"predicted minus actual ({UNIT_LABEL}), clipped to the 1st and 99th percentile")
    ax.set_ylabel("rows")
    ax.set_title(f"{name}: test residuals")
    ax.legend()
    show(fig, name, "residuals")


def plot_timeseries(name, n_services=3):
    """Actual, predicted and persistence values over the test window for the services with the most rows."""
    test = RESULTS[name]["test"]
    frame = pd.DataFrame({"nid": test["nid"], "minute": test["minute"], "actual": to_percent(test["y"]),
                          "predicted": to_percent(test["pred"]), "persistence": to_percent(test["persistence"])})
    busiest = frame.nid.value_counts().index[:n_services]
    fig, axes = plt.subplots(len(busiest), 1, figsize=(10, 3 * len(busiest)), sharex=True, squeeze=False)
    for ax, nid in zip(axes[:, 0], busiest):
        service = frame[frame.nid == nid].sort_values("minute")
        ax.plot(service.minute, service.actual, c="k", lw=1.4, label="actual")
        ax.plot(service.minute, service.predicted, c=COLORS["model"], lw=1.2, label=name)
        ax.plot(service.minute, service.persistence, c=COLORS["persistence"], lw=1, ls="dashed", label="persistence")
        ax.set_ylabel(f"{TARGET_NAME} (%)")
        ax.set_title(f"node_id {int(nid)}", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("minute (t_idx)")
    show(fig, name, "timeseries")


def plot_error_by_service(name):
    """Distribution of per service test MAE for the model and for persistence."""
    test = RESULTS[name]["test"]
    frame = pd.DataFrame({"nid": test["nid"], "model": np.abs(test["pred"] - test["y"]),
                          "persistence": np.abs(test["persistence"] - test["y"])})
    per_service = frame.groupby("nid").mean() * PP_PER_UNIT
    high = float(np.percentile(per_service.values, 99))
    bins = np.linspace(0, high, 50)
    share_better = float((per_service.model < per_service.persistence).mean())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.clip(per_service.persistence, 0, high), bins=bins, alpha=0.6, color=COLORS["persistence"],
            label="persistence")
    ax.hist(np.clip(per_service.model, 0, high), bins=bins, alpha=0.6, color=COLORS["model"], label=name)
    ax.set_xlabel(f"per service MAE ({UNIT_LABEL}), clipped at the 99th percentile")
    ax.set_ylabel("services")
    ax.set_title(f"{name}: better than persistence on {share_better:.0%} of {len(per_service):,} services")
    ax.legend()
    show(fig, name, "error_by_service")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. TGCN

# COMMAND ----------

# 9a. TGCN hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGCN")

# COMMAND ----------

# 9b. TGCN training
train_model("TGCN")

# COMMAND ----------

# 9c. TGCN evaluation
evaluate_model("TGCN")
print_diagnosis("TGCN")

# COMMAND ----------

# 9d. TGCN chart: hyperparameter search
plot_tuning("TGCN")

# COMMAND ----------

# 9e. TGCN chart: train and validation loss
plot_loss_curves("TGCN")

# COMMAND ----------

# 9f. TGCN chart: train and validation MAE
plot_mae_curves("TGCN")

# COMMAND ----------

# 9g. TGCN chart: MAE by split
plot_split_mae("TGCN")

# COMMAND ----------

# 9h. TGCN chart: predicted against actual
plot_pred_vs_actual("TGCN")

# COMMAND ----------

# 9i. TGCN chart: residuals
plot_residuals("TGCN")

# COMMAND ----------

# 9j. TGCN chart: busiest services over time
plot_timeseries("TGCN")

# COMMAND ----------

# 9k. TGCN chart: error by service
plot_error_by_service("TGCN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. TGN

# COMMAND ----------

# 10a. TGN hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGN")

# COMMAND ----------

# 10b. TGN training
train_model("TGN")

# COMMAND ----------

# 10c. TGN evaluation
evaluate_model("TGN")
print_diagnosis("TGN")

# COMMAND ----------

# 10d. TGN chart: hyperparameter search
plot_tuning("TGN")

# COMMAND ----------

# 10e. TGN chart: train and validation loss
plot_loss_curves("TGN")

# COMMAND ----------

# 10f. TGN chart: train and validation MAE
plot_mae_curves("TGN")

# COMMAND ----------

# 10g. TGN chart: MAE by split
plot_split_mae("TGN")

# COMMAND ----------

# 10h. TGN chart: predicted against actual
plot_pred_vs_actual("TGN")

# COMMAND ----------

# 10i. TGN chart: residuals
plot_residuals("TGN")

# COMMAND ----------

# 10j. TGN chart: busiest services over time
plot_timeseries("TGN")

# COMMAND ----------

# 10k. TGN chart: error by service
plot_error_by_service("TGN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. TGAT

# COMMAND ----------

# 11a. TGAT hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("TGAT")

# COMMAND ----------

# 11b. TGAT training
train_model("TGAT")

# COMMAND ----------

# 11c. TGAT evaluation
evaluate_model("TGAT")
print_diagnosis("TGAT")

# COMMAND ----------

# 11d. TGAT chart: hyperparameter search
plot_tuning("TGAT")

# COMMAND ----------

# 11e. TGAT chart: train and validation loss
plot_loss_curves("TGAT")

# COMMAND ----------

# 11f. TGAT chart: train and validation MAE
plot_mae_curves("TGAT")

# COMMAND ----------

# 11g. TGAT chart: MAE by split
plot_split_mae("TGAT")

# COMMAND ----------

# 11h. TGAT chart: predicted against actual
plot_pred_vs_actual("TGAT")

# COMMAND ----------

# 11i. TGAT chart: residuals
plot_residuals("TGAT")

# COMMAND ----------

# 11j. TGAT chart: busiest services over time
plot_timeseries("TGAT")

# COMMAND ----------

# 11k. TGAT chart: error by service
plot_error_by_service("TGAT")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. DCRNN

# COMMAND ----------

# 12a. DCRNN hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("DCRNN")

# COMMAND ----------

# 12b. DCRNN training
train_model("DCRNN")

# COMMAND ----------

# 12c. DCRNN evaluation
evaluate_model("DCRNN")
print_diagnosis("DCRNN")

# COMMAND ----------

# 12d. DCRNN chart: hyperparameter search
plot_tuning("DCRNN")

# COMMAND ----------

# 12e. DCRNN chart: train and validation loss
plot_loss_curves("DCRNN")

# COMMAND ----------

# 12f. DCRNN chart: train and validation MAE
plot_mae_curves("DCRNN")

# COMMAND ----------

# 12g. DCRNN chart: MAE by split
plot_split_mae("DCRNN")

# COMMAND ----------

# 12h. DCRNN chart: predicted against actual
plot_pred_vs_actual("DCRNN")

# COMMAND ----------

# 12i. DCRNN chart: residuals
plot_residuals("DCRNN")

# COMMAND ----------

# 12j. DCRNN chart: busiest services over time
plot_timeseries("DCRNN")

# COMMAND ----------

# 12k. DCRNN chart: error by service
plot_error_by_service("DCRNN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. GraphMixer

# COMMAND ----------

# 13a. GraphMixer hyperparameter search (skipped when run_tuning is off)
if RUN_TUNING:
    tune_model("GraphMixer")

# COMMAND ----------

# 13b. GraphMixer training
train_model("GraphMixer")

# COMMAND ----------

# 13c. GraphMixer evaluation
evaluate_model("GraphMixer")
print_diagnosis("GraphMixer")

# COMMAND ----------

# 13d. GraphMixer chart: hyperparameter search
plot_tuning("GraphMixer")

# COMMAND ----------

# 13e. GraphMixer chart: train and validation loss
plot_loss_curves("GraphMixer")

# COMMAND ----------

# 13f. GraphMixer chart: train and validation MAE
plot_mae_curves("GraphMixer")

# COMMAND ----------

# 13g. GraphMixer chart: MAE by split
plot_split_mae("GraphMixer")

# COMMAND ----------

# 13h. GraphMixer chart: predicted against actual
plot_pred_vs_actual("GraphMixer")

# COMMAND ----------

# 13i. GraphMixer chart: residuals
plot_residuals("GraphMixer")

# COMMAND ----------

# 13j. GraphMixer chart: busiest services over time
plot_timeseries("GraphMixer")

# COMMAND ----------

# 13k. GraphMixer chart: error by service
plot_error_by_service("GraphMixer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Model comparison on the test window
# MAGIC Run this after every model you want to compare has been evaluated.
# MAGIC The final table lists MSE, RMSE, MAE, R2, bias and skill against persistence for every model and both baselines.
# MAGIC Each value has a 95% interval from the per minute jackknife with the equal weighted cosine variance.

# COMMAND ----------

# Comparison tables


TABLE_METRICS = {"mse": "mse_pp2", "rmse": "rmse_pp", "mae": "mae_pp", "r2": "r2", "bias": "bias_pp"}
PERSISTENCE_SKILL = {"est": 0.0, "low": float("nan"), "high": float("nan"), "p": float("nan")}


def comparison_table():
    """Test metrics of both baselines and every evaluated model, each with its 95% interval."""
    baseline = next(iter(RESULTS.values()))["inference"]
    entries = [
        ("Persistence", baseline, "persistence", PERSISTENCE_SKILL),
        ("Rolling mean", baseline, "rolling", baseline["rolling_skill"])
    ]
    for name, result in RESULTS.items():
        entries.append((name, result["inference"], "model", result["inference"]["model_skill"]))

    rows = []
    for label, inference, series, skill in entries:
        row = {"model": label}
        for metric, column in TABLE_METRICS.items():
            summary = inference[f"{series}_{metric}"]
            row[column] = summary["est"]
            row[f"{column}_low"] = summary["low"]
            row[f"{column}_high"] = summary["high"]
        row["skill"] = skill["est"]
        row["skill_low"] = skill["low"]
        row["skill_high"] = skill["high"]
        row["skill_p"] = skill["p"]
        rows.append(row)
    return pd.DataFrame(rows)


def with_interval(table, column, digits):
    """Text such as 1.234 [1.100, 1.350] for one column of the comparison table."""
    texts = []
    for estimate, low, high in zip(table[column], table[f"{column}_low"], table[f"{column}_high"]):
        text = f"{estimate:.{digits}f}"
        if np.isfinite(low) and np.isfinite(high):
            text += f" [{low:.{digits}f}, {high:.{digits}f}]"
        texts.append(text)
    return texts


def final_table(table):
    """The comparison table in a readable form, one row per model and baseline."""
    return pd.DataFrame({
        "Model": table.model,
        f"MSE ({UNIT_LABEL}^2)": with_interval(table, "mse_pp2", 3),
        f"RMSE ({UNIT_LABEL})": with_interval(table, "rmse_pp", 3),
        f"MAE ({UNIT_LABEL})": with_interval(table, "mae_pp", 3),
        "R2": with_interval(table, "r2", 4),
        f"Bias ({UNIT_LABEL})": with_interval(table, "bias_pp", 3),
        "Skill vs persistence": with_interval(table, "skill", 4),
        "Skill p value": table.skill_p.map(lambda p: f"{p:.4g}" if np.isfinite(p) else "")
    })


def pairwise_tests():
    """MAE and MSE differences for every pair of models and baselines on the same test rows, Holm adjusted per loss."""
    names = list(RESULTS)
    first = RESULTS[names[0]]["test"]
    for name in names[1:]:
        other = RESULTS[name]["test"]
        assert np.array_equal(other["minute"], first["minute"]) and np.array_equal(other["nid"], first["nid"]), \
            "the models were scored on different rows"
    series = {"Persistence": first["persistence"], "Rolling mean": first["rolling"]}
    for name in names:
        series[name] = RESULTS[name]["test"]["pred"]
    sums, n_minutes = minute_sums(first, series)

    rows = []
    labels = list(series)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            mae = summarise_statistic(mae_difference_statistic(sums, a, b), n_minutes)
            mse = summarise_statistic(mse_difference_statistic(sums, a, b), n_minutes)
            rows.append({
                "a": a, "b": b,
                "mae_diff_pp": mae["est"], "mae_low": mae["low"], "mae_high": mae["high"], "mae_p": mae["p"],
                "mse_diff": mse["est"], "mse_low": mse["low"], "mse_high": mse["high"], "mse_p": mse["p"]
            })
    table = pd.DataFrame(rows)
    table["mae_p_holm"] = holm_adjust(table.mae_p.to_numpy())
    table["mse_p_holm"] = holm_adjust(table.mse_p.to_numpy())
    return table


def hyperparameter_table():
    rows = []
    for name in RESULTS:
        run = RESULTS[name]["run"]
        row = {"model": name, "tuned": run["tuned"], "max_epochs": MAX_EPOCHS[name], "best_epoch": run["best_epoch"],
               "epochs_run": run.get("epochs_run"), "train_seconds": round(run.get("train_seconds", float("nan")), 1)}
        row.update(run["params"])
        rows.append(row)
    return pd.DataFrame(rows)


COMPARISON = comparison_table()
FINAL_TABLE = final_table(COMPARISON)
PAIRS = pairwise_tests()
HYPERPARAMETERS = hyperparameter_table()
SEED_SPREAD = pd.concat([result["seeds"].assign(model=name) for name, result in RESULTS.items()], ignore_index=True)

print(f"Model comparison on the test window, {TARGET_NAME}, 95% intervals in brackets:")
print(FINAL_TABLE.to_string(index=False))
models_only = COMPARISON.iloc[2:]
best = models_only.loc[models_only.mse_pp2.idxmin(), "model"]
print(f"\nLowest test MSE: {best}. Use the Holm adjusted pairwise tests below to see which differences are significant.")
print("\nPairwise tests (a minus b, negative means a is better, Holm adjusted within each loss):")
print(PAIRS.round(4).to_string(index=False))
print("\nHyperparameters and selected epochs:")
print(HYPERPARAMETERS.to_string(index=False))
if len(SEEDS) > 1:
    print("\nSpread over seeds:")
    print(SEED_SPREAD.groupby("model")[["test_mae_pp", "test_skill"]].agg(["mean", "std"]).round(4).to_string())

# COMMAND ----------

# Comparison chart: test MAE with 95% intervals

fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = [COLORS["persistence"], COLORS["rolling"]] + [COLORS["model"]] * (len(COMPARISON) - 2)
error_bars = np.vstack([COMPARISON.mae_pp - COMPARISON.mae_pp_low, COMPARISON.mae_pp_high - COMPARISON.mae_pp])
ax.bar(COMPARISON.model, COMPARISON.mae_pp, color=colors, yerr=error_bars, capsize=5, ecolor="#333333")
ax.set_ylabel(f"test MAE ({UNIT_LABEL})")
ax.set_title(f"Test MAE with 95% intervals: {TARGET_NAME}")
ax.grid(axis="y", alpha=0.25)
show(fig, "comparison", "test_mae")

# COMMAND ----------

# Comparison chart: test RMSE with 95% intervals

fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = [COLORS["persistence"], COLORS["rolling"]] + [COLORS["model"]] * (len(COMPARISON) - 2)
error_bars = np.vstack([COMPARISON.rmse_pp - COMPARISON.rmse_pp_low, COMPARISON.rmse_pp_high - COMPARISON.rmse_pp])
ax.bar(COMPARISON.model, COMPARISON.rmse_pp, color=colors, yerr=error_bars, capsize=5, ecolor="#333333")
ax.set_ylabel(f"test RMSE ({UNIT_LABEL})")
ax.set_title(f"Test RMSE with 95% intervals: {TARGET_NAME}")
ax.grid(axis="y", alpha=0.25)
show(fig, "comparison", "test_rmse")

# COMMAND ----------

# Comparison chart: validation MAE per epoch for every model

fig, ax = plt.subplots(figsize=(8, 4.8))
for name, result in RESULTS.items():
    history = pd.DataFrame(result["run"]["history"])
    line, = ax.plot(history.epoch, history.val_mae_pp, marker="o", label=f"{name} validation")
    ax.plot(history.epoch, history.train_mae_pp, ls="dashed", alpha=0.6, color=line.get_color(), label=f"{name} train")
ax.set_ylabel(f"MAE ({UNIT_LABEL})")
ax.set_title("MAE per epoch, first seed (dashed lines are training)")
ax.legend(fontsize=8)
integer_epochs(ax)
show(fig, "comparison", "mae_by_epoch")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecasts for the load balancing notebook
# MAGIC
# MAGIC One row per service, target minute, model and seed, for the validation, calibration and test windows. Values are utilisation fractions (0 to 1),
# MAGIC converted back from the robust scale with the training median and IQR. `persistence` and `rolling` are the two baselines on the same rows.
# MAGIC `val` rows chose the hyperparameters and the epoch. `cal` rows were not used for any choice, so the load balancing notebook
# MAGIC calibrates its safety margins on them. Decisions are made on the `test` rows.

# COMMAND ----------

# Forecast export

from pyspark.sql import functions as F


def to_fraction(scaled):
    """Scaled value back to a utilisation fraction between 0 and 1."""
    return np.asarray(scaled, dtype=float) * TARGET_IQR + TARGET_MEDIAN


import hashlib

service_names = (
    spark.read.parquet(f"{NODE_INDEX_ROOT}/node_index")
    .select(F.col("node_id").cast("long").alias("node_id"), F.col("node_key").alias("msname"))
    .toPandas()
    .sort_values("node_id")
)
current_fingerprint = hashlib.sha256("\n".join(f"{key}:{node_id}" for key, node_id in zip(service_names.msname, service_names.node_id)).encode()).hexdigest()
stored_fingerprint = stats.get("node_index_fingerprint")
if stored_fingerprint:
    assert current_fingerprint == stored_fingerprint,         "node_index changed after this FE dataset was built. Rerun the FE notebooks before exporting forecasts."
else:
    print("dataset_stats has no node_index_fingerprint, so the node ids cannot be checked. Rerun the FE notebook to add it.")

forecast_parts = []
for name, result in RESULTS.items():
    for seed_rank, seed_result in enumerate(result["seed_results"]):
        for split in ("val", "cal", "test"):
            part = seed_result[split]
            forecast_parts.append(pd.DataFrame({
                "model": name,
                "pipeline": PIPELINE,
                "target": TARGET_COL,
                "split": split,
                "horizon": HORIZON,
                "seed": int(seed_result["seed"]),
                "seed_rank": seed_rank,
                "node_id": np.asarray(part["nid"], dtype=np.int64),
                "t_idx": np.asarray(part["minute"], dtype=np.int64),
                "actual": to_fraction(part["y"]),
                "predicted": to_fraction(part["pred"]),
                "persistence": to_fraction(part["persistence"]),
                "rolling": to_fraction(part["rolling"])
            }))
FORECASTS = pd.concat(forecast_parts, ignore_index=True).merge(service_names, on="node_id", how="left")
missing_names = int(FORECASTS.msname.isna().sum())
assert missing_names == 0, f"{missing_names} forecast rows have no msname in node_index"

FORECAST_PATH = f"{FE_ROOT}/{FILE_TAG}_forecasts"
spark.createDataFrame(FORECASTS).write.mode("overwrite").parquet(FORECAST_PATH)
print(f"{len(FORECASTS):,} forecast rows for {FORECASTS.model.nunique()} models written to {FORECAST_PATH}")
print(FORECASTS[FORECASTS.seed_rank == 0].groupby(["model", "split"]).size().unstack().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Save results

# COMMAND ----------

# Save

summary = {
    "target": TARGET_NAME,
    "pipeline": PIPELINE,
    "settings": {
        "horizon": HORIZON,
        "loss": LOSS_TYPE,
        "huber_beta": HUBER_BETA,
        "seeds": SEEDS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "tuning": RUN_TUNING,
        "tuning_trials": TUNING_TRIALS,
        "tuning_max_epochs": TUNING_MAX_EPOCHS,
        "target_median": TARGET_MEDIAN,
        "target_iqr": TARGET_IQR,
        "unit_scale": UNIT_SCALE,
        **EXTRA_SETTINGS
    },
    "hyperparameters": HYPERPARAMETERS.to_dict(orient="records"),
    "comparison": COMPARISON.to_dict(orient="records"),
    "final_table": FINAL_TABLE.to_dict(orient="records"),
    "pairwise_tests": PAIRS.to_dict(orient="records"),
    "models": {
        name: {
            "best_epoch": result["run"]["best_epoch"],
            "history": result["run"]["history"],
            "metrics": result["metrics"],
            "inference": result["inference"],
            "cold_rows": result["cold_rows"],
            "seeds": result["seeds"].to_dict(orient="records")
        }
        for name, result in RESULTS.items()
    }
}
summary_json = json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))

with open(os.path.join(OUT_DIR, f"{FILE_TAG}_results.json"), "w") as f:
    f.write(summary_json)
COMPARISON.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_comparison.csv"), index=False)
FINAL_TABLE.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_final_table.csv"), index=False)
PAIRS.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_pairwise_tests.csv"), index=False)
HYPERPARAMETERS.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_hyperparameters.csv"), index=False)
SEED_SPREAD.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_seeds.csv"), index=False)
for name, table in TUNING_RESULTS.items():
    table.to_csv(os.path.join(OUT_DIR, f"{FILE_TAG}_{name}_tuning.csv"), index=False)
print(f"results saved to {OUT_DIR}")

try:
    target_path = f"{FE_ROOT}/{FILE_TAG}_prediction_results"
    spark.createDataFrame([(summary_json,)], ["json"]).coalesce(1).write.mode("overwrite").text(target_path)
    print(f"also written to {target_path}")
except Exception as error:
    print(f"could not write to {FE_ROOT}: {str(error)[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC
# MAGIC **Epochs and tuning.** Set the maximum epochs of each model separately, for example `tgcn_max_epochs` or `graphmixer_max_epochs`.
# MAGIC Early stopping ends a run when the validation MAE has not improved for `patience` epochs. The best epoch is kept.
# MAGIC Each tuning trial runs for at most the model's own tuning epochs (`tgcn_tuning_epochs` and so on).
# MAGIC Short trials can favour configurations that learn fast, so raise the tuning epochs if the selected learning rate looks too high.
# MAGIC Edit `SEARCH_SPACE` to change what the search can pick. Set `run_tuning` to `no` to use `DEFAULT_PARAMS`.
# MAGIC
# MAGIC **Sampling is part of the experiment.** `event_filter` and `max_events_per_minute` decide how much of the call stream the models see.
# MAGIC Report them with the results. They are saved in the results file. The filter `touches_resource_node` drops calls between two services
# MAGIC that both lack resource data. Those calls only reach a predicted service through a neighbour without resource data, for example
# MAGIC through that neighbour's TGN memory, the second GCN layer of TGCN or the second attention layer of TGAT.
# MAGIC
# MAGIC **Order of calls.** Calls are sorted by their exact time and reach TGN and TGAT in batches of `event_batch_size`. Inside one batch the
# MAGIC memory update treats the calls together. Smaller batches give finer time resolution and run slower.
# MAGIC
# MAGIC **What the tests cover.** The intervals describe uncertainty from the finite test window. They do not include training randomness,
# MAGIC which the `seeds` widget measures, or the choice of hyperparameters and epoch, which are fixed on validation before the test window is used.
# MAGIC Read `mae_p_holm` and `mse_p_holm` for the pairwise comparisons.
# MAGIC
# MAGIC **DCRNN in this notebook.** The diffusion convolution and the DCGRU cell follow Li et al. (ICLR 2018). The default stacks two
# MAGIC cells, as in the paper, and `num_layers` can change that. The paper predicts a whole sequence with an encoder and a decoder trained
# MAGIC with scheduled sampling. Here the stacked cells give a one step forecast through the shared regression head, like TGCN.
# MAGIC The paper uses a fixed road network. Here the graph is rebuilt every minute from the sampled calls, and the edge weight is the number
# MAGIC of calls from caller to callee. `diffusion_steps` is K in the paper, so the longest walk has K - 1 steps.
# MAGIC A service with no outgoing or incoming calls in a minute gets zero from that walk direction, and longer walks lose weight when
# MAGIC they reach such a service. Every node is updated every minute. A service without data in a minute gets zero input, which after
# MAGIC robust scaling is the same as a median value. TGCN has the same limitation.
# MAGIC
# MAGIC **GraphMixer in this notebook.** The link encoder, the fixed time encoding (d = 100, alpha = beta = sqrt(d)) and the node encoder
# MAGIC follow Cong et al. (ICLR 2023). The paper predicts links. Here the encoder outputs feed the regression head instead of the link
# MAGIC classifier. The paper does not state the hidden sizes inside the mixer. Here the token mixing layer uses K / 2 units and the channel
# MAGIC mixing layer uses `hidden_dim` units. The calls of a service enter the mixer newest first. Time differences are in seconds, the unit
# MAGIC of the call timestamps. Node features change every minute here, so the node encoder uses the last observed features of each service.
# MAGIC The paper defines the node window as the time span of the last 2,000 interactions. Here that is the last `node_window_calls` sampled
# MAGIC calls, so it depends on `max_events_per_minute`. A neighbour is any service that called or was called by the service in that window,
# MAGIC counted once. The paper gives featureless nodes one hot features. That is not practical for this graph, so the mean only uses
# MAGIC neighbours with resource data.
# MAGIC
# MAGIC **Accuracy.** Accuracy counts correct class labels, so it is only defined for classification. This notebook predicts a continuous
# MAGIC value, so it reports MSE, RMSE, MAE, R2, bias and skill against persistence instead. The edge overload notebook is a classifier
# MAGIC and reports accuracy there.
# MAGIC
# MAGIC **Reading R2.** R2 is pooled over all services and minutes, so differences between services count towards it.
# MAGIC Persistence benefits from those differences too. Skill against persistence isolates what a model adds over the last observed value.
# MAGIC
# MAGIC **Data handling.** Targets and baselines use unclipped values. NaN targets are dropped. Rows of a service with no earlier observation are
# MAGIC left out of loss and metrics because persistence is undefined for them. The FE imputes missing utilisation per instance before averaging
# MAGIC and does not flag it, so some targets are partly imputed.
# MAGIC
# MAGIC **Units.** `target_median` and `target_iqr` only convert errors to percentage points. The event FE fits the same scaler as the snapshot FE
# MAGIC on the same data but does not print it, so the defaults are the values printed by the snapshot FE.
# MAGIC
# MAGIC **Memory.** The driver briefly holds the collected call table (its size is printed in section 2) and then 21 bytes per call of arrays.
# MAGIC Each TGN, TGAT or GraphMixer model also keeps a 13 byte per call log on its device. Use a high memory driver and a GPU.