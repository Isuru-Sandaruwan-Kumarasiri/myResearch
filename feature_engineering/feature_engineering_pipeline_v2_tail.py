# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Feature Engineering Pipeline — TGNN Node/Edge Dataset (CORRECTED)
# MAGIC
# MAGIC Reads ETL-stage Parquet tables (`resource`, `rtqps`, `callgraph_clean`), builds a
# MAGIC leakage-safe, cold-start-safe, msname-aligned node/edge feature dataset for a
# MAGIC Temporal GNN, and writes partitioned snapshot Parquet + manifests + dataset
# MAGIC stats.
# MAGIC
# MAGIC ### What this notebook fixes vs. the two prior versions
# MAGIC 1. **Node-key mismatch** — the exploratory notebook built `node_index` from
# MAGIC    `msinstanceid` (resource) unioned with `msname` (call graph UM/DM). Those are
# MAGIC    different ID spaces, so resource-based "nodes" never lined up with edges.
# MAGIC    Here, resource is aggregated to `msname` **before** the node index is built,
# MAGIC    so every node key is an `msname` — the same key call-graph edges use.
# MAGIC 2. **No missing-value handling** — the notebook never imputed nulls. Here we
# MAGIC    median-impute (stats fit on TRAIN rows only) before any aggregation/scaling.
# MAGIC 3. **Split-then-window cold start** — computing rolling/lag features *after*
# MAGIC    splitting by `t_idx` means the first rows of val/test have no real history
# MAGIC    and get null rolling stats. Here, all row-level feature engineering (scaling,
# MAGIC    rolling windows, cyclical encoding) runs on the **full, continuous timeline**;
# MAGIC    only the train-only rows are used to *fit* any statistic. The train/val/test
# MAGIC    split happens **last**, right before writing snapshots.
# MAGIC 4. **Cyclical period tied to the split boundary** — the notebook used
# MAGIC    `PERIOD = TRAIN_END`, which makes the sin/cos features change meaning if you
# MAGIC    ever move the split. Here the period is a fixed config constant describing
# MAGIC    the dataset's real span (default 720, i.e. 12h at 1-minute `t_idx` buckets
# MAGIC    per the Alibaba cluster-trace-microservices-v2021 schema).
# MAGIC 5. **One-hot-as-string edge categorical** — the notebook cast a `SparseVector`
# MAGIC    to a string for storage. Here `rpctype` is a plain ordinal index
# MAGIC    (`rpctype_idx`), meant to be fed through an `nn.Embedding` layer downstream —
# MAGIC    simpler, smaller, and doesn't require `SparseVector.parse()` on read.
# MAGIC 6. **`rt > 0` filter silently dropping valid data** — per the dataset docs, RPC/
# MAGIC    HTTP call `rt` is recorded as a *negative* integer on the DM side by design,
# MAGIC    not as an error. Filtering to `rt > 0` throws away every DM-side record.
# MAGIC    Here we aggregate on `rt_abs` (magnitude) and keep signed `rt` as an
# MAGIC    auxiliary feature instead of filtering it out.
# MAGIC 7. **Fixed, unbalanced split ratio** — the split cutoffs are now derived from
# MAGIC    `train_ratio`/`val_ratio` widgets applied to the observed min/max `t_idx`,
# MAGIC    defaulting to a 70/15/15 split, instead of hardcoded absolute t_idx values
# MAGIC    that silently gave ~90/7/3.
# MAGIC 8. **Storage target** — output can go to a plain S3/DBFS root, or, by setting
# MAGIC    `use_unity_catalog=true`, to a Unity Catalog **Volume**
# MAGIC    (`/Volumes/<catalog>/<schema>/<volume>/...`) for the raw partitioned Parquet
# MAGIC    snapshots, with `node_index` / `manifests` / `dataset_stats` also registered
# MAGIC    as governed managed **Delta tables** under `catalog.schema`.
# MAGIC
# MAGIC Run **after** `01_etl_pipeline.py` has completed successfully.

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

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import LongType
from pyspark.ml.feature import StringIndexer
import numpy as np
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widgets / Config
# MAGIC
# MAGIC Storage can be either a plain S3/DBFS root **or** a Unity Catalog Volume path
# MAGIC (`/Volumes/<catalog>/<schema>/<volume>/...`). Set `use_unity_catalog=true` to
# MAGIC target a Volume and also register the node index / manifests / stats as
# MAGIC managed Delta tables under `catalog.schema`.
# MAGIC
# MAGIC The train/val/test split is **ratio-based**, computed from the actual max
# MAGIC `t_idx` found in the data — not hardcoded absolute cutoffs. Default is a
# MAGIC 70/15/15 temporal split, which is a reasonable default for a single-trace
# MAGIC time-ordered dataset (val/test each large enough to evaluate on, most of the
# MAGIC cycle still available for training). Adjust the ratios to fit your needs, but
# MAGIC keep `train_ratio + val_ratio < 1.0` (test gets the remainder).

# COMMAND ----------

dbutils.widgets.text("etl_output_root", "s3://research-s20426/etl_output_root", "ETL output S3 root (input)")
dbutils.widgets.text("msrtqps_path", "s3://research-s20426/msrtqps_parquet/parquet_final/", "Corrected MSRTQps parquet folder")

dbutils.widgets.dropdown("use_unity_catalog", "false", ["true", "false"], "Store output in Unity Catalog Volume + Delta tables?")
dbutils.widgets.text("uc_catalog", "main", "Unity Catalog: catalog name")
dbutils.widgets.text("uc_schema", "tgnn", "Unity Catalog: schema name")
dbutils.widgets.text("uc_volume", "fe_dataset", "Unity Catalog: volume name (for raw snapshot files)")
dbutils.widgets.text("fe_output_root", "s3://research-s20426/tail_features/tgnn_dataset", "Feature-engineering output root (used when use_unity_catalog=false)")

dbutils.widgets.text("train_ratio", "0.70", "Train ratio of total t_idx span")
dbutils.widgets.text("val_ratio", "0.15", "Val ratio of total t_idx span (test gets the remainder)")
dbutils.widgets.text("cycle_period", "0", "Cyclical-encoding period in t_idx units; 0 = auto-detect from data (dataset span, NOT the split boundary)")
dbutils.widgets.text("rolling_window_size", "5", "Rolling window size in t_idx steps (was hardcoded to 5; see ACF diagnostic cell before changing)")

ETL_OUTPUT_ROOT = dbutils.widgets.get("etl_output_root")
MSRTQPS_PATH = dbutils.widgets.get("msrtqps_path")
USE_UC = dbutils.widgets.get("use_unity_catalog") == "true"
UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
TRAIN_RATIO = float(dbutils.widgets.get("train_ratio"))
VAL_RATIO = float(dbutils.widgets.get("val_ratio"))
CYCLE_PERIOD_OVERRIDE = int(dbutils.widgets.get("cycle_period"))
ROLLING_WINDOW_SIZE = int(dbutils.widgets.get("rolling_window_size"))

assert 0 < TRAIN_RATIO < 1 and 0 < VAL_RATIO < 1 and TRAIN_RATIO + VAL_RATIO < 1.0, \
    "train_ratio + val_ratio must be < 1.0 (test gets the remainder)"
assert ROLLING_WINDOW_SIZE >= 1, "rolling_window_size must be >= 1"

if USE_UC:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {UC_CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
    FE_OUTPUT_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/tail_features/tgnn_dataset"
else:
    FE_OUTPUT_ROOT = dbutils.widgets.get("fe_output_root")

PREVIOUS_APPROACH_ROOTS = {"s3://research-s20426/tgnn_dataset2", "s3://research-s20426/tgnn_event_dataset", "s3://research-s20426/load_balancing_work"}
assert FE_OUTPUT_ROOT.rstrip("/") not in PREVIOUS_APPROACH_ROOTS, "This is the tail feature version. It must not write into the folders of the previous approach."
print(f"etl_output_root = {ETL_OUTPUT_ROOT}")
print(f"msrtqps_path    = {MSRTQPS_PATH}")
print(f"fe_output_root  = {FE_OUTPUT_ROOT}   (unity_catalog={USE_UC})")
print(f"train_ratio={TRAIN_RATIO}  val_ratio={VAL_RATIO}  test_ratio={round(1 - TRAIN_RATIO - VAL_RATIO, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load ETL Output

# COMMAND ----------

res = spark.read.parquet(f"{ETL_OUTPUT_ROOT}/resource")
rtq = spark.read.parquet(MSRTQPS_PATH)
cg = spark.read.parquet(f"{ETL_OUTPUT_ROOT}/callgraph_1")

# Guarantee t_idx exists (60s recording interval -> minute buckets)

res = res.withColumn("t_idx", (F.col("timestamp") / 60000).cast("int"))

rtq = rtq.withColumn("t_idx", (F.col("timestamp") / 60000).cast("int"))

cg = cg.withColumn("t_idx", (F.col("timestamp") / 60000).cast("int"))

def nan_to_null(df):
    """Spark sorts NaN above every number and isNull() does not catch it, so NaN becomes null before any statistic."""
    for column_name, data_type in df.dtypes:
        if data_type in ("double", "float"):
            df = df.withColumn(column_name, F.when(~F.isnan(F.col(column_name)), F.col(column_name)))
    return df


res = nan_to_null(res)
rtq = nan_to_null(rtq)
cg = nan_to_null(cg)

table_bounds = {}
for table_name, table in (("MSResource", res), ("MSRTQps", rtq), ("MSCallGraph", cg)):
    table_bounds[table_name] = tuple(table.agg(F.min("t_idx"), F.max("t_idx")).first())
    print(f"{table_name} minutes: {table_bounds[table_name][0]} to {table_bounds[table_name][1]}")
if table_bounds["MSResource"] != table_bounds["MSRTQps"]:
    print("WARNING: MSResource and MSRTQps cover different minutes. The split boundaries use the widest range, so check the MSRTQps conversion.")

print("res schema:"); res.printSchema()
print("rtq schema:"); rtq.printSchema()
print("cg  schema:"); cg.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Derive split cutoffs from the observed data span (ratio-based, not hardcoded)

# COMMAND ----------

min_t_idx = min(
    res.agg(F.min("t_idx")).first()[0],
    rtq.agg(F.min("t_idx")).first()[0],
    cg.agg(F.min("t_idx")).first()[0],
)
max_t_idx = max(
    res.agg(F.max("t_idx")).first()[0],
    rtq.agg(F.max("t_idx")).first()[0],
    cg.agg(F.max("t_idx")).first()[0],
)
span = max_t_idx - min_t_idx

TRAIN_END = min_t_idx + int(round(span * TRAIN_RATIO))
VAL_END = min_t_idx + int(round(span * (TRAIN_RATIO + VAL_RATIO)))
TEST_END = max_t_idx

CYCLE_PERIOD = CYCLE_PERIOD_OVERRIDE if CYCLE_PERIOD_OVERRIDE > 0 else span

print(f"Observed t_idx range: [{min_t_idx}, {max_t_idx}]  (span={span})")
print(f"-> TRAIN_END={TRAIN_END}  VAL_END={VAL_END}  TEST_END={TEST_END}  CYCLE_PERIOD={CYCLE_PERIOD}")
print(f"-> train={TRAIN_END - min_t_idx} steps ({TRAIN_RATIO:.0%}), "
      f"val={VAL_END - TRAIN_END} steps ({VAL_RATIO:.0%}), "
      f"test={TEST_END - VAL_END} steps ({1 - TRAIN_RATIO - VAL_RATIO:.0%})")

TRAIN_COND = F.col("t_idx") < TRAIN_END  # used only to compute fit statistics, never to filter before FE

# COMMAND ----------

# MAGIC %md
# MAGIC #### Detect RTQ metric columns (MCR / RT) actually present after ETL's pivot

# COMMAND ----------

rtq_metric_cols = [c for c in rtq.columns if c not in ("timestamp", "msname", "msinstanceid", "t_idx")]
mcr_cols = [c for c in rtq_metric_cols if c.endswith("_MCR")]
rt_cols = [c for c in rtq_metric_cols if c.endswith("_RT")]
all_rtq_cols = mcr_cols + rt_cols

print("RTQ wide columns available:", rtq_metric_cols)
print("Using MCR_COLS:", mcr_cols)
print("Using RT_COLS :", rt_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Missing Value Imputation
# MAGIC Fit medians on **TRAIN rows only** (`t_idx < TRAIN_END`), apply the same fixed
# MAGIC values across the whole (still unsplit) table. No statistic is ever computed on
# MAGIC val/test rows.

# COMMAND ----------

def fit_median_imputer(df, cols, train_cond):
    """Fit per-column medians using only rows satisfying train_cond."""
    train_df = df.filter(train_cond)
    stats = {}
    for c in cols:
        med = train_df.approxQuantile(c, [0.5], 0.001)
        stats[c] = med[0] if med else 0.0
    return stats


def apply_median_imputer(df, stats):
    for c, med in stats.items():
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(med)).otherwise(F.col(c)))
    return df



def fit_service_median_imputer(df, cols, train_cond):
    """Per service TRAIN medians, with the overall TRAIN median as the fallback for services without TRAIN values."""
    train_df = df.filter(train_cond)
    service_medians = train_df.groupBy("msname").agg(*[F.percentile_approx(c, 0.5, 10000).alias(f"{c}_service_median") for c in cols])
    overall = {c: (train_df.approxQuantile(c, [0.5], 0.001) or [0.0])[0] for c in cols}
    return service_medians, overall


def apply_service_median_imputer(df, cols, service_medians, overall):
    df = df.join(service_medians, "msname", "left")
    for c in cols:
        df = df.withColumn(c, F.coalesce(F.col(c), F.col(f"{c}_service_median"), F.lit(overall[c])))
    return df.drop(*[f"{c}_service_median" for c in cols])


RES_IMPUTE_COLS = ["cpu_utilization", "memory_utilization"]
missing_before = res.select(*[F.mean(F.col(c).isNull().cast("double")).alias(c) for c in RES_IMPUTE_COLS]).first().asDict()
res_service_medians, res_overall_medians = fit_service_median_imputer(res, RES_IMPUTE_COLS, TRAIN_COND)
res = apply_service_median_imputer(res, RES_IMPUTE_COLS, res_service_medians, res_overall_medians)
res_impute_stats = res_overall_medians
print("missing resource share before imputation:", missing_before, "| filled with the TRAIN median of the same service")


# MSRTQps cleaning. Every statistic is fitted on TRAIN rows only.
# 1. Some RT columns repeat the MCR value instead of a response time. They are found on TRAIN rows that have calls (MCR > 0)
#    and dropped, so rows without calls (0 == 0) cannot make a real RT column look like a copy.
# 2. A missing MCR means no calls of that paradigm in that record, so it becomes 0 calls per second, not a median.
#    The check below prints how often an RT is present while its MCR is missing. A share near 0 supports this rule.
# 3. RT is undefined without calls. It is set to 0 when MCR is 0, and the MCR feature already tells the model there were no calls.
#    An RT that is missing while MCR > 0 is filled with the TRAIN median of rows with calls.
SUSPECT_RT_SHARE = 0.5
dropped_rt_cols = []
train_rtq = rtq.filter(TRAIN_COND)
for rt_col in list(rt_cols):
    mcr_col = rt_col.replace("_RT", "_MCR")
    if mcr_col in mcr_cols:
        with_calls = train_rtq.filter(F.col(rt_col).isNotNull() & (F.col(mcr_col) > 0))
        equal_share = with_calls.select(F.mean((F.col(rt_col) == F.col(mcr_col)).cast("double"))).first()[0]
        if equal_share is not None and equal_share > SUSPECT_RT_SHARE:
            dropped_rt_cols.append(rt_col)
rt_cols = [c for c in rt_cols if c not in dropped_rt_cols]
print("RT columns dropped because they repeat the MCR value:", dropped_rt_cols)

for mcr_col in mcr_cols:
    rt_col = mcr_col.replace("_MCR", "_RT")
    if rt_col in rt_cols:
        share = train_rtq.select(F.mean((F.col(mcr_col).isNull() & F.col(rt_col).isNotNull()).cast("double"))).first()[0]
        print(f"{mcr_col}: share of TRAIN rows with a missing MCR but a present RT = {share:.5f}")

rtq = rtq.drop(*dropped_rt_cols)
all_rtq_cols = mcr_cols + rt_cols
rtq = rtq.fillna(0.0, subset=mcr_cols)
for rt_col in rt_cols:
    mcr_col = rt_col.replace("_RT", "_MCR")
    if mcr_col in mcr_cols:
        rtq = rtq.withColumn(rt_col, F.when(F.col(mcr_col) == 0, F.lit(0.0)).otherwise(F.col(rt_col)))
rtq_impute_stats = {}
for rt_col in rt_cols:
    mcr_col = rt_col.replace("_RT", "_MCR")
    condition = TRAIN_COND & (F.col(mcr_col) > 0) if mcr_col in mcr_cols else TRAIN_COND
    rtq_impute_stats.update(fit_median_imputer(rtq, [rt_col], condition))
rtq = apply_median_imputer(rtq, rtq_impute_stats)
print("MCR nulls set to 0 calls per second, RT set to 0 without calls. RT TRAIN medians of rows with calls:", rtq_impute_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3 — Resource Feature Engineering
# MAGIC Aggregate to `msname` (the same key call-graph edges use), scale (train-only
# MAGIC fit), then compute rolling/lag features and cyclical time encoding on the
# MAGIC **full continuous timeline** — the split happens after this, not before.

# COMMAND ----------

def compute_robust_stats(df, col_name):
    q1, q2, q3 = df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
    iqr = q3 - q1
    if iqr < 1e-8:
        iqr = 1.0
    return q2, iqr


def robust_scale_column(df, col_name, q2, iqr, output_col=None):
    output_col = output_col or f"{col_name}_rs"
    return df.withColumn(output_col, (F.col(col_name) - F.lit(q2)) / F.lit(iqr))


# --- aggregate instance-level resource to msname-level, per t_idx ---
res_ms = (
    res.groupBy("msname", "t_idx")
    .agg(
        F.mean("cpu_utilization").alias("cpu_utilization"),
        F.mean("memory_utilization").alias("memory_utilization"),
    )
)

# --- fit scaler stats on TRAIN rows only, apply to the full (unsplit) table ---
print("Fitting resource scaler on TRAIN rows only...")
cpu_q2, cpu_iqr = compute_robust_stats(res_ms.filter(TRAIN_COND), "cpu_utilization")
mem_q2, mem_iqr = compute_robust_stats(res_ms.filter(TRAIN_COND), "memory_utilization")
print(f"cpu: q2={cpu_q2:.4f} iqr={cpu_iqr:.4f}   memory: q2={mem_q2:.4f} iqr={mem_iqr:.4f}")

res_ms = robust_scale_column(res_ms, "cpu_utilization", cpu_q2, cpu_iqr, "cpu_rs")
res_ms = robust_scale_column(res_ms, "memory_utilization", mem_q2, mem_iqr, "memory_rs")


# Tail and count features: highest container, 95th percentile and spread across containers, and the number of containers.
# The target stays the mean over containers. These columns are model inputs.
instance_minute = res.groupBy("msname", "msinstanceid", "t_idx").agg(
    F.mean("cpu_utilization").alias("cpu_utilization"),
    F.mean("memory_utilization").alias("memory_utilization")
)
res_tail = instance_minute.groupBy("msname", "t_idx").agg(
    F.max("cpu_utilization").alias("cpu_max"),
    F.percentile_approx("cpu_utilization", 0.95, 1000).alias("cpu_p95"),
    F.coalesce(F.stddev("cpu_utilization"), F.lit(0.0)).alias("cpu_std"),
    F.max("memory_utilization").alias("memory_max"),
    F.percentile_approx("memory_utilization", 0.95, 1000).alias("memory_p95"),
    F.coalesce(F.stddev("memory_utilization"), F.lit(0.0)).alias("memory_std"),
    F.log1p(F.countDistinct("msinstanceid")).alias("instance_count_log")
)
res_ms = res_ms.join(res_tail, ["msname", "t_idx"], "left")
TAIL_RESOURCE_RAW = ["cpu_max", "cpu_p95", "cpu_std", "memory_max", "memory_p95", "memory_std", "instance_count_log"]
tail_scale_stats = {}
for column in TAIL_RESOURCE_RAW:
    q2, iqr = compute_robust_stats(res_ms.filter(TRAIN_COND), column)
    tail_scale_stats[column] = (q2, iqr)
    res_ms = robust_scale_column(res_ms, column, q2, iqr, f"{column}_rs")
TAIL_NODE_FEATURE_COLS = [f"{column}_rs" for column in TAIL_RESOURCE_RAW]
print("tail node features:", TAIL_NODE_FEATURE_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rolling / lag features — computed on the full continuous timeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### ACF/PACF diagnostic — empirical justification for `rolling_window_size`
# MAGIC
# MAGIC Rather than hardcoding the rolling window at 5 steps, compute the
# MAGIC autocorrelation function (ACF) of the robust-scaled series on TRAIN rows
# MAGIC and pick the lag where autocorrelation decays below a threshold (default
# MAGIC 0.2). Averages ACF across the busiest `msname`s (by row count), since ACF
# MAGIC on one short per-node series is noisy. This only *informs* the choice —
# MAGIC the actual validation is a grid search over `rolling_window_size`
# MAGIC evaluated on downstream time-ordered validation loss, not train loss.

# COMMAND ----------

def suggest_window_from_acf(df, value_col, node_col="msname", time_col="t_idx",
                              train_cond=None, top_n_nodes=20, max_lag=20, acf_threshold=0.2):
    """Average ACF across the busiest nodes' TRAIN-only series; suggest the lag
    where mean autocorrelation first drops below acf_threshold."""
    from statsmodels.tsa.stattools import acf
    import pandas as pd

    base = df.filter(train_cond) if train_cond is not None else df
    busiest = [
        row[0] for row in
        base.groupBy(node_col).count().orderBy(F.desc("count")).limit(top_n_nodes)
        .select(node_col).collect()
    ]
    pdf = (
        base.filter(F.col(node_col).isin(busiest))
        .select(node_col, time_col, value_col)
        .orderBy(node_col, time_col)
        .toPandas()
    )

    acf_curves = []
    for _, g in pdf.groupby(node_col):
        series = g.sort_values(time_col)[value_col].to_numpy()
        if len(series) > max_lag * 2:
            vals = acf(series, nlags=max_lag, fft=True)
            acf_curves.append(vals)
    if not acf_curves:
        print("Not enough per-node history for ACF; keeping default window.")
        return None, None

    mean_acf = np.mean(np.vstack(acf_curves), axis=0)
    suggested_lag = next((lag for lag, v in enumerate(mean_acf) if lag > 0 and v < acf_threshold), max_lag)
    print("Mean ACF by lag (t_idx steps):")
    for lag, v in enumerate(mean_acf):
        marker = "  <-- suggested window" if lag == suggested_lag else ""
        print(f"  lag {lag:2d}: {v: .3f}{marker}")
    return suggested_lag, mean_acf


try:
    suggested_cpu_lag, cpu_acf = suggest_window_from_acf(res_ms, "cpu_rs", train_cond=TRAIN_COND)
    suggested_mem_lag, mem_acf = suggest_window_from_acf(res_ms, "memory_rs", train_cond=TRAIN_COND)
    if suggested_cpu_lag is not None and suggested_mem_lag is not None:
        print(f"\nACF-suggested window — cpu: {suggested_cpu_lag} steps, memory: {suggested_mem_lag} steps")
        print(f"Currently configured rolling_window_size = {ROLLING_WINDOW_SIZE}")
except ImportError:
    print("statsmodels not installed on this cluster — run `%pip install statsmodels` in a cell "
          "above and re-run, or skip this diagnostic and rely on the grid search instead.")

# COMMAND ----------

roll_window = Window.partitionBy("msname").orderBy("t_idx").rowsBetween(-(ROLLING_WINDOW_SIZE - 1), 0)
lag_window = Window.partitionBy("msname").orderBy("t_idx")
row_num_window = Window.partitionBy("msname").orderBy("t_idx")


def add_temporal_features(df, base_col, prefix):
    df = df.withColumn(f"{prefix}_rolling_mean", F.avg(base_col).over(roll_window))
    df = df.withColumn(f"{prefix}_rolling_std", F.stddev(base_col).over(roll_window))
    # stddev of a 1-row window is null in Spark (not 0) -> fill explicitly instead of
    # silently leaking nulls into the very first snapshot of every node.
    df = df.withColumn(
        f"{prefix}_rolling_std",
        F.when(F.col(f"{prefix}_rolling_std").isNull(), 0.0).otherwise(F.col(f"{prefix}_rolling_std")),
    )
    lag1 = F.lag(base_col, 1).over(lag_window)
    df = df.withColumn(
        f"{prefix}_delta",
        F.when(lag1.isNull(), 0.0).otherwise(F.col(base_col) - lag1),
    )
    return df


res_ms = add_temporal_features(res_ms, "cpu_rs", "cpu")
res_ms = add_temporal_features(res_ms, "memory_rs", "memory")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cyclical time encoding — period is the dataset span, not TRAIN_END

# COMMAND ----------

def add_cyclical_time_features(df, period):
    df = df.withColumn("t_sin", F.sin(2 * np.pi * F.col("t_idx") / period))
    df = df.withColumn("t_cos", F.cos(2 * np.pi * F.col("t_idx") / period))
    return df


res_ms = add_cyclical_time_features(res_ms, CYCLE_PERIOD)

RES_FEATURE_COLS = [
    "cpu_rs", "cpu_rolling_mean", "cpu_rolling_std", "cpu_delta",
    "memory_rs", "memory_rolling_mean", "memory_rolling_std", "memory_delta",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4 — RTQ Feature Engineering
# MAGIC Aggregate to msname, log-transform, scale (train-only fit) — again on the full
# MAGIC continuous timeline.

# COMMAND ----------

rtq_ms = rtq.groupBy("msname", "t_idx").agg(*[F.mean(c).alias(c) for c in all_rtq_cols])

# log1p is defined at 0 (log1p(0) = 0), so no artificial epsilon floor is needed.
# Defensive clip in case of any residual negative values from upstream.
for c in all_rtq_cols:
    rtq_ms = rtq_ms.withColumn(c, F.greatest(F.col(c), F.lit(0.0)))
    rtq_ms = rtq_ms.withColumn(f"{c}_log", F.log1p(F.col(c)))

RTQ_LOG_COLS = [f"{c}_log" for c in all_rtq_cols]

print("Fitting RTQ scaler on TRAIN rows only...")
rtq_scale_stats = {}
for c in RTQ_LOG_COLS:
    q2, iqr = compute_robust_stats(rtq_ms.filter(TRAIN_COND), c)
    rtq_scale_stats[c] = (q2, iqr)
    rtq_ms = robust_scale_column(rtq_ms, c, q2, iqr, f"{c}_rs")

RTQ_SCALED_COLS = [f"{c}_rs" for c in RTQ_LOG_COLS]
print("RTQ scaled columns:", RTQ_SCALED_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5 — Global Node Index (msname-keyed, consistent with call-graph edges)
# MAGIC Built from the union of all `msname`s across resource and call-graph — over the
# MAGIC full unsplit timeline. This is vocabulary construction (node identity), not a
# MAGIC fitted statistic, so including val/test rows here is not a leakage issue.

# COMMAND ----------

all_msnames = (
    res_ms.select(F.col("msname").alias("node_key"))
    .union(cg.select(F.col("UM").alias("node_key")))
    .union(cg.select(F.col("DM").alias("node_key")))
    .distinct()
)

# row_number gives 0 to N - 1 without gaps. monotonically_increasing_id leaves gaps between partitions,
# and the models index tensors of size num_nodes with node_id.
node_index = all_msnames.withColumn("node_id", (F.row_number().over(Window.orderBy("node_key")) - 1).cast(LongType()))

node_index.write.mode("overwrite").parquet(f"{FE_OUTPUT_ROOT}/node_index")
node_index = spark.read.parquet(f"{FE_OUTPUT_ROOT}/node_index")

node_id_check = node_index.agg(F.min("node_id"), F.max("node_id"), F.countDistinct("node_id"), F.count("*")).first()
assert node_id_check[0] == 0 and node_id_check[1] == node_id_check[3] - 1 and node_id_check[2] == node_id_check[3], \
    f"node ids must be 0 to N - 1 without gaps or repeats, got min {node_id_check[0]}, max {node_id_check[1]}, " \
    f"{node_id_check[2]} distinct of {node_id_check[3]}"
print(f"node ids are dense: 0 to {node_id_check[1]}")

if USE_UC:
    # Also register as a managed Delta table for governed SQL access / lineage.
    (
        node_index.write.format("delta").mode("overwrite")
        .saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.tail_node_index")
    )
    print(f"Registered {UC_CATALOG}.{UC_SCHEMA}.tail_node_index as a managed Delta table.")


import hashlib
fingerprint_rows = node_index.select("node_key", "node_id").orderBy("node_id").toPandas()
NODE_INDEX_FINGERPRINT = hashlib.sha256("\n".join(f"{key}:{node_id}" for key, node_id in zip(fingerprint_rows.node_key, fingerprint_rows.node_id)).encode()).hexdigest()
print("node_index fingerprint:", NODE_INDEX_FINGERPRINT)
print(f"num_nodes = {node_index.count():,}")
node_index.limit(5).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify node-index coverage (should be ~100% since node_index is built from these exact keys)

# COMMAND ----------

um_count = cg.select("UM").distinct().count()
dm_count = cg.select("DM").distinct().count()
res_msname_count = res_ms.select("msname").distinct().count()

um_hits = cg.select(F.col("UM").alias("node_key")).distinct().join(node_index, "node_key").count()
dm_hits = cg.select(F.col("DM").alias("node_key")).distinct().join(node_index, "node_key").count()
res_hits = res_ms.select(F.col("msname").alias("node_key")).distinct().join(node_index, "node_key").count()

print(f"UM  coverage : {um_hits:,} / {um_count:,}")
print(f"DM  coverage : {dm_hits:,} / {dm_count:,}")
print(f"res coverage : {res_hits:,} / {res_msname_count:,}")
assert um_hits == um_count and dm_hits == dm_count and res_hits == res_msname_count, \
    "Node index does not fully cover msname keys — investigate before continuing."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6 — Build Final Node Feature Table (still unsplit)

# COMMAND ----------

ALL_NODE_FEATURE_COLS = RES_FEATURE_COLS + RTQ_SCALED_COLS + ["t_sin", "t_cos"] + TAIL_NODE_FEATURE_COLS
print(f"Total node feature dims: {len(ALL_NODE_FEATURE_COLS)}")
for i, c in enumerate(ALL_NODE_FEATURE_COLS, 1):
    print(f"  {i:2d}. {c}")

node_feat = (
    res_ms.alias("r")
    .join(rtq_ms.alias("q"), on=["msname", "t_idx"], how="left")
    .join(node_index, F.col("r.msname") == node_index["node_key"], how="inner")
    .select(
        F.col("node_id"),
        F.col("r.msname").alias("msname"),
        F.col("r.t_idx").alias("t_idx"),
        *[F.col(c) for c in RES_FEATURE_COLS],
        *[F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in RTQ_SCALED_COLS],
        F.col("r.t_sin").alias("t_sin"),
        F.col("r.t_cos").alias("t_cos"),
        *[F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in TAIL_NODE_FEATURE_COLS],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 7 — Call Graph (Edge) Feature Engineering
# MAGIC `rt` can legitimately be negative on the DM side of a call — filtering to
# MAGIC `rt > 0` would silently drop every DM-side record. We aggregate on `rt_abs`
# MAGIC (magnitude) and keep signed `rt` as an auxiliary feature instead.

# COMMAND ----------

if "rt_abs" not in cg.columns:
    cg = cg.withColumn("rt_abs", F.abs(F.col("rt")))

cg_clean = (
    cg.groupBy("UM", "DM", "t_idx", "rpctype")
    .agg(
        F.mean("rt").alias("rt_mean"),
        F.mean("rt_abs").alias("rt_abs_mean"),
        F.log1p(F.count("*")).alias("call_count_log"),
        F.log1p(F.percentile_approx("rt_abs", 0.95, 1000)).alias("rt_abs_p95_log"),
        F.log1p(F.coalesce(F.stddev("rt_abs"), F.lit(0.0))).alias("rt_abs_std_log"),
    )
)

print("Fitting rpctype encoder on TRAIN rows only...")
rpctype_indexer = StringIndexer(inputCol="rpctype", outputCol="rpctype_idx", handleInvalid="keep")
rpctype_model = rpctype_indexer.fit(cg_clean.filter(TRAIN_COND))
cg_clean = rpctype_model.transform(cg_clean)

print("Fitting edge scaler on TRAIN rows only...")
EDGE_RAW_COLS = ["rt_mean", "rt_abs_mean"]
edge_scale_stats = {}
for c in EDGE_RAW_COLS:
    q2, iqr = compute_robust_stats(cg_clean.filter(TRAIN_COND), c)
    edge_scale_stats[c] = (q2, iqr)
    cg_clean = robust_scale_column(cg_clean, c, q2, iqr, f"{c}_rs")

EDGE_SCALED_COLS = [f"{c}_rs" for c in EDGE_RAW_COLS]
EDGE_FEATURE_COLS = EDGE_SCALED_COLS + ["rpctype_idx"]
print("Edge feature columns:", EDGE_FEATURE_COLS)

# Sampled call count and response time tail of the edge, used as edge features.
TAIL_EDGE_RAW = ["call_count_log", "rt_abs_p95_log", "rt_abs_std_log"]
for c in TAIL_EDGE_RAW:
    q2, iqr = compute_robust_stats(cg_clean.filter(TRAIN_COND), c)
    edge_scale_stats[c] = (q2, iqr)
    cg_clean = robust_scale_column(cg_clean, c, q2, iqr, f"{c}_rs")
TAIL_EDGE_FEATURE_COLS = [f"{c}_rs" for c in TAIL_EDGE_RAW]
EDGE_FEATURE_COLS = EDGE_SCALED_COLS + TAIL_EDGE_FEATURE_COLS + ["rpctype_idx"]
print("Edge feature columns with the tail features:", EDGE_FEATURE_COLS)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join edges to the msname-keyed node index

# COMMAND ----------

src_lookup = node_index.select(F.col("node_key"), F.col("node_id").alias("src_node_id"))
dst_lookup = node_index.select(F.col("node_key"), F.col("node_id").alias("dst_node_id"))

edge_feat = (
    cg_clean
    .join(src_lookup, cg_clean["UM"] == src_lookup["node_key"], "left").drop("node_key")
    .join(dst_lookup, cg_clean["DM"] == dst_lookup["node_key"], "left").drop("node_key")
    .select(
        F.col("src_node_id"),
        F.col("dst_node_id"),
        F.col("t_idx"),
        F.col("rpctype"),
        *[F.col(c) for c in EDGE_FEATURE_COLS],
    )
    .filter(F.col("src_node_id").isNotNull() & F.col("dst_node_id").isNotNull())
)

null_src = cg_clean.join(src_lookup, cg_clean["UM"] == src_lookup["node_key"], "left") \
    .filter(F.col("src_node_id").isNull()).count()
print(f"Edge rows dropped for unresolved src/dst node id: {null_src:,} (should be ~0 given full node index)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 8 — Temporal Split (applied LAST, after all row-level FE is complete)

# COMMAND ----------

def temporal_split(df, train_end, val_end, test_end):
    train = df.filter(F.col("t_idx") < train_end)
    val = df.filter((F.col("t_idx") >= train_end) & (F.col("t_idx") < val_end))
    test = df.filter((F.col("t_idx") >= val_end) & (F.col("t_idx") <= test_end))
    return train, val, test


node_feat_train, node_feat_val, node_feat_test = temporal_split(node_feat, TRAIN_END, VAL_END, TEST_END)
edge_feat_train, edge_feat_val, edge_feat_test = temporal_split(edge_feat, TRAIN_END, VAL_END, TEST_END)

print(f"node_feat — train: {node_feat_train.count():,}  val: {node_feat_val.count():,}  test: {node_feat_test.count():,}")
print(f"edge_feat — train: {edge_feat_train.count():,}  val: {edge_feat_val.count():,}  test: {edge_feat_test.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 9 — Write Snapshots + Manifests + Dataset Stats

# COMMAND ----------

def write_snapshots(node_df, edge_df, split_name, output_root):
    node_cols = [c for c in node_df.columns if c != "t_idx"]
    (
        node_df.select("t_idx", *node_cols)
        .write.mode("overwrite").partitionBy("t_idx").option("compression", "snappy")
        .parquet(f"{output_root}/snapshots/{split_name}/nodes")
    )

    edge_cols = [c for c in edge_df.columns if c != "t_idx"]
    (
        edge_df.select("t_idx", *edge_cols)
        .write.mode("overwrite").partitionBy("t_idx").option("compression", "snappy")
        .parquet(f"{output_root}/snapshots/{split_name}/edges")
    )

    node_parts = dbutils.fs.ls(f"{output_root}/snapshots/{split_name}/nodes")
    edge_parts = dbutils.fs.ls(f"{output_root}/snapshots/{split_name}/edges")
    n_node_parts = sum(1 for p in node_parts if "t_idx=" in p.name)
    n_edge_parts = sum(1 for p in edge_parts if "t_idx=" in p.name)
    print(f"[{split_name}] node snapshots: {n_node_parts} partitions")
    print(f"[{split_name}] edge snapshots: {n_edge_parts} partitions")


write_snapshots(node_feat_train, edge_feat_train, "train", FE_OUTPUT_ROOT)
write_snapshots(node_feat_val, edge_feat_val, "val", FE_OUTPUT_ROOT)
write_snapshots(node_feat_test, edge_feat_test, "test", FE_OUTPUT_ROOT)

# COMMAND ----------

def write_manifest(node_df, split_name, output_root):
    t_indices = (
        node_df.select("t_idx").distinct().orderBy("t_idx")
        .toPandas()["t_idx"].astype(int).tolist()
    )
    manifest = {
        "split": split_name,
        "t_indices": t_indices,
        "count": len(t_indices),
        "nodes_path": f"{output_root}/snapshots/{split_name}/nodes",
        "edges_path": f"{output_root}/snapshots/{split_name}/edges",
    }
    mdf = spark.createDataFrame([(json.dumps(manifest, indent=2),)], ["json"])
    mdf.coalesce(1).write.mode("overwrite").text(f"{output_root}/manifests/{split_name}")

    if USE_UC:
        table_name = f"{UC_CATALOG}.{UC_SCHEMA}.tail_manifests"
        manifest_row = spark.createDataFrame(
            [(split_name, manifest["count"], manifest["t_indices"][0], manifest["t_indices"][-1],
              manifest["nodes_path"], manifest["edges_path"])],
            ["split", "num_snapshots", "t_idx_start", "t_idx_end", "nodes_path", "edges_path"],
        )
        if spark.catalog.tableExists(table_name):
            spark.sql(f"DELETE FROM {table_name} WHERE split = '{split_name}'")
            manifest_row.write.format("delta").mode("append").saveAsTable(table_name)
        else:
            manifest_row.write.format("delta").mode("overwrite").saveAsTable(table_name)

    print(f"[{split_name}] manifest: {len(t_indices)} snapshots (t={t_indices[0]} -> t={t_indices[-1]})")
    return manifest


manifests = {
    "train": write_manifest(node_feat_train, "train", FE_OUTPUT_ROOT),
    "val": write_manifest(node_feat_val, "val", FE_OUTPUT_ROOT),
    "test": write_manifest(node_feat_test, "test", FE_OUTPUT_ROOT),
}

# COMMAND ----------

stats = {
    "num_nodes": node_index.count(),
    "num_node_features": len(ALL_NODE_FEATURE_COLS),
    "num_edge_features": len(EDGE_FEATURE_COLS),
    "node_feature_names": ALL_NODE_FEATURE_COLS,
    "edge_feature_names": EDGE_FEATURE_COLS,
    "train_t_range": [manifests["train"]["t_indices"][0], manifests["train"]["t_indices"][-1]],
    "val_t_range": [manifests["val"]["t_indices"][0], manifests["val"]["t_indices"][-1]],
    "test_t_range": [manifests["test"]["t_indices"][0], manifests["test"]["t_indices"][-1]],
    "fe_output_root": FE_OUTPUT_ROOT,
    "rolling_window_size": ROLLING_WINDOW_SIZE,
    "cycle_period": CYCLE_PERIOD,
    "target_scalers": {
        "cpu_rs": {"median": cpu_q2, "iqr": cpu_iqr},
        "memory_rs": {"median": mem_q2, "iqr": mem_iqr}
    },
    "msrtqps_path": MSRTQPS_PATH,
    "dropped_rt_columns": dropped_rt_cols,
    "node_index_fingerprint": NODE_INDEX_FINGERPRINT,
    "notes": {
        "node_key": "All nodes are keyed by msname, aligned with call-graph UM/DM.",
        "rpctype_idx": "Ordinal index for nn.Embedding lookup, not one-hot.",
        "edge_rt": "rt_mean_rs/rt_abs_mean_rs are robust-scaled; rt can be negative by design (DM-side calls) and was not filtered out.",
    },
}

stats_df = spark.createDataFrame([(json.dumps(stats, indent=2),)], ["json"])
stats_df.coalesce(1).write.mode("overwrite").text(f"{FE_OUTPUT_ROOT}/dataset_stats")

if USE_UC:
    spark.createDataFrame([(json.dumps(stats),)], ["stats_json"]) \
        .write.format("delta").mode("overwrite") \
        .saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.tail_dataset_stats")
    print(f"Registered {UC_CATALOG}.{UC_SCHEMA}.tail_dataset_stats as a managed Delta table.")

print(json.dumps(stats, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 10 — Sanity Checks

# COMMAND ----------

for split, manifest in manifests.items():
    t0 = manifest["t_indices"][0]
    nodes = spark.read.option("basePath", f"{FE_OUTPUT_ROOT}/snapshots/{split}/nodes") \
        .parquet(f"{FE_OUTPUT_ROOT}/snapshots/{split}/nodes/t_idx={t0}")
    edges = spark.read.option("basePath", f"{FE_OUTPUT_ROOT}/snapshots/{split}/edges") \
        .parquet(f"{FE_OUTPUT_ROOT}/snapshots/{split}/edges/t_idx={t0}")

    n_nodes = nodes.count()
    n_edges = edges.count()
    null_feats = 0
    for c in ALL_NODE_FEATURE_COLS:
        null_feats += nodes.filter(F.col(c).isNull()).count()

    print(f"\n=== {split.upper()} ===")
    print(f"  Snapshots     : {manifest['count']}  (t={manifest['t_indices'][0]} -> t={manifest['t_indices'][-1]})")
    print(f"  t={t0} nodes   : {n_nodes:,}")
    print(f"  t={t0} edges   : {n_edges:,}")
    print(f"  Null feature values at t={t0}: {null_feats:,}")
    if n_nodes == 0:
        print("  \u26a0\ufe0f  WARNING: zero nodes at first snapshot")
    if n_edges == 0:
        print("  \u26a0\ufe0f  WARNING: zero edges at first snapshot")
    if null_feats > 0:
        print("  \u26a0\ufe0f  WARNING: null feature values found — investigate imputation/join logic")

print("\nFeature engineering stage complete.")