# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "statsmodels",
# ]
# ///
# MAGIC %md
# MAGIC ## Continuous-Time Event-Stream FE Pipeline 
# MAGIC
# MAGIC

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

# MAGIC %pip install statsmodels

# COMMAND ----------

# MAGIC  %restart_python

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import LongType
from pyspark.ml.feature import StringIndexer
import numpy as np
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ### Widgets  and  Configurations

# COMMAND ----------

dbutils.widgets.text("etl_output_root", "s3://research-s20426/etl_output_root", "ETL output S3 root (input)")
dbutils.widgets.text("msrtqps_path", "s3://research-s20426/msrtqps_parquet/parquet_final/", "Corrected MSRTQps parquet folder")
dbutils.widgets.text("fe_snapshot_root", "s3://research-s20426/tail_features/tgnn_dataset", "Existing snapshot-FE output root (to read node_index from)")

dbutils.widgets.dropdown("use_unity_catalog", "false", ["true", "false"], "Store output in Unity Catalog Volume + Delta tables?")
dbutils.widgets.text("uc_catalog", "main", "Unity Catalog: catalog name")
dbutils.widgets.text("uc_schema", "tgnn", "Unity Catalog: schema name")
dbutils.widgets.text("uc_volume", "fe_dataset", "Unity Catalog: volume name (for raw files, if used)")
dbutils.widgets.text("fe_output_root", "s3://research-s20426/tail_features/tgnn_event_dataset", "Event-stream FE output root (used when use_unity_catalog=false)")

dbutils.widgets.text("train_ratio", "0.70", "Train ratio of total t_idx span")
dbutils.widgets.text("val_ratio", "0.15", "Val ratio of total t_idx span (test gets the remainder)")
dbutils.widgets.text("rolling_window_size", "5", "Rolling window size in t_idx steps (was hardcoded to 5; see ACF cell below before changing)")

ETL_OUTPUT_ROOT = dbutils.widgets.get("etl_output_root")
MSRTQPS_PATH = dbutils.widgets.get("msrtqps_path")
FE_SNAPSHOT_ROOT = dbutils.widgets.get("fe_snapshot_root")
USE_UC = dbutils.widgets.get("use_unity_catalog") == "true"
UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
TRAIN_RATIO = float(dbutils.widgets.get("train_ratio"))
VAL_RATIO = float(dbutils.widgets.get("val_ratio"))
ROLLING_WINDOW_SIZE = int(dbutils.widgets.get("rolling_window_size"))
assert ROLLING_WINDOW_SIZE >= 1, "rolling_window_size must be >= 1"

assert 0 < TRAIN_RATIO < 1 and 0 < VAL_RATIO < 1 and TRAIN_RATIO + VAL_RATIO < 1.0, \
    "train_ratio + val_ratio must be < 1.0 (test gets the remainder)"

if USE_UC:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {UC_CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
    FE_OUTPUT_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/tail_features/tgnn_event_dataset"
else:
    FE_OUTPUT_ROOT = dbutils.widgets.get("fe_output_root")

PREVIOUS_APPROACH_ROOTS = {"s3://research-s20426/tgnn_dataset2", "s3://research-s20426/tgnn_event_dataset", "s3://research-s20426/load_balancing_work"}
assert FE_OUTPUT_ROOT.rstrip("/") not in PREVIOUS_APPROACH_ROOTS, "This is the tail feature version. It must not write into the folders of the previous approach."
print(f"fe_output_root = {FE_OUTPUT_ROOT}   (unity_catalog={USE_UC})")
print(f"msrtqps_path   = {MSRTQPS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Load ETL output 

# COMMAND ----------

res = spark.read.parquet(f"{ETL_OUTPUT_ROOT}/resource")
rtq = spark.read.parquet(MSRTQPS_PATH)
cg = spark.read.parquet(f"{ETL_OUTPUT_ROOT}/callgraph_1")

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

# Reuse the node index the snapshot notebook already built (same msname keys),
# so node ids match 1:1 across both FE outputs.
node_index = spark.read.parquet(f"{FE_SNAPSHOT_ROOT}/node_index")
print(f"Loaded node_index: {node_index.count():,} nodes from {FE_SNAPSHOT_ROOT}/node_index")

node_id_check = node_index.agg(F.min("node_id"), F.max("node_id"), F.countDistinct("node_id"), F.count("*")).first()
assert node_id_check[0] == 0 and node_id_check[1] == node_id_check[3] - 1 and node_id_check[2] == node_id_check[3], \
    f"node ids must be 0 to N - 1 without gaps or repeats, got min {node_id_check[0]}, max {node_id_check[1]}, " \
    f"{node_id_check[2]} distinct of {node_id_check[3]}"
print(f"node ids are dense: 0 to {node_id_check[1]}")
import hashlib
fingerprint_rows = node_index.select("node_key", "node_id").orderBy("node_id").toPandas()
NODE_INDEX_FINGERPRINT = hashlib.sha256("\n".join(f"{key}:{node_id}" for key, node_id in zip(fingerprint_rows.node_key, fingerprint_rows.node_id)).encode()).hexdigest()
print("node_index fingerprint:", NODE_INDEX_FINGERPRINT)


# COMMAND ----------

# MAGIC %md
# MAGIC ####  Time boundaries 

# COMMAND ----------

min_t_idx = min(res.agg(F.min("t_idx")).first()[0], rtq.agg(F.min("t_idx")).first()[0], cg.agg(F.min("t_idx")).first()[0])
max_t_idx = max(res.agg(F.max("t_idx")).first()[0], rtq.agg(F.max("t_idx")).first()[0], cg.agg(F.max("t_idx")).first()[0])
span = max_t_idx - min_t_idx

TRAIN_END = min_t_idx + int(round(span * TRAIN_RATIO))
VAL_END = min_t_idx + int(round(span * (TRAIN_RATIO + VAL_RATIO)))
TEST_END = max_t_idx

print(f"t_idx range [{min_t_idx}, {max_t_idx}]  TRAIN_END={TRAIN_END}  VAL_END={VAL_END}  TEST_END={TEST_END}")

TRAIN_COND = F.col("t_idx") < TRAIN_END  # fit-only condition, never a filter before FE

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Build the Node features
# MAGIC

# COMMAND ----------


# Handling Missing values - Meadian imputations

def fit_median_imputer(df, cols, train_cond):
    train_df = df.filter(train_cond)
    stats = {}
    for c in cols:                                             # find median from each column
        med = train_df.approxQuantile(c, [0.5], 0.001)
        stats[c] = med[0] if med else 0.0
    return stats


def apply_median_imputer(df, stats):
    for c, med in stats.items():
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(med)).otherwise(F.col(c)))
    return df
                                                            # apply median each column

def compute_robust_stats(df, col_name):
    q1, q2, q3 = df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
    iqr = q3 - q1
    if iqr < 1e-8:
        iqr = 1.0
    return q2, iqr


def robust_scale_column(df, col_name, q2, iqr, output_col=None):
    output_col = output_col or f"{col_name}_rs"
    return df.withColumn(output_col, (F.col(col_name) - F.lit(q2)) / F.lit(iqr))


def add_temporal_features(df, base_col, prefix, roll_window, lag_window):
    df = df.withColumn(f"{prefix}_rolling_mean", F.avg(base_col).over(roll_window))
    df = df.withColumn(f"{prefix}_rolling_std", F.stddev(base_col).over(roll_window))
    df = df.withColumn(
        f"{prefix}_rolling_std",
        F.when(F.col(f"{prefix}_rolling_std").isNull(), 0.0).otherwise(F.col(f"{prefix}_rolling_std")),
    )
    lag1 = F.lag(base_col, 1).over(lag_window)
    df = df.withColumn(f"{prefix}_delta", F.when(lag1.isNull(), 0.0).otherwise(F.col(base_col) - lag1))
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

rtq_metric_cols = [c for c in rtq.columns if c not in ("timestamp", "msname", "msinstanceid", "t_idx")]
mcr_cols = [c for c in rtq_metric_cols if c.endswith("_MCR")]
rt_cols = [c for c in rtq_metric_cols if c.endswith("_RT")]
all_rtq_cols = mcr_cols + rt_cols

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

res_ms = res.groupBy("msname", "t_idx").agg(
    F.mean("cpu_utilization").alias("cpu_utilization"),
    F.mean("memory_utilization").alias("memory_utilization"),
)
cpu_q2, cpu_iqr = compute_robust_stats(res_ms.filter(TRAIN_COND), "cpu_utilization")
mem_q2, mem_iqr = compute_robust_stats(res_ms.filter(TRAIN_COND), "memory_utilization")
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
# MAGIC ### ACF/PACF diagnostic to selecting rolling window size
# MAGIC
# MAGIC

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
    print("importing error")

# COMMAND ----------

roll_window = Window.partitionBy("msname").orderBy("t_idx").rowsBetween(-(ROLLING_WINDOW_SIZE - 1), 0)
lag_window = Window.partitionBy("msname").orderBy("t_idx")
res_ms = add_temporal_features(res_ms, "cpu_rs", "cpu", roll_window, lag_window)
res_ms = add_temporal_features(res_ms, "memory_rs", "memory", roll_window, lag_window)

CYCLE_PERIOD = span
res_ms = res_ms.withColumn("t_sin", F.sin(2 * np.pi * F.col("t_idx") / CYCLE_PERIOD))
res_ms = res_ms.withColumn("t_cos", F.cos(2 * np.pi * F.col("t_idx") / CYCLE_PERIOD))

RES_FEATURE_COLS = [
    "cpu_rs", "cpu_rolling_mean", "cpu_rolling_std", "cpu_delta",
    "memory_rs", "memory_rolling_mean", "memory_rolling_std", "memory_delta",
]

rtq_ms = rtq.groupBy("msname", "t_idx").agg(*[F.mean(c).alias(c) for c in all_rtq_cols])
for c in all_rtq_cols:
    rtq_ms = rtq_ms.withColumn(c, F.greatest(F.col(c), F.lit(0.0)))
    rtq_ms = rtq_ms.withColumn(f"{c}_log", F.log1p(F.col(c)))
RTQ_LOG_COLS = [f"{c}_log" for c in all_rtq_cols]
rtq_scale_stats = {}
for c in RTQ_LOG_COLS:
    q2, iqr = compute_robust_stats(rtq_ms.filter(TRAIN_COND), c)
    rtq_scale_stats[c] = (q2, iqr)
    rtq_ms = robust_scale_column(rtq_ms, c, q2, iqr, f"{c}_rs")
RTQ_SCALED_COLS = [f"{c}_rs" for c in RTQ_LOG_COLS]

NODE_FEATURE_COLS = RES_FEATURE_COLS + RTQ_SCALED_COLS + ["t_sin", "t_cos"] + TAIL_NODE_FEATURE_COLS

node_feat_timeline = (
    res_ms.alias("r")
    .join(rtq_ms.alias("q"), on=["msname", "t_idx"], how="left")
    .join(node_index, F.col("r.msname") == node_index["node_key"], how="inner")
    .select(
        F.col("node_id"),
        F.col("r.t_idx").alias("t_idx"),
        *[F.col(c) for c in RES_FEATURE_COLS],
        *[F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in RTQ_SCALED_COLS],
        F.col("r.t_sin").alias("t_sin"),
        F.col("r.t_cos").alias("t_cos"),
        *[F.coalesce(F.col(c), F.lit(0.0)).alias(c) for c in TAIL_NODE_FEATURE_COLS],
    )
)
print(f"node_feat_timeline: {node_feat_timeline.count():,} rows, {len(NODE_FEATURE_COLS)} feature dims")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Raw event stream 

# COMMAND ----------

if "rt_abs" not in cg.columns:
    cg = cg.withColumn("rt_abs", F.abs(F.col("rt")))

cg_events = cg.select("UM", "DM", "timestamp", "t_idx", "rpctype", "rt", "rt_abs")

#Fitting rpctype encoder on TRAIN events only
rpctype_indexer = StringIndexer(inputCol="rpctype", outputCol="rpctype_idx", handleInvalid="keep")
rpctype_model = rpctype_indexer.fit(cg_events.filter(TRAIN_COND))
cg_events = rpctype_model.transform(cg_events)

#"Fitting event-level rt/rt_abs scaler on TRAIN events only
cg_events = cg_events.withColumn("rt_abs_log", F.log1p(F.greatest(F.col("rt_abs"), F.lit(0.0))))
EDGE_RAW_COLS = ["rt", "rt_abs_log"]
edge_scale_stats = {}
for c in EDGE_RAW_COLS:
    q2, iqr = compute_robust_stats(cg_events.filter(TRAIN_COND), c)
    edge_scale_stats[c] = (q2, iqr)
    cg_events = robust_scale_column(cg_events, c, q2, iqr, f"{c}_rs")
EDGE_SCALED_COLS = [f"{c}_rs" for c in EDGE_RAW_COLS]
EDGE_FEATURE_COLS = EDGE_SCALED_COLS + ["rpctype_idx"]

src_lookup = node_index.select(F.col("node_key"), F.col("node_id").alias("src_node_id"))
dst_lookup = node_index.select(F.col("node_key"), F.col("node_id").alias("dst_node_id"))

event_stream = (
    cg_events
    .join(src_lookup, cg_events["UM"] == src_lookup["node_key"], "left").drop("node_key")
    .join(dst_lookup, cg_events["DM"] == dst_lookup["node_key"], "left").drop("node_key")
    .filter(F.col("src_node_id").isNotNull() & F.col("dst_node_id").isNotNull())
    .select(
        F.col("src_node_id"),
        F.col("dst_node_id"),
        F.col("timestamp").alias("t"),   # exact event time, ms — 
        F.col("t_idx"),                 
        F.col("rpctype"),
        *[F.col(c) for c in EDGE_FEATURE_COLS],
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Split by time, write chronologically ordered partitions

# COMMAND ----------

def temporal_split(df, train_end, val_end, test_end, time_col="t_idx"):
    train = df.filter(F.col(time_col) < train_end)
    val = df.filter((F.col(time_col) >= train_end) & (F.col(time_col) < val_end))
    test = df.filter((F.col(time_col) >= val_end) & (F.col(time_col) <= test_end))
    return train, val, test


event_train, event_val, event_test = temporal_split(event_stream, TRAIN_END, VAL_END, TEST_END)
node_tl_train, node_tl_val, node_tl_test = temporal_split(node_feat_timeline, TRAIN_END, VAL_END, TEST_END)


def write_events(df, split_name, output_root):
    cols = [c for c in df.columns if c != "t_idx"]
    (
        df.select("t_idx", *cols)
        .sortWithinPartitions("t")
        .write.mode("overwrite").partitionBy("t_idx").option("compression", "snappy")
        .parquet(f"{output_root}/events/{split_name}")
    )
    print(f"[{split_name}] events written to {output_root}/events/{split_name}")


def write_node_timeline(df, split_name, output_root):
    cols = [c for c in df.columns if c != "t_idx"]
    (
        df.select("t_idx", *cols)
        .write.mode("overwrite").partitionBy("t_idx").option("compression", "snappy")
        .parquet(f"{output_root}/node_feat_timeline/{split_name}")
    )
    print(f"[{split_name}] node feature timeline written to {output_root}/node_feat_timeline/{split_name}")


for split_name, edf, ndf in [("train", event_train, node_tl_train), ("val", event_val, node_tl_val), ("test", event_test, node_tl_test)]:
    write_events(edf, split_name, FE_OUTPUT_ROOT)
    write_node_timeline(ndf, split_name, FE_OUTPUT_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Manifests + dataset stats

# COMMAND ----------

stats = {
    "num_nodes": node_index.count(),
    "num_node_features": len(NODE_FEATURE_COLS),
    "num_edge_features": len(EDGE_FEATURE_COLS),
    "node_feature_names": NODE_FEATURE_COLS,
    "edge_feature_names": EDGE_FEATURE_COLS,
    "event_counts": {
        "train": event_train.count(),
        "val": event_val.count(),
        "test": event_test.count(),
    },
    "t_idx_boundaries": {"train_end": TRAIN_END, "val_end": VAL_END, "test_end": TEST_END},
    "rolling_window_size": ROLLING_WINDOW_SIZE,
    "fe_output_root": FE_OUTPUT_ROOT,
    "target_scalers": {
        "cpu_rs": {"median": cpu_q2, "iqr": cpu_iqr},
        "memory_rs": {"median": mem_q2, "iqr": mem_iqr}
    },
    "msrtqps_path": MSRTQPS_PATH,
    "dropped_rt_columns": dropped_rt_cols,
    "node_index_fingerprint": NODE_INDEX_FINGERPRINT,
    "notes": {
        "format": "Continuous-time event stream. Each row = one RPC call at exact "
                   "timestamp `t` (ms). No windowed aggregation of edges.",
        "ordering": "Partitioned by t_idx, sortWithinPartitions('t') — read partitions "
                    "in ascending t_idx order for exact chronological replay.",
        "node_feat_timeline": "Fixed-interval node state (resource/RTQP gauges), looked "
                               "up at-or-before an event's t — not graph structure.",
        "node_index": f"Reused verbatim from {FE_SNAPSHOT_ROOT}/node_index for parity "
                       "with the discrete-time snapshot dataset.",
    },
}

stats_df = spark.createDataFrame([(json.dumps(stats, indent=2),)], ["json"])
stats_df.coalesce(1).write.mode("overwrite").text(f"{FE_OUTPUT_ROOT}/dataset_stats")

if USE_UC:
    # Register queryable artifacts as managed Delta tables under catalog.schema.
    # Event stream + node timeline are also written as Delta (not just Volume
    # Parquet) so they can be Z-ordered for fast (node, time-range) lookups —
    # see the note on Unity Catalog storage options below.
    for split_name, edf in [("train", event_train), ("val", event_val), ("test", event_test)]:
        (
            edf.write.format("delta").mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.tail_event_stream_{split_name}")
        )
        spark.sql(f"OPTIMIZE {UC_CATALOG}.{UC_SCHEMA}.tail_event_stream_{split_name} ZORDER BY (src_node_id, t)")
    spark.createDataFrame([(json.dumps(stats),)], ["stats_json"]) \
        .write.format("delta").mode("overwrite") \
        .saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.tail_event_dataset_stats")
    print(f"Registered {UC_CATALOG}.{UC_SCHEMA}.tail_event_stream_{{train,val,test}} and tail_event_dataset_stats as managed Delta tables.")

print(json.dumps(stats, indent=2))