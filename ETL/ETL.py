# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField,
                                LongType, StringType, DoubleType)
import os, time
import os, time, glob, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# COMMAND ----------

spark = (SparkSession.builder
    .appName("Alibaba-MS-Load-Optimized")
 
    # ── Memory ──
    .config("spark.driver.memory",            "8g")
    .config("spark.executor.memory",          "20g")
    .config("spark.executor.memoryOverhead",  "4g")   # off-heap buffer
    .config("spark.driver.maxResultSize",     "4g")
 
    # ── Cores ──
    .config("spark.executor.cores",           "4")
 
    # ── Shuffle & parallelism ──
    # 200 partitions default is too low for 60 GB
    # Rule: ~128 MB per partition → 60 GB / 128 MB ≈ 480 partitions
    .config("spark.sql.shuffle.partitions",   "480")
    .config("spark.default.parallelism",      "96")   # 2 executors × 4 cores × 12
 
    # ── CSV reading ──
    .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128 MB
    .config("spark.sql.files.openCostInBytes",   "4194304")    # 4 MB
 
    # ── Arrow (fast pandas conversion) ──
    .config("spark.sql.execution.arrow.pyspark.enabled",         "true")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch",      "50000")
 
    # ── Parquet ──
    .config("spark.sql.parquet.compression.codec",   "snappy")
    .config("spark.sql.parquet.mergeSchema",          "false")  # faster reads
    .config("spark.sql.parquet.filterPushdown",       "true")
    .config("spark.sql.parquet.enableVectorizedReader","true")  # columnar scan
 
    # ── Adaptive Query Execution (Spark 3+) ──
    # Automatically coalesces partitions and optimizes joins at runtime
    .config("spark.sql.adaptive.enabled",                        "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled",     "true")
    .config("spark.sql.adaptive.skewJoin.enabled",               "true")
 
    # ── Spill to disk (safety net) ──
    .config("spark.local.dir",  "/tmp/spark-scratch")
    .config("spark.memory.fraction",          "0.7")
    .config("spark.memory.storageFraction",   "0.4")
 
    .getOrCreate())

# COMMAND ----------

spark.sparkContext.setLogLevel("WARN")

# COMMAND ----------

print(f"Spark version: {spark.version}")
print(f"Executors: {spark.sparkContext.defaultParallelism}")

# COMMAND ----------

DATA_ROOT   = "/home/s20426/S20426_Research_Microservices_Dataset/clusterdata/cluster-trace-microservices-v2021/data"
ARQUET_DIR = "data_parquet"
 
# [MODIFIED] added EXTRACT_DIR for MSResource and MSRTQps tar.gz extraction
#            MSCallGraph does not use this — its CSVs are already readable
EXTRACT_DIR = "data_extracted"
os.makedirs(EXTRACT_DIR, exist_ok=True)
# [END MODIFIED]
 
os.makedirs(PARQUET_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Extraction 

# COMMAND ----------

def _extract_one(tb, out_dir):
    """Extract a single tar.gz using the system tar command."""
    csv_name = os.path.basename(tb).replace(".tar.gz", ".csv")
    out_file = os.path.join(out_dir, csv_name)
 
    # Skip if already extracted successfully
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return f"  skip  {csv_name} (already exists)"
 
    result = subprocess.run(
        ["tar", "-xzf", tb, "-C", out_dir])
 
    if result.returncode != 0:
        return f"  ERROR {os.path.basename(tb)}: {result.stderr.strip()}"
 
    size_mb = os.path.getsize(out_file) / 1e6 if os.path.exists(out_file) else 0
    return f"  done  {csv_name}  ({size_mb:.0f} MB)"

# COMMAND ----------



def extract_parallel(table, max_workers=4):
    """
    Extract all tar.gz archives for a table in parallel.
    Output: EXTRACT_DIR/table/TableName_N.csv
    """
    out_dir  = os.path.join(EXTRACT_DIR, table)
    os.makedirs(out_dir, exist_ok=True)
    tarballs = sorted(glob.glob(f"{DATA_ROOT}/{table}/*.tar.gz"))
 
    if not tarballs:
        raise FileNotFoundError(
            f"No tar.gz found in {DATA_ROOT}/{table}/")
 
    print(f"\n  {table}: extracting {len(tarballs)} archives "
          f"with {max_workers} parallel workers ...")
    t0 = time.time()
 
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_extract_one, tb, out_dir): tb
                   for tb in tarballs}
        for future in as_completed(futures):
            print(future.result())
 
    csv_files = glob.glob(os.path.join(out_dir, "*.csv"))
    size_gb   = sum(os.path.getsize(f) for f in csv_files) / 1e9
    print(f"  ✓ {table}: {len(csv_files)} CSVs ready, "
          f"{size_gb:.1f} GB  ({(time.time()-t0)/60:.1f} min)")
 
 
print("\n── Extracting tar.gz (MSResource & MSRTQps only) ──")
extract_parallel("MSResource", max_workers=4)
extract_parallel("MSRTQps",    max_workers=4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MSRTQps 

# COMMAND ----------

def read_csv_fast(path, schema, name):
    t0 = time.time()
    df = (spark.read
          .option("header",        "true")
          .option("mode",          "DROPMALFORMED")  # skip bad rows silently
          .option("nullValue",     "")
          .option("nanValue",      "NaN")
          .option("compression",   "none")   # already extracted from tar.gz
          # Do NOT set inferSchema=true — we provide schema above
          .schema(schema)
          .csv(path))
    # Repartition to match parallelism budget
    df = df.repartition(480)
    print(f"  {name}: schema loaded in {time.time()-t0:.1f}s "
          f"(actual read happens on first action)")
    return df

# COMMAND ----------

from pyspark.sql.types import (StructType, StructField,
                                LongType, StringType, DoubleType,
                                IntegerType)

schema_rtqps_long = StructType([
    StructField("_c0",          IntegerType(), True),  # unnamed index
    StructField("timestamp",    LongType(),    True),
    StructField("msname",       StringType(),  True),
    StructField("msinstanceid", StringType(),  True),
    StructField("metric",       StringType(),  True),  # e.g. "providerRPC_MCR"
    StructField("value",        DoubleType(),  True),
])

# COMMAND ----------

# rtqps = read_csv_fast(
#     "/home/s20426/S20426_Research_Microservices_Dataset/clusterdata/cluster-trace-microservices-v2021/data/MSRTQps",
#     schema_rtqps,
#     "MSRTQps"
# )

# rtqps_csv_files = sorted(
#     glob.glob(os.path.join(EXTRACT_DIR, "MSRTQps")))
# rtqps = read_csv_fast(rtqps_csv_files, schema_rtqps, "MSRTQps")

rtqps_long = (spark.read
    .option("header",    "true")
    .option("mode",      "DROPMALFORMED")
    .option("nullValue", "")
    .schema(schema_rtqps_long)
    .csv("data_extracted/MSRTQps"))
 
# Check all distinct metric names before pivoting
print("\nDistinct metric values in MSRTQps:")
rtqps_long.select("metric").distinct().show(30, truncate=False)
 
# [FIXED] Aggregate to service level first (mean across instances),
#         then pivot metric column → wide format
rtqps = (rtqps_long
    .drop("_c0")
    # aggregate instances → service level
    .groupBy("timestamp", "msname", "metric")
    .agg(F.mean("value").alias("value"))
    # pivot: one column per metric
    .groupBy("timestamp", "msname")
    .pivot("metric")          # creates one col per distinct metric value
    .agg(F.first("value"))    # one value per (timestamp, msname, metric)
    .withColumn("t_idx", (F.col("timestamp") / 30000).cast("int")))
 
print("\nMSRTQps schema after pivot:")
rtqps.printSchema()
rtqps.show(3, truncate=True)

# COMMAND ----------

def write_parquet(df, out_path, name):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
 
    t0 = time.time()
    print(f"\n── Writing {name} → Parquet ──")
 
    sample = df.limit(3).collect()
    if not sample:
        raise ValueError(f"{name}: DataFrame still empty")
    print(f"  Sample: {sample[0].asDict()}")
 
    (df.repartition(480)
       .write
       .mode("overwrite")
       .option("compression", "snappy")
       .parquet(out_path))
 
    elapsed  = time.time() - t0
    pq_files = glob.glob(os.path.join(out_path, "*.parquet"))
    pq_size  = sum(os.path.getsize(f) for f in pq_files) / 1e9
    print(f"  ✓ {len(pq_files)} files, {pq_size:.1f} GB "
          f"({elapsed/60:.1f} min)")
 
# write_parquet(resource, f"{PARQUET_DIR}/resource", "MSResource")
write_parquet(rtqps,    f"{PARQUET_DIR}/rtqps",    "MSRTQps")

# COMMAND ----------

# rtqps = rtqps.withColumn(
#     "t_idx",    (F.col("timestamp") / 30000).cast("int")
# ).withColumn(
#     "t_bucket", ((F.col("timestamp") / 30000) / BUCKET_SIZE)
#                 .cast("int").cast("string")
# )

# rtqps = rtqps.withColumn(
#     "t_idx", (F.col("timestamp") / 30000).cast("int"))

# COMMAND ----------

# def write_parquet(df, out_path, partition_col=None, name=""):
#     t0 = time.time()
#     print(f"\n── Writing {name} → Parquet ──")
#     writer = df.write.mode("overwrite").format("parquet")
#     if partition_col:
#         writer = writer.partitionBy(partition_col)
#     writer.save(out_path)
#     elapsed = time.time() - t0
#     print(f"   Done in {elapsed/60:.1f} min  →  {out_path}")
# def write_parquet(df, out_path, name):
#     """Write DataFrame to flat Parquet (no partitioning)."""
#     if os.path.exists(out_path):
#         shutil.rmtree(out_path)
 
#     t0 = time.time()
#     print(f"\n── Writing {name} → Parquet ──")
 
#     # Spot-check before writing
#     sample = df.limit(3).collect()
#     if not sample:
#         raise ValueError(f"{name}: DataFrame is empty — check CSV paths")
#     print(f"  Sample: {sample[0].asDict()}")
 
#     (df.write
#        .mode("overwrite")
#        .option("compression", "snappy")
#        .parquet(out_path))
 
#     elapsed = time.time() - t0
#     print(f"  Done in {elapsed/60:.1f} min  →  {out_path}")
 
# # os.makedirs("./data", exist_ok=True)

# COMMAND ----------

# write_parquet(rtqps,
#               f"{PARQUET_DIR}/rtqps",
#               "MSRTQps")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resource Graph

# COMMAND ----------

schema_resource = StructType([
    StructField("_c0",                    IntegerType(), True),  # unnamed index
    StructField("msname",                 StringType(),  True),
    StructField("msinstanceid",           StringType(),  True),
    StructField("nodeid",                 StringType(),  True),
    StructField("instance_cpu_usage",     DoubleType(),  True),  # actual name
    StructField("instance_memory_usage",  DoubleType(),  True),  # actual name
    StructField("timestamp",              LongType(),    True),  # moved to end
])

# COMMAND ----------



# # resource  = read_csv_fast("/home/s20426/S20426_Research_Microservices_Dataset/clusterdata/cluster-trace-microservices-v2021/data/MSResource",
# #                            schema_resource,  "MSResource")


# resource_csv_files = sorted(
#     glob.glob(os.path.join(EXTRACT_DIR, "MSResource", "*.csv")))
# resource = read_csv_fast(resource_csv_files, schema_resource, "MSResource")

resource_files="data_extracted/MSResource"
resource_raw = (spark.read
    .option("header",    "true")
    .option("mode",      "DROPMALFORMED")
    .option("nullValue", "")
    .schema(schema_resource)
    .csv(resource_files))

# COMMAND ----------

# resource = resource.withColumn(
#     "t_idx",    (F.col("timestamp") / 30000).cast("int")
# ).withColumn(
#     "t_bucket", ((F.col("timestamp") / 30000) / BUCKET_SIZE)
#                 .cast("int").cast("string")
# )
# resource = resource.withColumn(
#     "t_idx", (F.col("timestamp") / 30000).cast("int"))
resource = (resource_raw
    .drop("_c0")
    .withColumnRenamed("instance_cpu_usage",    "cpu_utilization")
    .withColumnRenamed("instance_memory_usage",  "memory_utilization")
    .withColumn("t_idx", (F.col("timestamp") / 30000).cast("int")))
 
print("\nMSResource schema after fix:")
resource.printSchema()
resource.show(3, truncate=True)

# COMMAND ----------

write_parquet(resource,
              f"{PARQUET_DIR}/resource",
              "MSResource")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call Graph

# COMMAND ----------

import tarfile, glob, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F
from pyspark.sql.types import *

DATA_ROOT   = "/home/s20426/S20426_Research_Microservices_Dataset/clusterdata/cluster-trace-microservices-v2021/data"
EXTRACT_DIR = "/home/s20426/S20426_Research_Microservices_Dataset/data_extracted"
PARQUET_DIR = "data_parquet"

# ── Corrected schema (9 actual columns) ──────────────────────────
# header: ,traceid,timestamp,rpcid,um,rpctype,dm,interface,rt
schema_callgraph = StructType([
    StructField("_c0",       IntegerType(), True),  # unnamed index
    StructField("traceid",   StringType(),  True),  # trace ID
    StructField("timestamp", LongType(),    True),
    StructField("rpcid",     StringType(),  True),  # call chain position
    StructField("um",        StringType(),  True),  # upstream (callee)
    StructField("rpctype",   StringType(),  True),  # mc/rpc/http/mq
    StructField("dm",        StringType(),  True),  # downstream (caller)
    StructField("interface", StringType(),  True),  # API endpoint
    StructField("rt",        DoubleType(),  True),  # response time
])

# COMMAND ----------



# ── Step 1: Extract tar.gz in parallel ───────────────────────────
def extract_one(tb, out_dir):
    csv_name = os.path.basename(tb).replace(".tar.gz", ".csv")
    out_file = os.path.join(out_dir, csv_name)
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return f"  skip  {csv_name}"
    result = subprocess.run(["tar", "-xzf", tb, "-C", out_dir],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return f"  ERROR {csv_name}: {result.stderr.strip()}"
    size_mb = os.path.getsize(out_file)/1e6 if os.path.exists(out_file) else 0
    return f"  done  {csv_name} ({size_mb:.0f} MB)"

out_dir  = os.path.join(EXTRACT_DIR, "MSCallGraph")
os.makedirs(out_dir, exist_ok=True)
tarballs = sorted(glob.glob(f"{DATA_ROOT}/MSCallGraph/*.tar.gz"))
print(f"Extracting {len(tarballs)} MSCallGraph archives ...")

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(extract_one, tb, out_dir): tb
               for tb in tarballs}
    for f in as_completed(futures):
        print(f.result())

csv_files = sorted(glob.glob(f"{out_dir}/*.csv"))
size_gb   = sum(os.path.getsize(f) for f in csv_files) / 1e9
print(f"\n✓ {len(csv_files)} CSVs, {size_gb:.1f} GB")

# ── Step 2: Read + write Parquet ─────────────────────────────────
shutil.rmtree(f"{PARQUET_DIR}/callgraph", ignore_errors=True)

df = (spark.read
      .option("header",    "true")
      .option("mode",      "DROPMALFORMED")
      .option("nullValue", "")
      .schema(schema_callgraph)
      .csv(csv_files)
      .drop("_c0")
      # Rename to standard names (uppercase DM/UM as before)
      .withColumnRenamed("dm", "DM")
      .withColumnRenamed("um", "UM")
      .withColumn("t_idx", (F.col("timestamp") / 30000).cast("int"))
      # Deduplicate on DM+UM+timestamp (keep all calls, not just unique pairs)
      .dropDuplicates(["DM", "UM", "timestamp"]))

# Spot check
print("\nSample rows:")
df.show(3, truncate=True)
print("Columns:", df.columns)

sample = df.limit(3).collect()
if not sample:
    raise ValueError("Empty DataFrame — check schema vs actual headers")

(df.repartition(16)
   .write
   .mode("overwrite")
   .option("compression", "snappy")
   .parquet(f"{PARQUET_DIR}/callgraph"))

print(f"\n✓ Callgraph written with {len(df.columns)} columns:")
print(f"  {df.columns}")

# ── Verify ────────────────────────────────────────────────────────
result = spark.read.parquet(f"{PARQUET_DIR}/callgraph")
print(f"\nFinal columns : {result.columns}")
print(f"Row count     : {result.count():,}")
result.show(3, truncate=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reads parqet data

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

PARQUET_DIR = "data_parquet"

def load_resource(t_start=None, t_end=None):
    df = spark.read.parquet(f"{PARQUET_DIR}/resource")
    if t_start is not None: df = df.filter(F.col("t_idx") >= t_start)
    if t_end   is not None: df = df.filter(F.col("t_idx") <  t_end)
    return df

def load_rtqps(t_start=None, t_end=None):
    df = spark.read.parquet(f"{PARQUET_DIR}/rtqps")
    if t_start is not None: df = df.filter(F.col("t_idx") >= t_start)
    if t_end   is not None: df = df.filter(F.col("t_idx") <  t_end)
    return df

def load_callgraph():
    return spark.read.parquet(f"{PARQUET_DIR}/callgraph")

# Check row counts + schemas
for label, fn in [
    ("MSResource  (t 0–100)", lambda: load_resource(0, 100)),
    ("MSRTQps     (t 0–100)", lambda: load_rtqps(0, 100)),
    ("MSCallGraph (full)",     load_callgraph),
]:
    n = fn().count()
    print(f"  {'✓' if n > 0 else '✗'} {label}: {n:,} rows")

# Check column names — critical before EDA
print("\nMSResource columns:  ", load_resource(0,1).columns)
print("MSRTQps columns:     ", load_rtqps(0,1).columns)
print("MSCallGraph columns: ", load_callgraph().columns)

# COMMAND ----------



# COMMAND ----------

