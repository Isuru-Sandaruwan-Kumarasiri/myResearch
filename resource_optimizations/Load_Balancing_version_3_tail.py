# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Load Balancing and Scale-Out for Overloaded Microservice Connections
# MAGIC
# MAGIC This notebook turns call rate and CPU data from the Alibaba microservices trace v2021 into a load balancing experiment.
# MAGIC A **connection** is a caller to callee edge. A connection is overloaded when its calls land on callee containers
# MAGIC that run above the safe CPU utilisation.
# MAGIC
# MAGIC | Section | Content |
# MAGIC |---|---|
# MAGIC | 1 | Settings and chart helper |
# MAGIC | 2 | Load balancing environment: callees, connections, containers, CPU and call rates |
# MAGIC | 3 | Capacity model: safe calls per second for each container |
# MAGIC | 4 | Load balancing algorithms: Round Robin, Least Loaded, Weighted Routing, Linear Programming |
# MAGIC | 5 | Scale-out: containers added when existing capacity is not enough |
# MAGIC | 6 | Simulation and validation with actual future demand (oracle) |
# MAGIC | 7 | Results |
# MAGIC | 8 | GNN driven allocation: forecasts, risky connections, intelligent balancing and comparison |
# MAGIC | 9 | Conclusion |
# MAGIC | 10 | Methods and references |
# MAGIC
# MAGIC **Simulation principle.** Routing in the trace cannot be changed, so each policy is replayed on the test minutes.
# MAGIC A policy decides how the callee's incoming calls are split across its containers. The CPU of each container is then
# MAGIC computed with the capacity model: `CPU = container baseline + CPU per call x calls routed to the container`.
# MAGIC
# MAGIC **Scope.** This is a trace driven, open loop simulation, not a production test. Routing decisions do not feed back into
# MAGIC the trace. Connection overload is inferred from the callers' sampled share of calls and the simulated CPU of the callee,
# MAGIC not measured per connection. Memory is in utilisation units (share of a container's memory limit), not GB, because the
# MAGIC trace has no memory sizes.

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

dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from pyspark.sql import functions as F
from scipy import sparse, stats
from scipy.optimize import linprog

warnings.filterwarnings("ignore", category=FutureWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings and chart helper
# MAGIC
# MAGIC The test window is the test window of the forecasting notebooks, so every policy is judged on the same unseen minutes as the GNN models.
# MAGIC When no forecast files are found, it is the last `TEST_MINUTES` minutes of the trace. Everything that is estimated here (connection shares,
# MAGIC CPU per call, container baselines) uses only minutes before the test window.
# MAGIC `DEMAND_MULTIPLIERS` replays the same demand scaled up, to stress the containers and trigger scale-out.

# COMMAND ----------

RESOURCE_PATH = "s3://research-s20426/etl_output_root/resource"
MSRTQPS_PATH = "s3://research-s20426/msrtqps_parquet/parquet_final/"
CALLGRAPH_PATH = "s3://research-s20426/etl_output_root/callgraph_1"
WORK_PATH = "s3://research-s20426/tail_features/load_balancing_work"
OUTPUT_DIR = "./tail_features_load_balancing_outputs"
FE_SNAPSHOT_ROOT = "s3://research-s20426/tail_features/tgnn_dataset"
FE_EVENT_ROOT = "s3://research-s20426/tail_features/tgnn_event_dataset"
CPU_FORECAST_PATHS = {"snapshot": f"{FE_SNAPSHOT_ROOT}/cpu_forecasts", "event stream": f"{FE_EVENT_ROOT}/cpu_event_forecasts"}
MEMORY_FORECAST_PATHS = {"snapshot": f"{FE_SNAPSHOT_ROOT}/memory_forecasts", "event stream": f"{FE_EVENT_ROOT}/memory_event_forecasts"}
EDGE_PREDICTION_PATH = f"{FE_SNAPSHOT_ROOT}/edge_overload_predictions"
PREVIOUS_APPROACH_ROOTS = {"s3://research-s20426/tgnn_dataset2", "s3://research-s20426/tgnn_event_dataset", "s3://research-s20426/load_balancing_work"}
assert WORK_PATH.rstrip("/") not in PREVIOUS_APPROACH_ROOTS, "This is the tail feature version. It must not write into the folders of the previous approach."
EDGE_CLASSIFIER_MODEL = "best"       # "best" picks the classifier with the highest validation score, or name one: TGCN, TGN, TGAT
FORECAST_MARGIN_LEVEL = 0.90         # planned CPU covers this share of validation forecast errors (split conformal)
MEMORY_LIMIT = 0.90                  # memory utilisation above this is reported as memory risk

FIRST_MINUTE = 0
LAST_MINUTE = 719
TEST_MINUTES = 144                   # used only when no forecasts are found; otherwise the forecast test window is used

CALLEE_COUNT = 20                    # busiest callee microservices included in the experiment
MIN_CONTAINERS = 2
MAX_CONTAINERS = 500
MIN_TRAFFIC_CPU_SPEARMAN = 0.3       # callees whose CPU does not follow traffic cannot be balanced by routing

TARGET_UTILISATION = 0.70            # safe CPU utilisation of one container
CAPACITY_QUANTILE = 0.90             # safety margin covers this share of the capacity model residuals
CAPACITY_FIT_ROWS = 3000

LEAST_LOADED_CHUNKS_PER_CONTAINER = 10   # calls arrive in this many chunks per container in each minute
LP_STABILITY_WEIGHT = 0.01           # cost of moving traffic between containers from one minute to the next
SCALE_IN_DELAY_MINUTES = 5           # added containers are kept until demand stays low for this many minutes
MAX_EXTRA_CONTAINER_FACTOR = 2.0

DEMAND_MULTIPLIERS = [1.0, 1.5]
DEMAND_SOURCES = ["oracle", "persistence"]
INTELLIGENT_BALANCERS = ["Weighted Routing", "Linear Programming"]
GATES = ["classifier", "capacity check", "every minute"]

INCOMING_MCR_COLUMNS = ["providerRPC_MCR", "HTTP_MCR", "consumerMQ_MCR"]
MISSING_SERVICE_NAMES = ["(?)", "NAN", "nan", "NaN", "", "UNKNOWN", "UNAVAILABLE"]
BALANCERS = ["Round Robin", "Least Loaded", "Weighted Routing", "Linear Programming"]

TRAINING_LAST_MINUTE = LAST_MINUTE - TEST_MINUTES
TEST_FIRST_MINUTE = TRAINING_LAST_MINUTE + 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COMMAND ----------

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
SERIES_YELLOW = "#eda100"
SERIES_VIOLET = "#6250d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE_COLOR = "#fcfcfb"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("sequential_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
BALANCER_COLORS = {
    "Observed": TEXT_MUTED,
    "Round Robin": SERIES_ORANGE,
    "Least Loaded": SERIES_YELLOW,
    "Weighted Routing": SERIES_AQUA,
    "Linear Programming": SERIES_BLUE
}

plt.rcParams.update({
    "figure.dpi": 110,
    "figure.facecolor": SURFACE_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "savefig.facecolor": SURFACE_COLOR,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": AXIS_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelcolor": TEXT_SECONDARY,
    "ytick.labelcolor": TEXT_SECONDARY,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "patch.edgecolor": SURFACE_COLOR,
    "patch.linewidth": 0.8
})

THOUSANDS = FuncFormatter(lambda value, position: f"{value:,.0f}" if abs(value) >= 100 or value == 0 else f"{value:.3g}")


def plain_number(value, position=None):
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def show_table(frame, name):
    """Save a result table as CSV and show it in the notebook output."""
    frame.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
    if frame.empty:
        print("no rows")
        return
    try:
        display(frame)
    except NameError:
        print(frame.to_string(index=False))


def tell(*sentences):
    """Print a short plain language interpretation, one sentence per line."""
    print("\nInterpretation")
    for sentence in sentences:
        if sentence:
            print("  " + sentence)


def calls(value):
    """Readable call count: whole numbers when large, one decimal when small."""
    return f"{value:,.0f}" if abs(value) >= 10 else f"{value:.1f}"


def callee_list(labels):
    labels = list(labels)
    if not labels:
        return "none"
    return ", ".join(labels[:-1]) + (" and " if len(labels) > 1 else "") + labels[-1]


def format_axes(ax, count_axis=None):
    if count_axis == "y":
        ax.yaxis.set_major_formatter(THOUSANDS)
    if count_axis == "x":
        ax.xaxis.set_major_formatter(THOUSANDS)
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_major_formatter(FuncFormatter(plain_number))
            axis.set_minor_formatter(NullFormatter())


def save_and_show(fig, file_name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def finish_chart(fig, ax, title, x_label, y_label, file_name, count_axis=None):
    """Give every chart a title, both axis labels, readable tick numbers and a saved copy."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    format_axes(ax, count_axis)
    save_and_show(fig, file_name)


def finish_figure(fig, title, x_label, y_label, file_name):
    """finish_chart for a figure with several panels. Pass None for a label that each panel sets itself."""
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    if x_label:
        fig.supxlabel(x_label, fontsize=11, color=TEXT_SECONDARY)
    if y_label:
        fig.supylabel(y_label, fontsize=11, color=TEXT_SECONDARY)
    save_and_show(fig, file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Forecast and classifier inputs
# MAGIC
# MAGIC * **CPU forecasts** from the snapshot and event stream notebooks: TGCN, TGN, TGAT, DCRNN and GraphMixer, plus the persistence and rolling mean
# MAGIC   baselines on the same rows. Each row is one service and one target minute, with utilisation as a fraction.
# MAGIC * **Memory forecasts** from the two memory notebooks, in the same layout.
# MAGIC * **Edge predictions** from the edge overload classifier: the probability that a connection carries unusually high traffic in that minute,
# MAGIC   with the decision threshold chosen on validation.
# MAGIC
# MAGIC A missing file is skipped with a message, so the notebook still runs with the baselines only.

# COMMAND ----------

def read_optional_parquet(path):
    try:
        return spark.read.parquet(path).toPandas()
    except Exception as error:
        print(f"skipped, not readable: {path} ({str(error).splitlines()[0][:90]})")
        return None


def read_forecasts(paths):
    tables = []
    for pipeline_label, path in paths.items():
        table = read_optional_parquet(path)
        if table is not None:
            tables.append(table.assign(pipeline_label=pipeline_label))
            print(f"loaded {len(table):,} rows from {path}")
    return pd.concat(tables, ignore_index=True) if tables else None


CPU_FORECASTS = read_forecasts(CPU_FORECAST_PATHS)
MEMORY_FORECASTS = read_forecasts(MEMORY_FORECAST_PATHS)
EDGE_PREDICTIONS = read_optional_parquet(EDGE_PREDICTION_PATH)

FORECAST_HORIZON = 1
if CPU_FORECASTS is not None:
    horizons = sorted(CPU_FORECASTS.horizon.unique())
    assert len(horizons) == 1, f"all CPU forecasts must use one horizon, found {horizons}"
    FORECAST_HORIZON = int(horizons[0])
    forecast_test_minutes = CPU_FORECASTS.loc[CPU_FORECASTS.split == "test", "t_idx"]
    TEST_FIRST_MINUTE = max(int(forecast_test_minutes.min()), FIRST_MINUTE + 1)
    LAST_MINUTE = min(LAST_MINUTE, int(forecast_test_minutes.max()))
    TRAINING_LAST_MINUTE = TEST_FIRST_MINUTE - 1
    TEST_MINUTES = LAST_MINUTE - TEST_FIRST_MINUTE + 1
    calibration_rows = CPU_FORECASTS.loc[CPU_FORECASTS.split == "cal", "t_idx"]
    if len(calibration_rows):
        CALIBRATION_FIRST_MINUTE = int(calibration_rows.min())
CALIBRATION_FIRST_MINUTE = globals().get("CALIBRATION_FIRST_MINUTE", max(FIRST_MINUTE + 1, TEST_FIRST_MINUTE - 45))
print(f"calibration window: minutes {CALIBRATION_FIRST_MINUTE} to {TEST_FIRST_MINUTE - 1}. The capacity model is fitted on earlier minutes only.")
print(f"test window: minutes {TEST_FIRST_MINUTE} to {LAST_MINUTE} ({TEST_MINUTES} minutes), forecast horizon {FORECAST_HORIZON} minute(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load balancing environment
# MAGIC
# MAGIC Four building blocks are collected for each callee microservice.
# MAGIC
# MAGIC * **Containers.** Every container (`msinstanceid`) of the callee and the minutes in which it reports traffic.
# MAGIC * **Call rate per container.** Incoming calls per second = providerRPC + HTTP + consumerMQ calls, averaged per container and minute.
# MAGIC   A paradigm with no value has no calls, and a container with no values at all has 0 calls, as in the FE notebooks.
# MAGIC * **Observed CPU per container.** MSResource CPU utilisation, averaged per container and minute.
# MAGIC * **Connections.** Caller to callee edges from MSCallGraph. The call graph is a 0.5% sample, so it gives each connection's
# MAGIC   **share** of the callee's calls, not absolute rates. The connection call rate is `share x callee call rate` (a ratio estimator).
# MAGIC
# MAGIC Callees are the busiest microservices with between `MIN_CONTAINERS` and `MAX_CONTAINERS` containers. A callee with one container has nothing to balance.

# COMMAND ----------

def add_minute_index(table):
    """Turn NaN into null in numeric columns and add the minute index."""
    for column_name, data_type in table.dtypes:
        if data_type in ("double", "float"):
            table = table.withColumn(column_name, F.when(~F.isnan(F.col(column_name)), F.col(column_name)))
    table = table.withColumn("t_idx", F.floor(F.col("timestamp") / 60000).cast("int"))
    return table.filter(F.col("t_idx").between(FIRST_MINUTE, LAST_MINUTE))


def null_safe_sum(columns):
    """Sum that treats null as zero but stays null when every column is null."""
    total = sum(F.coalesce(F.col(column), F.lit(0.0)) for column in columns)
    any_present = None
    for column in columns:
        present = F.col(column).isNotNull()
        any_present = present if any_present is None else (any_present | present)
    return F.when(any_present, total)


resource = add_minute_index(spark.read.parquet(RESOURCE_PATH))
msrtqps = add_minute_index(spark.read.parquet(MSRTQPS_PATH))
callgraph = add_minute_index(spark.read.parquet(CALLGRAPH_PATH))
INCOMING_MCR_COLUMNS = [column for column in INCOMING_MCR_COLUMNS if column in msrtqps.columns]

container_traffic = (
    msrtqps.groupBy("msname", "msinstanceid", "t_idx")
    .agg(*[F.mean(column).alias(column) for column in INCOMING_MCR_COLUMNS])
    .withColumn("incoming_mcr", F.coalesce(null_safe_sum(INCOMING_MCR_COLUMNS), F.lit(0.0)))
    .select("msname", "msinstanceid", "t_idx", "incoming_mcr")
)
container_resource = resource.groupBy("msname", "msinstanceid", "t_idx").agg(
    F.mean("cpu_utilization").alias("cpu_utilization"),
    F.mean("memory_utilization").alias("memory_utilization"),
    F.first("nodeid", ignorenulls=True).alias("nodeid")
)

# COMMAND ----------

callee_candidates = (
    container_traffic.filter(F.col("t_idx") <= TRAINING_LAST_MINUTE)
    .groupBy("msname")
    .agg(F.countDistinct("msinstanceid").alias("containers"), F.sum("incoming_mcr").alias("training_calls"))
    .filter(F.col("containers").between(MIN_CONTAINERS, MAX_CONTAINERS))
)
called_services = (
    callgraph.filter(~F.col("DM").isin(MISSING_SERVICE_NAMES) & F.col("DM").isNotNull())
    .select(F.col("DM").alias("msname")).distinct()
)
selected_callees = (
    callee_candidates.join(called_services, "msname")
    .orderBy(F.desc("training_calls")).limit(CALLEE_COUNT).toPandas()
)
callee_names = list(selected_callees.msname)
callee_label = {name: f"Callee {rank}" for rank, name in enumerate(callee_names, start=1)}
selected_callees.insert(0, "label", [callee_label[name] for name in callee_names])
print(f"callees selected: {len(callee_names)}")

container_minutes = (
    container_traffic.filter(F.col("msname").isin(callee_names))
    .join(container_resource, ["msname", "msinstanceid", "t_idx"], "left")
)
container_minutes.write.mode("overwrite").parquet(f"{WORK_PATH}/container_minutes")
container_minutes = spark.read.parquet(f"{WORK_PATH}/container_minutes").toPandas()
container_minutes["period"] = np.where(container_minutes.t_idx <= TRAINING_LAST_MINUTE, "training", "test")
print(f"container minutes: {len(container_minutes):,}, containers: {container_minutes.msinstanceid.nunique():,}")

# COMMAND ----------

unique_calls = (
    callgraph.filter(F.col("DM").isin(callee_names) & (F.col("t_idx") <= TRAINING_LAST_MINUTE))
    .filter(F.col("UM").isNotNull() & ~F.col("UM").isin(MISSING_SERVICE_NAMES))
    .dropDuplicates(["traceid", "rpcid"])
)
connections = unique_calls.groupBy("UM", "DM").agg(F.count("*").alias("sampled_calls")).toPandas()
connections = connections.rename(columns={"UM": "caller", "DM": "callee"})
connections["share"] = connections.sampled_calls / connections.groupby("callee").sampled_calls.transform("sum")
connections = connections.sort_values(["callee", "share"], ascending=[True, False]).reset_index(drop=True)
connections["callee_label"] = connections.callee.map(callee_label)

caller_number = {name: number for number, name in enumerate(connections.caller.unique(), start=1)}
connections["connection"] = [f"Caller {caller_number[caller]} → {callee_label[callee]}" for caller, callee in zip(connections.caller, connections.callee)]
print(f"connections: {len(connections):,}")

training_rows = container_minutes[container_minutes.period == "training"]
environment = pd.DataFrame({
    "label": [callee_label[name] for name in callee_names],
    "msname": callee_names,
    "containers": [training_rows.loc[training_rows.msname == name, "msinstanceid"].nunique() for name in callee_names],
    "incoming_connections": [int((connections.callee == name).sum()) for name in callee_names],
    "mean_calls_per_second": [training_rows[training_rows.msname == name].groupby("t_idx").incoming_mcr.sum().mean() for name in callee_names],
    "median_cpu_percent": [100 * training_rows.loc[training_rows.msname == name, "cpu_utilization"].median() for name in callee_names]
})
show_table(environment.round(2), "environment")

busiest = environment.loc[environment.mean_calls_per_second.idxmax()]
tell(
    f"We picked {len(environment)} busy callee microservices.",
    f"Together they run {environment.containers.sum():,} containers and receive calls through {len(connections):,} connections.",
    f"The busiest one, {busiest.label}, receives about {busiest.mean_calls_per_second:,.0f} calls per second.",
    f"A typical callee has {environment.containers.median():.0f} containers and {environment.incoming_connections.median():.0f} incoming connections.",
    f"Its containers usually run at about {environment.median_cpu_percent.median():.0f}% CPU, so there is room to move traffic around."
)

# COMMAND ----------

TOP_CONNECTIONS = 5
fig, ax = plt.subplots(figsize=(12, 0.42 * len(callee_names) + 1.8))
labels = [callee_label[name] for name in callee_names][::-1]
left = np.zeros(len(labels))
for position in range(TOP_CONNECTIONS):
    shares = []
    for name in callee_names[::-1]:
        rows = connections[connections.callee == name]
        shares.append(100 * rows.share.iloc[position] if len(rows) > position else 0.0)
    ax.barh(labels, shares, left=left, color=SEQUENTIAL_BLUE(0.95 - 0.17 * position), height=0.7,
            label=f"Connection rank {position + 1}")
    left += np.array(shares)
ax.barh(labels, 100 - left, left=left, color=GRID_COLOR, height=0.7, label="Other connections")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
finish_chart(fig, ax, "Incoming Connection Shares per Callee", "Share of callee calls (%)", "Callee microservice",
             "connection_shares.png")

top_share = connections.groupby("callee").share.max()
dominated = int((top_share > 0.5).sum())
tell(
    f"In a typical callee, the biggest caller sends {100 * top_share.median():.0f}% of all calls.",
    f"In {dominated} of {len(top_share)} callees, one caller sends more than half of the calls.",
    "So when that one caller sends a burst, the whole callee feels it." if dominated
    else "Calls come from several callers, so one caller alone rarely decides the load.",
    "The biggest connections are the first ones to watch for overload."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Capacity model
# MAGIC
# MAGIC The capacity model is fitted on minutes before the calibration window, so the forecast margins in section 8 are calibrated
# MAGIC on minutes the capacity model has not seen. For each callee, CPU utilisation of a container is modelled as
# MAGIC
# MAGIC `CPU = baseline of the container + CPU per call x incoming calls per second`
# MAGIC
# MAGIC * **CPU per call** is the Theil Sen slope of CPU on call rate over training container minutes. Theil Sen ignores outliers.
# MAGIC * **Baseline** is the median over training minutes of `CPU - CPU per call x calls` for that container. A container first seen
# MAGIC   in the test minutes gets the callee's median baseline.
# MAGIC * **Safety margin** is the `CAPACITY_QUANTILE` quantile of the training residuals, so the model covers 90% of observed CPU.
# MAGIC * **Safe capacity** of a container = `(TARGET_UTILISATION - baseline - margin) / CPU per call` calls per second.
# MAGIC
# MAGIC Callees whose CPU does not follow traffic (within callee Spearman below `MIN_TRAFFIC_CPU_SPEARMAN`) or whose slope is not
# MAGIC positive are left out, because routing calls does not change their CPU.

# COMMAND ----------

def fit_capacity_model(rows):
    rows = rows[["msinstanceid", "incoming_mcr", "cpu_utilization"]].dropna()
    if len(rows) < 30 or rows.incoming_mcr.nunique() < 3:
        return None
    sample = rows.sample(min(len(rows), CAPACITY_FIT_ROWS), random_state=7)
    spearman = stats.spearmanr(sample.incoming_mcr, sample.cpu_utilization).correlation
    cpu_per_call = stats.theilslopes(sample.cpu_utilization, sample.incoming_mcr)[0]
    if not np.isfinite(spearman) or spearman < MIN_TRAFFIC_CPU_SPEARMAN or cpu_per_call <= 0:
        return {"spearman": spearman, "cpu_per_call": cpu_per_call, "usable": False}
    residual_base = rows.cpu_utilization - cpu_per_call * rows.incoming_mcr
    baselines = residual_base.groupby(rows.msinstanceid).median()
    residuals = residual_base - rows.msinstanceid.map(baselines)
    margin = max(0.0, float(np.quantile(residuals, CAPACITY_QUANTILE)))
    median_baseline = float(baselines.median())
    return {
        "spearman": spearman,
        "cpu_per_call": float(cpu_per_call),
        "baselines": baselines.to_dict(),
        "median_baseline": median_baseline,
        "margin": margin,
        "new_container_capacity": max(0.0, (TARGET_UTILISATION - median_baseline - margin) / cpu_per_call),
        "usable": True
    }


capacity_models = {}
model_rows = []
for name in callee_names:
    model = fit_capacity_model(training_rows[(training_rows.msname == name) & (training_rows.t_idx < CALIBRATION_FIRST_MINUTE)])
    if model is None:
        model_rows.append({"label": callee_label[name], "usable": False})
        continue
    capacity_models[name] = model
    model_rows.append({
        "label": callee_label[name],
        "spearman": model["spearman"],
        "cpu_percent_per_100_calls": 100 * 100 * model["cpu_per_call"],
        "median_baseline_percent": 100 * model.get("median_baseline", np.nan),
        "margin_percent": 100 * model.get("margin", np.nan),
        "new_container_safe_calls_per_second": model.get("new_container_capacity", np.nan),
        "usable": model["usable"]
    })
capacity_table = pd.DataFrame(model_rows)
show_table(capacity_table.round(3), "capacity_models")

balanced_callees = [name for name in callee_names if capacity_models.get(name, {}).get("usable")]
excluded = [callee_label[name] for name in callee_names if name not in balanced_callees]
usable_table = capacity_table[capacity_table.usable == True]
tell(
    f"CPU follows the number of calls in {len(balanced_callees)} of {len(callee_names)} callees. These are kept.",
    f"Left out: {callee_list(excluded)}. Their CPU hardly moves when calls change, so routing calls cannot help them."
    if excluded else "No callee had to be left out.",
    f"A new container can safely take between {usable_table.new_container_safe_calls_per_second.min():,.0f} and "
    f"{usable_table.new_container_safe_calls_per_second.max():,.0f} calls per second, depending on the callee.",
    "Heavy services need many containers. Light services can serve a lot of calls with one container."
)

# COMMAND ----------

def container_baseline(model, container_id):
    return model["baselines"].get(container_id, model["median_baseline"])


def safe_capacity(model, baseline):
    return max(0.0, (TARGET_UTILISATION - baseline - model["margin"]) / model["cpu_per_call"])


validation_rows = []
safe_capacities = []
test_rows = container_minutes[(container_minutes.period == "test") & container_minutes.msname.isin(balanced_callees)]
for name in balanced_callees:
    model = capacity_models[name]
    rows = test_rows[test_rows.msname == name].dropna(subset=["cpu_utilization"])
    if rows.empty:
        continue
    baselines = rows.msinstanceid.map(lambda container_id: container_baseline(model, container_id))
    expected_cpu = baselines + model["cpu_per_call"] * rows.incoming_mcr
    validation_rows.append({
        "label": callee_label[name],
        "test_container_minutes": len(rows),
        "mean_absolute_error_percent": 100 * np.mean(np.abs(rows.cpu_utilization - expected_cpu)),
        "coverage_with_margin": np.mean(rows.cpu_utilization <= expected_cpu + model["margin"])
    })
    safe_capacities.extend(safe_capacity(model, value) for value in model["baselines"].values())
capacity_validation = pd.DataFrame(validation_rows)
show_table(capacity_validation.round(3), "capacity_validation")

worst_error = capacity_validation.loc[capacity_validation.mean_absolute_error_percent.idxmax()]
tell(
    "We tested the capacity model on the last minutes of the trace, which it never saw.",
    f"In a typical callee its CPU guess was off by about {capacity_validation.mean_absolute_error_percent.median():.1f} CPU points.",
    f"The biggest miss is {worst_error.label}, at {worst_error.mean_absolute_error_percent:.1f} points.",
    "The next chart shows how often the model with its safety margin stays above the real CPU."
)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 4.8))
ordered = capacity_validation.sort_values("coverage_with_margin")
bar_colors = [SERIES_BLUE if value >= CAPACITY_QUANTILE - 0.05 else SERIES_ORANGE for value in ordered.coverage_with_margin]
ax.bar(ordered.label, 100 * ordered.coverage_with_margin, color=bar_colors, width=0.7)
ax.axhline(100 * CAPACITY_QUANTILE, color=TEXT_PRIMARY, linestyle="dashed", linewidth=1.3, label="Target coverage")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="lower right")
finish_chart(fig, ax, "Capacity Model Coverage on Test Minutes", "Callee microservice", "Container minutes covered (%)",
             "capacity_coverage.png")

below_target = capacity_validation[capacity_validation.coverage_with_margin < CAPACITY_QUANTILE]
tell(
    f"We wanted the model to stay above the real CPU {100 * CAPACITY_QUANTILE:.0f}% of the time.",
    f"In a typical callee it did so {100 * capacity_validation.coverage_with_margin.median():.0f}% of the time.",
    f"Callees below the target: {callee_list(below_target.label)}." if len(below_target) else "Every callee reached the target.",
    "For those callees the safe capacity is a little optimistic, so a larger margin would be safer." if len(below_target)
    else "So the safe capacities can be trusted on new minutes."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Is a straight line good enough?
# MAGIC
# MAGIC The capacity model assumes CPU grows in a straight line with calls. On the held-out test minutes it is compared with a
# MAGIC traffic blind model that predicts each container's training median CPU. The residual (real CPU minus model CPU) is also
# MAGIC checked by load level. A residual that grows at high load means CPU grows faster than the line, so safe capacity is optimistic.

# COMMAND ----------

training_container_cpu = training_rows[training_rows.t_idx < CALIBRATION_FIRST_MINUTE].groupby("msinstanceid").cpu_utilization.median()
training_callee_cpu = training_rows[training_rows.t_idx < CALIBRATION_FIRST_MINUTE].groupby("msname").cpu_utilization.median()
linearity_rows = []
residual_parts = []
for name in balanced_callees:
    model = capacity_models[name]
    rows = test_rows[test_rows.msname == name].dropna(subset=["cpu_utilization", "incoming_mcr"])
    if len(rows) < 20:
        continue
    linear_cpu = rows.msinstanceid.map(lambda container_id: container_baseline(model, container_id)) + model["cpu_per_call"] * rows.incoming_mcr
    blind_cpu = rows.msinstanceid.map(training_container_cpu).fillna(training_callee_cpu.get(name, np.nan))
    residual = rows.cpu_utilization - linear_cpu
    total_variation = np.sum((rows.cpu_utilization - rows.cpu_utilization.mean()) ** 2)
    linearity_rows.append({
        "label": callee_label[name],
        "test_container_minutes": len(rows),
        "linear_mae_percent": 100 * residual.abs().mean(),
        "traffic_blind_mae_percent": 100 * (rows.cpu_utilization - blind_cpu).abs().mean(),
        "linear_r2": 1 - np.sum(residual ** 2) / total_variation if total_variation > 0 else np.nan,
        "bias_percent": 100 * residual.mean()
    })
    load_level = np.ceil(rows.incoming_mcr.rank(pct=True) * 5).clip(1, 5).astype(int)
    residual_parts.append(pd.DataFrame({"load_level": load_level, "residual_percent": 100 * residual}))
capacity_linearity = pd.DataFrame(linearity_rows)
show_table(capacity_linearity.round(3), "capacity_linearity")

by_level = pd.concat(residual_parts).groupby("load_level").residual_percent.quantile([0.25, 0.5, 0.75]).unstack()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(by_level.index, by_level[0.25], by_level[0.75], color=SERIES_BLUE, alpha=0.2, linewidth=0, label="Middle half")
ax.plot(by_level.index, by_level[0.5], color=SERIES_BLUE, marker="o", linewidth=2, label="Median")
ax.axhline(0, color=TEXT_PRIMARY, linewidth=1)
ax.set_xticks(range(1, 6))
ax.set_xticklabels(["Lowest 20%", "20 to 40%", "40 to 60%", "60 to 80%", "Highest 20%"])
ax.legend(loc="upper left")
finish_chart(fig, ax, "Capacity Model Residual by Load Level", "Container call rate within the callee", "Real minus model CPU (points)",
             "capacity_residual_by_load.png")

wins = int((capacity_linearity.linear_mae_percent < capacity_linearity.traffic_blind_mae_percent).sum())
high_load_gap = float(by_level[0.5].iloc[-1] - by_level[0.5].iloc[0])
tell(
    f"The straight line beats the traffic blind model in {wins} of {len(capacity_linearity)} callees.",
    f"Its median R squared on the test minutes is {capacity_linearity.linear_r2.median():.2f}.",
    f"From the quietest to the busiest fifth of minutes the median residual moves by {high_load_gap:+.2f} CPU points.",
    "The residual rises at high load, so CPU grows faster than the line there and the margin matters most." if high_load_gap > 1
    else "The residual stays flat across load levels, so a straight line is a fair description of these callees."
)

# COMMAND ----------

positive_capacities = np.array([value for value in safe_capacities if value > 0])
fig, ax = plt.subplots(figsize=(10, 4.5))
if len(positive_capacities):
    edges = np.logspace(np.log10(positive_capacities.min()), np.log10(positive_capacities.max()) + 1e-9, 41)
    counts, _ = np.histogram(positive_capacities, bins=edges)
    heights = np.append(counts, counts[-1])
    ax.fill_between(edges, heights, step="post", color=SERIES_BLUE, alpha=0.22, linewidth=0)
    ax.step(edges, heights, where="post", color=SERIES_BLUE, linewidth=1.6)
    ax.axvline(np.median(positive_capacities), color=TEXT_PRIMARY, linestyle="dashed", linewidth=1.3, label="Median")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
finish_chart(fig, ax, "Safe Capacity per Container", "Safe calls per second (log scale)", "Containers", "safe_capacity.png")

no_headroom = np.mean(np.array(safe_capacities) <= 0) if safe_capacities else 0.0
if len(positive_capacities):
    tell(
        f"A typical container can safely take about {np.median(positive_capacities):,.0f} calls per second.",
        f"The weakest take about {np.percentile(positive_capacities, 5):,.0f}, the strongest more than {np.percentile(positive_capacities, 95):,.0f}.",
        "Containers are not equal. Some start with a higher CPU before any calls arrive.",
        "That is why sending the same number of calls to every container is not always fair."
        if no_headroom == 0 else f"{100 * no_headroom:.1f}% of containers are already too busy before any calls arrive."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load balancing algorithms
# MAGIC
# MAGIC Each algorithm returns the fraction of the callee's calls sent to each container in one minute. Every incoming connection uses
# MAGIC the same split, because CPU per call is estimated per callee and all connections into a callee cost the same CPU.
# MAGIC
# MAGIC | Algorithm | Rule |
# MAGIC |---|---|
# MAGIC | Observed | The split in the trace. Reference only, available without scale-out. |
# MAGIC | Round Robin | Equal share for every container. |
# MAGIC | Least Loaded | Calls arrive in `LEAST_LOADED_CHUNKS_PER_CONTAINER x containers` equal chunks. Each chunk goes to the container with the lowest current CPU (greedy list scheduling). |
# MAGIC | Weighted Routing | Share proportional to each container's safe capacity. |
# MAGIC | Linear Programming | Minimises the highest container CPU, plus a small cost for moving traffic between containers from the previous minute. |
# MAGIC
# MAGIC The linear program for one callee and one minute, with containers `k`, planned demand `D`, CPU per call `b` and baseline `a_k`:
# MAGIC
# MAGIC * minimise `z + w x sum_k (u_k + v_k)`
# MAGIC * subject to `a_k + b x D x x_k <= z` for every container, `sum_k x_k = 1`, `x_k - previous_k = u_k - v_k`, `x_k, u_k, v_k >= 0`
# MAGIC
# MAGIC It is solved with the HiGHS solver in `scipy.optimize.linprog`.

# COMMAND ----------

def round_robin_split(baselines, demand, cpu_per_call, previous_fractions):
    return np.full(len(baselines), 1.0 / len(baselines))


def weighted_split(baselines, demand, cpu_per_call, previous_fractions, capacities):
    positive = np.maximum(capacities, 0.0)
    if positive.sum() <= 0:
        return np.full(len(baselines), 1.0 / len(baselines))
    return positive / positive.sum()


def water_filling_fractions(baselines, demand, cpu_per_call):
    """Split that raises the coolest containers to one common CPU level. It is the limit of least loaded with very small chunks."""
    baselines = np.asarray(baselines, dtype=float)
    total_cpu = cpu_per_call * demand
    if total_cpu <= 0:
        return np.full(len(baselines), 1.0 / len(baselines))
    ordered = np.sort(baselines)
    filled_count = np.arange(1, len(ordered) + 1)
    level_candidates = (np.cumsum(ordered) + total_cpu) / filled_count
    valid = level_candidates >= ordered
    level = level_candidates[np.nonzero(valid)[0].max()]
    return np.maximum(level - baselines, 0.0) / total_cpu


def least_loaded_split(baselines, demand, cpu_per_call, previous_fractions):
    """Greedy chunk assignment. Whole chunks follow the water filling split, the remaining chunks go one by one to the coolest container."""
    baselines = np.asarray(baselines, dtype=float)
    chunk_count = LEAST_LOADED_CHUNKS_PER_CONTAINER * len(baselines)
    chunk_cpu = cpu_per_call * demand / chunk_count
    chunks = np.floor(water_filling_fractions(baselines, demand, cpu_per_call) * chunk_count + 1e-9)
    current_cpu = baselines + chunks * chunk_cpu
    for _ in range(int(chunk_count - chunks.sum())):
        target = int(np.argmin(current_cpu))
        chunks[target] += 1
        current_cpu[target] += chunk_cpu
    return chunks / chunk_count


def linear_programming_split(baselines, demand, cpu_per_call, previous_fractions):
    container_count = len(baselines)
    if demand <= 0 or container_count == 1:
        return np.full(container_count, 1.0 / container_count)
    previous = previous_fractions if previous_fractions is not None else np.full(container_count, 1.0 / container_count)
    identity = sparse.identity(container_count, format="csr")
    zero_block = sparse.csr_matrix((container_count, container_count))
    peak_column = sparse.csr_matrix(-np.ones((container_count, 1)))
    capacity_rows = sparse.hstack([identity * (cpu_per_call * demand), peak_column, zero_block, zero_block])
    sum_row = sparse.hstack([sparse.csr_matrix(np.ones((1, container_count))), sparse.csr_matrix((1, 1 + 2 * container_count))])
    stability_rows = sparse.hstack([identity, sparse.csr_matrix((container_count, 1)), -identity, identity])
    objective = np.concatenate([np.zeros(container_count), [1.0], np.full(2 * container_count, LP_STABILITY_WEIGHT)])
    bounds = [(0, 1)] * container_count + [(None, None)] + [(0, None)] * (2 * container_count)
    result = linprog(
        objective,
        A_ub=capacity_rows, b_ub=-np.asarray(baselines, dtype=float),
        A_eq=sparse.vstack([sum_row, stability_rows]), b_eq=np.concatenate([[1.0], previous]),
        bounds=bounds, method="highs"
    )
    if not result.success:
        return least_loaded_split(baselines, demand, cpu_per_call, previous_fractions)
    fractions = np.clip(result.x[:container_count], 0, None)
    return fractions / fractions.sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scale-out
# MAGIC
# MAGIC Before routing, the planned demand is compared with the total safe capacity of the callee's containers.
# MAGIC When it is larger, new containers are added:
# MAGIC
# MAGIC `extra containers = ceil((planned demand - total safe capacity) / safe capacity of a new container)`
# MAGIC
# MAGIC A new container starts with the callee's median baseline. Added containers are kept until the requirement stays lower for
# MAGIC `SCALE_IN_DELAY_MINUTES` minutes, which avoids adding and removing containers every minute (the same idea as the
# MAGIC Kubernetes HPA scale-down stabilisation window). The number of extra containers is capped at `MAX_EXTRA_CONTAINER_FACTOR` times the current count.
# MAGIC Each new container also needs memory. Memory does not follow traffic in this trace, so the memory of a new container is the callee's median container memory.
# MAGIC Added memory is a sum of utilisation shares (1.0 is one container's full memory limit), not GB.

# COMMAND ----------

def required_extra_containers(model, planned_demand, capacities, container_count):
    shortfall = planned_demand - np.maximum(capacities, 0.0).sum()
    if shortfall <= 0 or model["new_container_capacity"] <= 0:
        return 0
    extra = math.ceil(shortfall / model["new_container_capacity"])
    return int(min(extra, math.ceil(MAX_EXTRA_CONTAINER_FACTOR * container_count)))


class ScaleOutController:
    """Keeps the number of extra containers, with a delay before removing them."""

    def __init__(self):
        self.extra_containers = 0
        self.minutes_below = 0

    def update(self, required):
        if required >= self.extra_containers:
            self.extra_containers = required
            self.minutes_below = 0
        else:
            self.minutes_below += 1
            if self.minutes_below >= SCALE_IN_DELAY_MINUTES:
                self.extra_containers = required
                self.minutes_below = 0
        return self.extra_containers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simulation and validation with actual future demand
# MAGIC
# MAGIC Each policy is replayed on the test minutes. The **planned demand** is what the policy believes the callee will receive.
# MAGIC The containers then receive the **actual demand** of that minute, split with the planned fractions.
# MAGIC
# MAGIC | Demand source | Planned demand for minute t |
# MAGIC |---|---|
# MAGIC | oracle | The actual demand of minute t. Perfect information, the best any forecast can reach. |
# MAGIC | persistence | The actual demand `FORECAST_HORIZON` minutes earlier. The simplest forecast. |
# MAGIC
# MAGIC Section 8 adds the GNN forecasts as further demand sources.
# MAGIC
# MAGIC Minutes where persistence has no earlier value are left out for every source, so every source is scored on the same minutes.
# MAGIC The simulation assumes the platform knows which containers run in a minute, because it starts and stops them itself.
# MAGIC
# MAGIC A container is **overloaded** when its CPU from the capacity model (without the margin) exceeds `TARGET_UTILISATION`.
# MAGIC A connection is overloaded in a minute when part of its calls land on an overloaded container.

# COMMAND ----------

test_traffic = container_minutes[(container_minutes.period == "test") & container_minutes.msname.isin(balanced_callees)]
test_minutes = list(range(TEST_FIRST_MINUTE, LAST_MINUTE + 1))
containers_by_minute = {
    name: {minute: rows for minute, rows in test_traffic[test_traffic.msname == name].groupby("t_idx")}
    for name in balanced_callees
}
actual_demand = {
    name: test_traffic[test_traffic.msname == name].groupby("t_idx").incoming_mcr.sum().reindex(test_minutes)
    for name in balanced_callees
}
memory_per_container = training_rows.groupby("msname").memory_utilization.median().to_dict()
callee_demand = container_minutes.groupby(["msname", "t_idx"]).incoming_mcr.sum()



def planned_demand_series(name, source):
    actual = actual_demand[name]
    if source == "oracle":
        return actual
    full_series = callee_demand.loc[name].reindex(range(FIRST_MINUTE, LAST_MINUTE + 1)).ffill()
    return full_series.shift(FORECAST_HORIZON).reindex(test_minutes)

# COMMAND ----------

def split_traffic(balancer, baselines, capacities, planned, cpu_per_call, previous, observed_loads):
    if balancer == "Observed":
        total = observed_loads.sum()
        return observed_loads / total if total > 0 else np.full(len(baselines), 1.0 / len(baselines))
    if balancer == "Round Robin":
        return round_robin_split(baselines, planned, cpu_per_call, previous)
    if balancer == "Least Loaded":
        return least_loaded_split(baselines, planned, cpu_per_call, previous)
    if balancer == "Weighted Routing":
        return weighted_split(baselines, planned, cpu_per_call, previous, capacities)
    return linear_programming_split(baselines, planned, cpu_per_call, previous)


def simulate_callee(name, balancer, source, multiplier, scale_out):
    model = capacity_models[name]
    planned_series = planned_demand_series(name, source) * multiplier
    persistence_defined = planned_demand_series(name, "persistence").notna()
    controller = ScaleOutController()
    previous_by_container = {}
    minute_results = []
    for minute in test_minutes:
        rows = containers_by_minute[name].get(minute)
        if rows is None or rows.empty or not persistence_defined.get(minute, False):
            continue
        container_ids = list(rows.msinstanceid)
        observed_loads = rows.incoming_mcr.to_numpy(dtype=float) * multiplier
        actual = observed_loads.sum()
        planned = float(planned_series[minute])
        baselines = np.array([container_baseline(model, container_id) for container_id in container_ids])
        capacities = np.array([safe_capacity(model, value) for value in baselines])
        extra = 0
        if scale_out:
            extra = controller.update(required_extra_containers(model, planned, capacities, len(container_ids)))
            if extra:
                container_ids = container_ids + [f"added_{number}" for number in range(extra)]
                baselines = np.append(baselines, np.full(extra, model["median_baseline"]))
                capacities = np.append(capacities, np.full(extra, model["new_container_capacity"]))
                observed_loads = np.append(observed_loads, np.zeros(extra))
        previous = np.array([previous_by_container.get(container_id, 0.0) for container_id in container_ids])
        previous = previous / previous.sum() if previous.sum() > 0 else None
        fractions = split_traffic(balancer, baselines, capacities, planned, model["cpu_per_call"], previous, observed_loads)
        loads = fractions * actual
        cpu = baselines + model["cpu_per_call"] * loads
        overloaded = cpu > TARGET_UTILISATION
        limit_calls = np.maximum((TARGET_UTILISATION - baselines) / model["cpu_per_call"], 0.0)
        mean_cpu = cpu.mean()
        minute_results.append({
            "msname": name,
            "t_idx": minute,
            "containers": len(container_ids),
            "extra_containers": extra,
            "actual_demand": actual,
            "planned_demand": planned,
            "peak_cpu": cpu.max(),
            "overloaded_containers": int(overloaded.sum()),
            "overloaded_call_share": float(fractions[overloaded].sum()),
            "excess_calls": float(np.maximum(loads - limit_calls, 0.0).sum()),
            "cpu_coefficient_of_variation": cpu.std() / mean_cpu if mean_cpu > 0 else 0.0,
            "jain_fairness": (cpu.sum() ** 2) / (len(cpu) * (cpu ** 2).sum()) if (cpu ** 2).sum() > 0 else 1.0,
            "routing_change": 0.5 * np.abs(fractions - (previous if previous is not None else fractions)).sum(),
            "added_memory_share": extra * memory_per_container.get(name, np.nan)
        })
        previous_by_container = dict(zip(container_ids, fractions))
    return minute_results

# COMMAND ----------

policies = [("Observed", False)] + [(balancer, scale_out) for balancer in BALANCERS for scale_out in (False, True)]
all_minutes = []
for source in DEMAND_SOURCES:
    for multiplier in DEMAND_MULTIPLIERS:
        for balancer, scale_out in policies:
            if balancer == "Observed" and source != "oracle":
                continue
            for name in balanced_callees:
                for row in simulate_callee(name, balancer, source, multiplier, scale_out):
                    row.update({"source": source, "multiplier": multiplier, "balancer": balancer, "scale_out": scale_out})
                    all_minutes.append(row)
        print(f"finished demand source {source}, demand x {multiplier}")
minute_results = pd.DataFrame(all_minutes)
minute_results.to_csv(os.path.join(OUTPUT_DIR, "minute_results.csv"), index=False)
print(f"simulated callee minutes: {len(minute_results):,}")
tell(
    f"We replayed {len(test_minutes)} unseen minutes for {len(balanced_callees)} callees.",
    f"Every policy was tested at normal demand and at {DEMAND_MULTIPLIERS[-1]:g} times the demand.",
    "Each policy saw the same calls. Only the way calls were shared between containers changed."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Results
# MAGIC
# MAGIC Each row summarises one policy over all callees and test minutes.
# MAGIC
# MAGIC * **Excess calls** is the main overload measure: calls per second above the safe capacity, summed over containers and averaged over minutes.
# MAGIC   It is the volume of traffic on overloaded connections.
# MAGIC * **Overloaded container minutes** is the share of container minutes above the safe CPU.
# MAGIC * **Overloaded calls** is the share of calls that land on overloaded containers. It equals the share of connection traffic that is overloaded.
# MAGIC * **CPU imbalance** is the mean coefficient of variation of container CPU. **Jain fairness** is 1 when all containers have the same CPU.
# MAGIC * **Added container minutes** is the cost of scale-out. **Routing change** is the mean share of traffic moved between containers per minute.

# COMMAND ----------

def summarise(rows):
    total_container_minutes = rows.containers.sum()
    return pd.Series({
        "overloaded_container_minutes_percent": 100 * rows.overloaded_containers.sum() / total_container_minutes,
        "callee_minutes_with_overload_percent": 100 * (rows.overloaded_containers > 0).mean(),
        "overloaded_calls_percent": 100 * (rows.overloaded_call_share * rows.actual_demand).sum() / rows.actual_demand.sum(),
        "mean_excess_calls_per_second": rows.groupby("t_idx").excess_calls.sum().mean(),
        "mean_peak_cpu_percent": 100 * rows.peak_cpu.mean(),
        "cpu_imbalance": rows.cpu_coefficient_of_variation.mean(),
        "jain_fairness": rows.jain_fairness.mean(),
        "added_container_minutes": rows.extra_containers.sum(),
        "routing_change": rows.routing_change.mean()
    })


summary = (
    minute_results.groupby(["source", "multiplier", "balancer", "scale_out"]).apply(summarise).reset_index()
)
summary["policy"] = summary.balancer + np.where(summary.scale_out, " + scale-out", "")
show_table(summary.round(3), "policy_summary")

normal = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[0])]
stressed = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[-1])]
best_normal = normal.sort_values(["mean_excess_calls_per_second", "mean_peak_cpu_percent", "added_container_minutes"]).iloc[0]
best_stressed = stressed.sort_values(["mean_excess_calls_per_second", "added_container_minutes", "mean_peak_cpu_percent"]).iloc[0]
tell(
    "This table is the score card. Lower excess calls and lower peak CPU are better.",
    f"At normal demand the best policy is {best_normal.policy}. Its hottest container averages {best_normal.mean_peak_cpu_percent:.1f}% CPU.",
    f"At {DEMAND_MULTIPLIERS[-1]:g} times the demand the best policy is {best_stressed.policy}, "
    f"with {calls(best_stressed.mean_excess_calls_per_second)} excess calls per second.",
    "The charts below tell the same story step by step."
)

# COMMAND ----------

def policy_value(frame, balancer, scale_out, column):
    rows = frame[(frame.balancer == balancer) & (frame.scale_out == scale_out)]
    return float(rows[column].iloc[0]) if len(rows) else np.nan


fig, axes = plt.subplots(1, len(DEMAND_MULTIPLIERS), figsize=(7 * len(DEMAND_MULTIPLIERS), 5), sharey=True, squeeze=False)
bar_width = 0.38
for ax, multiplier in zip(axes[0], DEMAND_MULTIPLIERS):
    frame = summary[(summary.source == "oracle") & (summary.multiplier == multiplier)]
    names = ["Observed"] + BALANCERS
    positions = np.arange(len(names))
    without = [policy_value(frame, name, False, "mean_excess_calls_per_second") for name in names]
    with_scale = [np.nan] + [policy_value(frame, name, True, "mean_excess_calls_per_second") for name in BALANCERS]
    ax.bar(positions - bar_width / 2, without, width=bar_width, color=[BALANCER_COLORS[name] for name in names])
    ax.bar(positions + bar_width / 2, with_scale, width=bar_width, color=[BALANCER_COLORS[name] for name in names], alpha=0.45,
           hatch="//", edgecolor=SURFACE_COLOR)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title(f"Demand x {multiplier:g}", fontsize=12)
    ax.legend(handles=[Patch(facecolor=TEXT_MUTED, label="Without scale-out"),
                       Patch(facecolor=TEXT_MUTED, alpha=0.45, hatch="//", edgecolor=SURFACE_COLOR, label="With scale-out")],
              loc="upper right")
axes[0][0].set_ylabel("Calls per second above safe capacity (symmetric log scale)")
axes[0][0].yaxis.set_major_formatter(THOUSANDS)
finish_figure(fig, "Excess Calls by Load Balancer", "Load balancer", None, "excess_calls_by_balancer.png")

def excess(frame, balancer, scale_out=False):
    return policy_value(frame, balancer, scale_out, "mean_excess_calls_per_second")


normal = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[0])]
stressed = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[-1])]
balancers_clean_normal = [name for name in BALANCERS if excess(normal, name) < 1]
best_routing_stressed = min(BALANCERS, key=lambda name: excess(stressed, name))
tell(
    f"At normal demand, the routing in the trace pushes {calls(excess(normal, 'Observed'))} calls per second above safe capacity.",
    f"With {callee_list(balancers_clean_normal)}, almost nothing is left." if balancers_clean_normal
    else "Every balancer still leaves some excess at normal demand.",
    "So the overload in the trace comes from uneven sharing, not from too few containers." if balancers_clean_normal else "",
    f"At {DEMAND_MULTIPLIERS[-1]:g} times the demand, the trace routing pushes {calls(excess(stressed, 'Observed'))} calls per second over the line.",
    f"The best routing, {best_routing_stressed}, cuts this to {calls(excess(stressed, best_routing_stressed))}.",
    "It cannot reach zero, because there are not enough containers. Only adding containers removes the rest."
    if excess(stressed, best_routing_stressed) >= 1 else "Good sharing alone almost removes the overload.",
    f"With scale-out, Linear Programming leaves {calls(excess(stressed, 'Linear Programming', True))} and "
    f"Round Robin leaves {calls(excess(stressed, 'Round Robin', True))}.",
    "Round Robin still overloads, because it ignores that some containers start hotter than others."
    if excess(stressed, "Round Robin", True) > excess(stressed, "Linear Programming", True) + 1 else ""
)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
frame = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[0]) & ~summary.scale_out]
names = [name for name in ["Observed"] + BALANCERS if name in set(frame.balancer)]
values = frame.set_index("balancer").reindex(names)
axes[0].barh(names[::-1], values.cpu_imbalance[::-1], color=[BALANCER_COLORS[name] for name in names[::-1]], height=0.6)
axes[0].set_xlabel("Mean coefficient of variation of container CPU")
axes[0].set_ylabel("Load balancer")
axes[0].set_title("CPU Imbalance", fontsize=12)
axes[1].barh(names[::-1], values.jain_fairness[::-1], color=[BALANCER_COLORS[name] for name in names[::-1]], height=0.6)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("Mean Jain fairness index")
axes[1].set_title("Fairness", fontsize=12)
axes[1].set_yticklabels([])
for ax in axes:
    ax.grid(axis="y", visible=False)
finish_figure(fig, "Load Imbalance Across Containers", None, None, "imbalance_by_balancer.png")

fairest = values.jain_fairness.idxmax()
least_fair = values.jain_fairness.idxmin()
tell(
    "A fairness of 1 means every container works equally hard.",
    f"{fairest} is the fairest, at {values.loc[fairest, 'jain_fairness']:.3f}.",
    f"{least_fair} is the least fair, at {values.loc[least_fair, 'jain_fairness']:.3f}.",
    f"Round Robin gives every container the same number of calls and scores {values.loc['Round Robin', 'jain_fairness']:.3f}.",
    "Equal calls do not mean equal CPU, because containers start from different CPU levels."
)

# COMMAND ----------

busiest_callee = max(balanced_callees, key=lambda name: actual_demand[name].sum())
fig, ax = plt.subplots(figsize=(12, 4.8))
for balancer in ["Observed"] + BALANCERS:
    rows = minute_results[(minute_results.msname == busiest_callee) & (minute_results.source == "oracle")
                          & (minute_results.multiplier == DEMAND_MULTIPLIERS[-1]) & (minute_results.balancer == balancer)
                          & ~minute_results.scale_out]
    ax.plot(rows.t_idx, 100 * rows.peak_cpu, color=BALANCER_COLORS[balancer], linewidth=1.6, label=balancer)
ax.axhline(100 * TARGET_UTILISATION, color=TEXT_PRIMARY, linestyle="dashed", linewidth=1.2, label="Safe CPU")
ax.set_xlim(TEST_FIRST_MINUTE, LAST_MINUTE)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
finish_chart(fig, ax, f"Peak Container CPU of {callee_label[busiest_callee]}", "Minute of trace", "Highest container CPU (%)",
             "peak_cpu_busiest_callee.png")

peak_by_balancer = (
    minute_results[(minute_results.msname == busiest_callee) & (minute_results.source == "oracle")
                   & (minute_results.multiplier == DEMAND_MULTIPLIERS[-1]) & ~minute_results.scale_out]
    .groupby("balancer").peak_cpu.mean() * 100
)
coolest = peak_by_balancer.idxmin()
tell(
    f"This is {callee_label[busiest_callee]}, the busiest callee, at {DEMAND_MULTIPLIERS[-1]:g} times the demand.",
    f"With the trace routing, its hottest container averages {peak_by_balancer.get('Observed', np.nan):.1f}% CPU.",
    f"With {coolest}, it averages {peak_by_balancer[coolest]:.1f}%.",
    "All lines stay under the safe line, so this callee is safe with any balancer."
    if peak_by_balancer.max() < 100 * TARGET_UTILISATION else "Some lines cross the safe line, so this callee needs a smart balancer or more containers."
)

# COMMAND ----------

scaled = minute_results[(minute_results.balancer == "Linear Programming") & minute_results.scale_out]
added_by_minute = scaled.groupby(["source", "multiplier", "t_idx"]).extra_containers.sum().reset_index()
fig, axes = plt.subplots(len(DEMAND_MULTIPLIERS), 1, figsize=(12, 3.6 * len(DEMAND_MULTIPLIERS)), sharex=True, squeeze=False)
source_colors = {"oracle": SERIES_BLUE, "persistence": SERIES_ORANGE, "forecast": SERIES_AQUA}
for ax, multiplier in zip(axes[:, 0], DEMAND_MULTIPLIERS):
    for source in DEMAND_SOURCES:
        rows = added_by_minute[(added_by_minute.source == source) & (added_by_minute.multiplier == multiplier)]
        ax.step(rows.t_idx, rows.extra_containers, where="post", color=source_colors[source], linewidth=1.6, label=source.capitalize())
    ax.set_title(f"Demand x {multiplier:g}", loc="left", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncol=3)
axes[-1, 0].set_xlim(TEST_FIRST_MINUTE, LAST_MINUTE)
finish_figure(fig, "Containers Added by Scale-Out", "Minute of trace", "Extra containers", "containers_added.png")

oracle_added = added_by_minute[(added_by_minute.source == "oracle")]
normal_added = oracle_added[oracle_added.multiplier == DEMAND_MULTIPLIERS[0]].extra_containers
stressed_added = oracle_added[oracle_added.multiplier == DEMAND_MULTIPLIERS[-1]]
busy_minutes = stressed_added[stressed_added.extra_containers > 0].t_idx
tell(
    "At normal demand no extra containers are needed." if normal_added.max() == 0
    else f"Even at normal demand up to {normal_added.max():.0f} extra containers are needed.",
    f"At {DEMAND_MULTIPLIERS[-1]:g} times the demand, up to {stressed_added.extra_containers.max():.0f} extra "
    f"{'container is' if stressed_added.extra_containers.max() == 1 else 'containers are'} added at the same time.",
    f"They are needed in {len(busy_minutes)} of {len(test_minutes)} minutes, between minute {busy_minutes.min()} and minute {busy_minutes.max()}."
    if len(busy_minutes) else "",
    "Extra containers come only in short bursts. The rest of the time the existing containers are enough."
)

# COMMAND ----------

comparison = summary[summary.scale_out & (summary.balancer == "Linear Programming")]
comparison = comparison[["source", "multiplier", "overloaded_container_minutes_percent", "overloaded_calls_percent",
                         "added_container_minutes", "mean_excess_calls_per_second"]]
show_table(comparison.round(3), "oracle_validation")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
positions = np.arange(len(DEMAND_MULTIPLIERS))
width = 0.8 / len(DEMAND_SOURCES)
for offset, source in enumerate(DEMAND_SOURCES):
    rows = comparison[comparison.source == source].set_index("multiplier").reindex(DEMAND_MULTIPLIERS)
    shift = (offset - (len(DEMAND_SOURCES) - 1) / 2) * width
    axes[0].bar(positions + shift, rows.mean_excess_calls_per_second, width=width, color=source_colors[source], label=source.capitalize())
    axes[1].bar(positions + shift, rows.added_container_minutes, width=width, color=source_colors[source], label=source.capitalize())
for ax, label, title in ((axes[0], "Calls per second above safe capacity", "Excess calls"), (axes[1], "Added container minutes", "Scale-out cost")):
    ax.set_xticks(positions)
    ax.set_xticklabels([f"x {multiplier:g}" for multiplier in DEMAND_MULTIPLIERS])
    ax.set_xlabel("Demand multiplier")
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left")
axes[0].yaxis.set_major_formatter(THOUSANDS)
axes[1].yaxis.set_major_formatter(THOUSANDS)
finish_figure(fig, "Oracle and Persistence with Linear Programming and Scale-Out", None, None, "oracle_vs_persistence.png")

oracle_row = comparison[(comparison.source == "oracle") & (comparison.multiplier == DEMAND_MULTIPLIERS[-1])]
persistence_row = comparison[(comparison.source == "persistence") & (comparison.multiplier == DEMAND_MULTIPLIERS[-1])]
if len(oracle_row) and len(persistence_row):
    oracle_excess = float(oracle_row.mean_excess_calls_per_second.iloc[0])
    persistence_excess = float(persistence_row.mean_excess_calls_per_second.iloc[0])
    tell(
        "The oracle knows the exact demand of the next minute. It shows the best result possible.",
        f"With that knowledge, {calls(oracle_excess)} excess calls per second remain.",
        f"Persistence simply reuses last minute's demand. It leaves {calls(persistence_excess)} excess calls per second.",
        "Persistence reacts one minute late. It misses the first minute of every burst.",
        "This gap is the room for a good forecast. A GNN forecast should close part of it."
    )

# COMMAND ----------

def connection_overload(source, multiplier, balancer, scale_out):
    rows = minute_results[(minute_results.source == source) & (minute_results.multiplier == multiplier)
                          & (minute_results.balancer == balancer) & (minute_results.scale_out == scale_out)]
    per_callee = rows.groupby("msname").agg(
        excess_calls=("excess_calls", "sum"), overloaded_minutes=("overloaded_containers", lambda values: int((values > 0).sum()))
    )
    table = connections[connections.callee.isin(per_callee.index)].copy()
    table["overloaded_minutes"] = table.callee.map(per_callee.overloaded_minutes)
    table["excess_calls_per_second"] = table.share * table.callee.map(per_callee.excess_calls) / len(test_minutes)
    return table.sort_values("excess_calls_per_second", ascending=False)


stress = DEMAND_MULTIPLIERS[-1]
before = connection_overload("oracle", stress, "Round Robin", False)
after = connection_overload("oracle", stress, "Linear Programming", True)
before = before[before.excess_calls_per_second > 0]
top_connections = before.head(15)[["connection", "caller", "callee", "share", "overloaded_minutes", "excess_calls_per_second"]]
top_connections = top_connections.merge(
    after[["caller", "callee", "overloaded_minutes", "excess_calls_per_second"]],
    on=["caller", "callee"], suffixes=("_round_robin", "_lp_scale_out")
)
show_table(top_connections.round(3), "overloaded_connections")

fig, ax = plt.subplots(figsize=(11, 0.45 * len(top_connections) + 1.8))
positions = np.arange(len(top_connections))[::-1]
ax.barh(positions + 0.2, top_connections.excess_calls_per_second_round_robin, height=0.4, color=SERIES_ORANGE, label="Round Robin")
ax.barh(positions - 0.2, top_connections.excess_calls_per_second_lp_scale_out, height=0.4, color=SERIES_BLUE,
        label="Linear Programming + scale-out")
ax.set_yticks(positions)
ax.set_yticklabels(top_connections.connection, fontsize=9)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
finish_chart(fig, ax, "Most Overloaded Connections", "Excess calls per second (mean over test minutes)", "Connection",
             "overloaded_connections.png", count_axis="x")

round_robin_excess = top_connections.excess_calls_per_second_round_robin.sum()
if round_robin_excess > 0:
    reduction = 1 - top_connections.excess_calls_per_second_lp_scale_out.sum() / round_robin_excess
    worst = top_connections.iloc[0]
    hot_callees = top_connections.callee.map(callee_label).unique()
    tell(
        f"The most overloaded connection is {worst.connection}.",
        f"It carries {100 * worst.share:.0f}% of that callee's calls and is overloaded in {worst.overloaded_minutes_round_robin:.0f} of {len(test_minutes)} minutes.",
        f"The overloaded connections all lead into {callee_list(hot_callees)}.",
        "So the problem sits in a few callees, not everywhere." if len(hot_callees) <= 3 else "",
        f"Linear Programming with scale-out removes {100 * reduction:.0f}% of the excess on these connections."
    )
else:
    tell(f"No connection is overloaded under Round Robin at {stress:g} times the demand.",
         "Increase DEMAND_MULTIPLIERS to stress the callees.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. GNN driven allocation
# MAGIC
# MAGIC This section connects the forecasting and classification notebooks to load balancing. It runs four steps for every callee and test minute.
# MAGIC
# MAGIC 1. **Predict demand.** A CPU forecast says how busy the callee's containers will be. The capacity model turns it into calls per second:
# MAGIC    `planned calls = containers x (forecast CPU - mean baseline) / CPU per call`, over the containers MSResource reports in that minute,
# MAGIC    because the forecast target is their mean CPU. A safety margin in calls is then added. It is the split conformal quantile
# MAGIC    (level `FORECAST_MARGIN_LEVEL`) of that source's errors in calls on the **calibration minutes**, computed per callee.
# MAGIC    The calibration minutes were held out from tuning and early stopping in the forecasting notebooks, so the errors are honest.
# MAGIC    Every forecast source gets a margin by the same rule, including persistence. The oracles get none.
# MAGIC 2. **Predict risky connections.** The edge classifier flags connections that will carry unusually high traffic.
# MAGIC    It scores every regular connection from what was known one minute earlier. A callee is flagged when one incoming connection is flagged.
# MAGIC 3. **Check capacity.** For a flagged callee, the planned calls are routed with the routing known `FORECAST_HORIZON` minutes earlier.
# MAGIC    If any container would go above the safe CPU, the callee needs action.
# MAGIC 4. **Act.** Traffic is redistributed with Weighted Routing or Linear Programming. If the total safe capacity is still smaller than the
# MAGIC    planned calls, containers are added. A callee that needs no action keeps the routing of the trace.
# MAGIC
# MAGIC | Demand source | Planned demand |
# MAGIC |---|---|
# MAGIC | Oracle (calls) | The actual calls of the minute. The best possible information. |
# MAGIC | Persistence (calls) | The calls `FORECAST_HORIZON` minutes earlier. |
# MAGIC | Oracle (CPU) | The actual CPU of the minute, converted to calls. It shows the error of the conversion alone. |
# MAGIC | Persistence (CPU) | The CPU `FORECAST_HORIZON` minutes earlier, plus its margin, converted to calls. The fair baseline for the GNN models. |
# MAGIC | Model (pipeline) | A GNN CPU forecast plus its margin, converted to calls. |
# MAGIC
# MAGIC | Configuration | Routing | When it acts |
# MAGIC |---|---|---|
# MAGIC | Observed routing | The trace | Never. Reference only. |
# MAGIC | Round Robin + reactive scale-out | Round Robin | Every minute, with scale-out from last minute's calls. The conventional baseline. |
# MAGIC | Weighted Routing or Linear Programming + classifier | Capacity aware | Only for callees with a flagged connection that fail the capacity check |
# MAGIC | Weighted Routing or Linear Programming + capacity check | Capacity aware | For every callee that fails the capacity check, without the classifier |
# MAGIC | Weighted Routing or Linear Programming + every minute | Capacity aware | Every minute, for every callee |
# MAGIC
# MAGIC Comparing the three gates shows what the classifier adds, and what it costs when it misses an overload.
# MAGIC The best GNN source is chosen by its error on the calibration minutes, never on the test minutes.

# COMMAND ----------

def conformal_margin(residuals, level):
    """Split conformal upper margin: the ceil((n + 1) x level)-th smallest residual."""
    residuals = np.sort(np.asarray(residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return 0.0
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * level))
    return float(residuals[rank - 1])


def cpu_to_calls(cpu_fraction, baselines, model):
    """Calls per second that give the forecast mean CPU over these containers under the capacity model."""
    return max(0.0, len(baselines) * (cpu_fraction - float(np.mean(baselines))) / model["cpu_per_call"])


MIN_CALIBRATION_MINUTES = 10
resource_rows = (
    container_resource.filter(F.col("msname").isin(balanced_callees))
    .select("msname", "msinstanceid", "t_idx").toPandas()
)
resource_containers = {
    name: {int(minute): list(rows.msinstanceid) for minute, rows in frame.groupby("t_idx")}
    for name, frame in resource_rows.groupby("msname")
}
traffic_containers = {
    name: {int(minute): rows for minute, rows in frame.groupby("t_idx")}
    for name, frame in container_minutes[container_minutes.msname.isin(balanced_callees)].groupby("msname")
}
all_minutes = range(FIRST_MINUTE, LAST_MINUTE + 1)
callee_calls = {name: callee_demand.loc[name].reindex(all_minutes) for name in balanced_callees}


def resource_baselines(name, minute):
    container_ids = resource_containers.get(name, {}).get(int(minute))
    if not container_ids:
        return None
    return np.array([container_baseline(capacity_models[name], container_id) for container_id in container_ids])


def cpu_forecast_in_calls(frame, column):
    """Point forecasts in calls per second, per callee and minute."""
    converted = {}
    for name, rows in frame.groupby("msname"):
        values = {}
        for minute, value in zip(rows.t_idx, rows[column]):
            baselines = resource_baselines(name, minute)
            if baselines is not None and np.isfinite(value):
                values[int(minute)] = cpu_to_calls(float(value), baselines, capacity_models[name])
        converted[name] = pd.Series(values, dtype=float)
    return converted


point_calls = {
    "Oracle (calls)": callee_calls,
    "Persistence (calls)": {name: callee_calls[name].ffill().shift(FORECAST_HORIZON) for name in balanced_callees}
}
cpu_point_tables = {}
calibration_minutes = list(range(CALIBRATION_FIRST_MINUTE, TEST_FIRST_MINUTE))
if CPU_FORECASTS is not None:
    first_seed = CPU_FORECASTS[CPU_FORECASTS.seed_rank == 0] if "seed_rank" in CPU_FORECASTS else CPU_FORECASTS
    forecast_rows = first_seed[first_seed.msname.isin(balanced_callees)]
    calibration_split = "cal" if (forecast_rows.split == "cal").any() else "val"
    if calibration_split == "val":
        print("The forecasts have no calibration split. Margins use validation rows, which also chose the GNN epochs, so GNN margins are optimistic.")
    calibration_minutes = sorted(forecast_rows.loc[forecast_rows.split == calibration_split, "t_idx"].unique())
    used_rows = forecast_rows[forecast_rows.split.isin([calibration_split, "test"])]
    reference_pipeline = "snapshot" if "snapshot" in set(used_rows.pipeline_label) else used_rows.pipeline_label.iloc[0]
    reference_rows = used_rows[used_rows.pipeline_label == reference_pipeline]
    reference_rows = reference_rows[reference_rows.model == reference_rows.model.iloc[0]]
    point_calls["Oracle (CPU)"] = cpu_forecast_in_calls(reference_rows, "actual")
    point_calls["Persistence (CPU)"] = cpu_forecast_in_calls(reference_rows, "persistence")
    cpu_point_tables["Persistence (CPU)"] = reference_rows.rename(columns={"persistence": "point"})
    for (model_name, pipeline_label), frame in used_rows.groupby(["model", "pipeline_label"]):
        label = f"{model_name} ({pipeline_label})"
        point_calls[label] = cpu_forecast_in_calls(frame, "predicted")
        cpu_point_tables[label] = frame.rename(columns={"predicted": "point"})

SOURCE_LABELS = list(point_calls)
ORACLE_LABELS = {"Oracle (calls)", "Oracle (CPU)"}
calls_margins = {}
for label in SOURCE_LABELS:
    calls_margins[label] = {}
    if label in ORACLE_LABELS:
        calls_margins[label] = {name: 0.0 for name in balanced_callees}
        continue
    relative_residuals, residuals_by_callee = [], {}
    for name in balanced_callees:
        actual = callee_calls[name].reindex(calibration_minutes)
        point = point_calls[label].get(name, pd.Series(dtype=float)).reindex(calibration_minutes)
        residual = (actual - point).dropna()
        residuals_by_callee[name] = residual
        scale = callee_calls[name].reindex(calibration_minutes).mean()
        if len(residual) and scale > 0:
            relative_residuals.extend(residual / scale)
    pooled_relative_margin = conformal_margin(relative_residuals, FORECAST_MARGIN_LEVEL)
    for name, residual in residuals_by_callee.items():
        if len(residual) >= MIN_CALIBRATION_MINUTES:
            calls_margins[label][name] = conformal_margin(residual, FORECAST_MARGIN_LEVEL)
        else:
            calls_margins[label][name] = float(np.nan_to_num(pooled_relative_margin * callee_calls[name].reindex(calibration_minutes).mean()))

test_minutes = list(range(TEST_FIRST_MINUTE, LAST_MINUTE + 1))
evaluation_minutes = {}
for name in balanced_callees:
    minutes = {minute for minute in test_minutes if minute in traffic_containers.get(name, {})}
    for label in SOURCE_LABELS:
        series = point_calls[label].get(name)
        minutes &= set(series.dropna().index) if series is not None else set()
    evaluation_minutes[name] = sorted(minutes)
gnn_callees = [name for name in balanced_callees if evaluation_minutes[name]]
print(f"calibration minutes: {calibration_minutes[0]} to {calibration_minutes[-1]} ({len(calibration_minutes)})")
print(f"callees with every demand source available: {len(gnn_callees)} of {len(balanced_callees)}, "
      f"callee minutes: {sum(len(evaluation_minutes[name]) for name in gnn_callees):,}")


def planned_calls(label, name, minute, multiplier):
    """Point forecast plus the calibrated margin, in calls per second, scaled to the replayed demand."""
    point = point_calls[label].get(name, pd.Series(dtype=float)).get(minute, np.nan)
    return multiplier * (float(point) + calls_margins[label].get(name, 0.0))

# COMMAND ----------

accuracy_rows = []
for label in SOURCE_LABELS:
    if label in ORACLE_LABELS:
        continue
    calibration_errors, test_errors, covered = [], [], []
    for name in gnn_callees:
        actual = callee_calls[name]
        point = point_calls[label][name]
        calibration_errors.extend((point.reindex(calibration_minutes) - actual.reindex(calibration_minutes)).dropna().abs())
        test_minutes_used = evaluation_minutes[name]
        test_errors.extend((point.reindex(test_minutes_used) - actual.reindex(test_minutes_used)).abs())
        covered.extend(actual.reindex(test_minutes_used) <= point.reindex(test_minutes_used) + calls_margins[label][name])
    row = {
        "source": label,
        "calibration_mae_calls": float(np.mean(calibration_errors)) if calibration_errors else np.nan,
        "test_mae_calls": float(np.mean(test_errors)) if test_errors else np.nan,
        "test_coverage_with_margin": float(np.mean(covered)) if covered else np.nan
    }
    table = cpu_point_tables.get(label)
    if table is not None:
        test_cpu = table[(table.split == "test") & table.msname.isin(gnn_callees)]
        row["test_mae_cpu_points"] = 100 * float((test_cpu.point - test_cpu.actual).abs().mean())
    accuracy_rows.append(row)
forecast_accuracy = pd.DataFrame(accuracy_rows).sort_values("calibration_mae_calls").reset_index(drop=True)

if CPU_FORECASTS is not None and "seed_rank" in CPU_FORECASTS and CPU_FORECASTS.seed_rank.max() > 0:
    seed_rows = CPU_FORECASTS[(CPU_FORECASTS.split == "test") & CPU_FORECASTS.msname.isin(gnn_callees)]
    seed_spread = (
        seed_rows.assign(absolute_error=(seed_rows.predicted - seed_rows.actual).abs() * 100)
        .groupby(["model", "pipeline_label", "seed"]).absolute_error.mean()
        .groupby(["model", "pipeline_label"]).std().rename("seed_sd_mae_cpu_points").reset_index()
    )
    seed_spread["source"] = seed_spread.model + " (" + seed_spread.pipeline_label + ")"
    forecast_accuracy = forecast_accuracy.merge(seed_spread[["source", "seed_sd_mae_cpu_points"]], on="source", how="left")
show_table(forecast_accuracy.round(3), "forecast_accuracy_balanced_callees")

gnn_labels = [label for label in SOURCE_LABELS if label not in ORACLE_LABELS and not label.startswith("Persistence")]
best_gnn = None
if gnn_labels:
    best_gnn = forecast_accuracy[forecast_accuracy.source.isin(gnn_labels)].sort_values("calibration_mae_calls").source.iloc[0]

fig, ax = plt.subplots(figsize=(11, 0.45 * len(forecast_accuracy) + 1.8))
ordered = forecast_accuracy.iloc[::-1]
colors = [TEXT_MUTED if label.startswith("Persistence") else SERIES_BLUE for label in ordered.source]
ax.barh(ordered.source, ordered.test_mae_calls, color=colors, height=0.65)
persistence_mae = forecast_accuracy.loc[forecast_accuracy.source == "Persistence (calls)", "test_mae_calls"]
if len(persistence_mae):
    ax.axvline(float(persistence_mae.iloc[0]), color=TEXT_PRIMARY, linestyle="dashed", linewidth=1.2, label="Persistence (calls)")
    ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
finish_chart(fig, ax, "Demand Forecast Error on Balanced Callees", "Mean absolute error (calls per second)", "Demand source",
             "forecast_error_balanced_callees.png", count_axis="x")

persistence_rows = forecast_accuracy[forecast_accuracy.source.str.startswith("Persistence")]
tell(
    "All errors are in calls per second, the unit the balancer uses. GNN forecasts are CPU forecasts turned into calls.",
    f"Chosen on the calibration minutes, the best GNN source is {best_gnn}." if best_gnn else "No GNN forecasts were found.",
    f"On test minutes it is off by {calls(float(forecast_accuracy.set_index('source').loc[best_gnn, 'test_mae_calls']))} calls per second on average."
    if best_gnn else "",
    f"Persistence is off by {callee_list([calls(value) for value in persistence_rows.test_mae_calls])} calls per second "
    f"({callee_list(list(persistence_rows.source))})." if len(persistence_rows) else "",
    f"With its margin, a typical source stays above the real demand in {100 * forecast_accuracy.test_coverage_with_margin.median():.0f}% of test minutes."
)

# COMMAND ----------

memory_plan = {}
if MEMORY_FORECASTS is not None:
    memory_first_seed = MEMORY_FORECASTS[MEMORY_FORECASTS.seed_rank == 0] if "seed_rank" in MEMORY_FORECASTS else MEMORY_FORECASTS
    memory_rows = memory_first_seed[memory_first_seed.msname.isin(balanced_callees)]
    memory_calibration_split = "cal" if (memory_rows.split == "cal").any() else "val"
    summary_rows = []
    for (model_name, pipeline_label), frame in memory_rows.groupby(["model", "pipeline_label"]):
        calibration, test = frame[frame.split == memory_calibration_split], frame[frame.split == "test"]
        margin = conformal_margin(calibration.actual - calibration.predicted, FORECAST_MARGIN_LEVEL)
        summary_rows.append({
            "source": f"{model_name} ({pipeline_label})",
            "model": model_name,
            "pipeline_label": pipeline_label,
            "calibration_mae_memory_points": 100 * (calibration.predicted - calibration.actual).abs().mean(),
            "test_mae_memory_points": 100 * (test.predicted - test.actual).abs().mean(),
            "margin_memory_points": 100 * margin,
            "test_minutes_at_memory_risk": float(np.mean(test.predicted + margin > MEMORY_LIMIT))
        })
    memory_summary = pd.DataFrame(summary_rows).sort_values("calibration_mae_memory_points").reset_index(drop=True)
    show_table(memory_summary.drop(columns=["model", "pipeline_label"]).round(3), "memory_forecasts_balanced_callees")
    chosen = memory_summary.iloc[0]
    chosen_rows = memory_rows[(memory_rows.model == chosen.model) & (memory_rows.pipeline_label == chosen.pipeline_label)]
    memory_plan = {name: rows.set_index("t_idx").predicted + chosen.margin_memory_points / 100
                   for name, rows in chosen_rows[chosen_rows.split == "test"].groupby("msname")}
    tell(
        f"The memory forecast with the lowest calibration error is {chosen.source}. It sizes the memory of added containers.",
        f"On test minutes it is off by {chosen.test_mae_memory_points:.2f} memory points on average.",
        f"In {100 * chosen.test_minutes_at_memory_risk:.1f}% of callee minutes the planned memory is above {100 * MEMORY_LIMIT:.0f}%.",
        "Memory does not follow traffic, so adding containers does not lower it. These minutes need a larger memory limit, not more routing."
    )
else:
    print("No memory forecasts were found. Added containers use the training median memory of the callee.")


def memory_for_new_container(name, minute):
    series = memory_plan.get(name)
    if series is not None and minute in series.index and np.isfinite(series[minute]):
        return float(series[minute])
    return memory_per_container.get(name, np.nan)

# COMMAND ----------

FLAGGED_CALLEE_MINUTES = set()
classifier_name = None
if EDGE_PREDICTIONS is not None:
    classifier_horizon = int(EDGE_PREDICTIONS.horizon.max()) if "horizon" in EDGE_PREDICTIONS else None
    if classifier_horizon != FORECAST_HORIZON or "present" not in EDGE_PREDICTIONS:
        print(f"The edge predictions do not fit this run (classifier horizon {classifier_horizon}, forecast horizon {FORECAST_HORIZON}, "
              "or an older export that only scored edges present in the minute). The classifier gate is not used.")
        EDGE_PREDICTIONS = None
if EDGE_PREDICTIONS is not None:
    validation_scores = EDGE_PREDICTIONS.groupby("model").objective_value.first()
    classifier_name = validation_scores.idxmax() if EDGE_CLASSIFIER_MODEL == "best" else EDGE_CLASSIFIER_MODEL
    print(f"edge classifier: {classifier_name}, chosen on validation {EDGE_PREDICTIONS.objective.iloc[0]} = {validation_scores[classifier_name]:.3f}")
    test_edges = EDGE_PREDICTIONS[(EDGE_PREDICTIONS.split == "test") & (EDGE_PREDICTIONS.model == classifier_name)
                                  & EDGE_PREDICTIONS.callee.isin(balanced_callees)]
    connection_flags = test_edges.groupby(["caller", "callee", "t_idx"]).agg(
        probability=("probability", "max"), flagged=("predicted_overload", "max"), high_traffic=("label", "max")
    ).reset_index()
    callee_flags = connection_flags.groupby(["callee", "t_idx"]).flagged.max()
    FLAGGED_CALLEE_MINUTES = {(callee, int(minute)) for (callee, minute), flag in callee_flags.items() if flag == 1}
    print(f"flagged connection minutes: {int(connection_flags.flagged.sum()):,} of {len(connection_flags):,}, "
          f"flagged callee minutes: {len(FLAGGED_CALLEE_MINUTES):,}")
else:
    GATES = [gate for gate in GATES if gate != "classifier"]
    print("Only the capacity check and every minute gates are used.")

if FLAGGED_CALLEE_MINUTES:
    relevance_rows = []
    for multiplier in DEMAND_MULTIPLIERS:
        observed = minute_results[(minute_results.balancer == "Observed") & (minute_results.multiplier == multiplier)
                                  & (minute_results.source == "oracle")]
        observed = observed[[minute in set(evaluation_minutes.get(name, [])) for name, minute in zip(observed.msname, observed.t_idx)]]
        flagged = np.array([(name, int(minute)) in FLAGGED_CALLEE_MINUTES for name, minute in zip(observed.msname, observed.t_idx)])
        overloaded = (observed.overloaded_containers > 0).to_numpy()
        true_positive = int((flagged & overloaded).sum())
        relevance_rows.append({
            "multiplier": multiplier,
            "callee_minutes": len(observed),
            "overloaded_under_trace_routing": int(overloaded.sum()),
            "flagged": int(flagged.sum()),
            "flagged_and_overloaded": true_positive,
            "recall": true_positive / max(int(overloaded.sum()), 1),
            "precision": true_positive / max(int(flagged.sum()), 1)
        })
    classifier_relevance = pd.DataFrame(relevance_rows)
    show_table(classifier_relevance.round(3), "classifier_relevance")
    stressed_relevance = classifier_relevance.iloc[-1]
    tell(
        f"The classifier flags {int(stressed_relevance.flagged):,} of {int(stressed_relevance.callee_minutes):,} callee minutes.",
        f"At {stressed_relevance.multiplier:g} times the demand, {int(stressed_relevance.overloaded_under_trace_routing):,} callee minutes are overloaded with the trace routing.",
        f"The classifier catches {100 * stressed_relevance.recall:.0f}% of them, and {100 * stressed_relevance.precision:.0f}% of its flags are real overloads.",
        "The classifier predicts unusually high traffic, not capacity overload. The capacity check after it decides whether action is needed."
    )

# COMMAND ----------

CONFIGURATIONS = {"Observed routing": (None, "never"), "Round Robin + reactive scale-out": ("Round Robin", "always")}
for balancer in INTELLIGENT_BALANCERS:
    for gate in GATES:
        CONFIGURATIONS[f"{balancer} + {gate}"] = (balancer, gate)


def known_split(name, minute, container_ids):
    """Share of the callee's calls each container received FORECAST_HORIZON minutes earlier. New containers get 0."""
    rows = traffic_containers.get(name, {}).get(minute - FORECAST_HORIZON)
    if rows is None or rows.incoming_mcr.sum() <= 0:
        return np.full(len(container_ids), 1.0 / len(container_ids))
    shares = dict(zip(rows.msinstanceid, rows.incoming_mcr / rows.incoming_mcr.sum()))
    split = np.array([shares.get(container_id, 0.0) for container_id in container_ids])
    return split / split.sum() if split.sum() > 0 else np.full(len(container_ids), 1.0 / len(container_ids))


def simulate_allocation(name, source, multiplier, configuration):
    """Replay every test minute in order. Only the evaluation minutes are scored, so the controller and the routing state never skip a minute."""
    balancer, gate = CONFIGURATIONS[configuration]
    model = capacity_models[name]
    controller = ScaleOutController()
    previous_by_container = {}
    previous_extra = 0
    last_planned = None
    scored = set(evaluation_minutes[name])
    minute_rows = []
    for minute in test_minutes:
        rows = traffic_containers.get(name, {}).get(minute)
        if rows is None or rows.empty:
            continue
        planned = planned_calls(source, name, minute, multiplier)
        if not np.isfinite(planned):
            if last_planned is None:
                continue
            planned = last_planned
        last_planned = planned
        container_ids = list(rows.msinstanceid)
        observed_loads = rows.incoming_mcr.to_numpy(dtype=float) * multiplier
        actual = observed_loads.sum()
        baselines = np.array([container_baseline(model, container_id) for container_id in container_ids])
        capacities = np.array([safe_capacity(model, value) for value in baselines])
        trace_split = observed_loads / actual if actual > 0 else np.full(len(container_ids), 1.0 / len(container_ids))

        if gate in ("always", "every minute"):
            act = True
        elif gate == "never":
            act = False
        else:
            flagged = gate == "capacity check" or (name, minute) in FLAGGED_CALLEE_MINUTES
            projected_peak = float((baselines + model["cpu_per_call"] * planned * known_split(name, minute, container_ids)).max())
            act = flagged and projected_peak > TARGET_UTILISATION

        required = required_extra_containers(model, planned, capacities, len(container_ids)) if act else 0
        extra = controller.update(required) if gate != "never" else 0
        if extra:
            container_ids = container_ids + [f"added_{number}" for number in range(extra)]
            baselines = np.append(baselines, np.full(extra, model["median_baseline"]))
            capacities = np.append(capacities, np.full(extra, model["new_container_capacity"]))
            observed_loads = np.append(observed_loads, np.zeros(extra))
            trace_split = np.append(trace_split, np.zeros(extra))

        previous = np.array([previous_by_container.get(container_id, 0.0) for container_id in container_ids])
        previous = previous / previous.sum() if previous.sum() > 0 else None
        if act or extra:
            fractions = split_traffic(balancer, baselines, capacities, planned, model["cpu_per_call"], previous, observed_loads)
        else:
            fractions = trace_split

        loads = fractions * actual
        cpu = baselines + model["cpu_per_call"] * loads
        overloaded = cpu > TARGET_UTILISATION
        limit_calls = np.maximum((TARGET_UTILISATION - baselines) / model["cpu_per_call"], 0.0)
        if minute in scored:
            minute_rows.append({
                "msname": name,
                "t_idx": minute,
                "containers": len(container_ids),
                "extra_containers": extra,
                "added_memory_share": extra * memory_for_new_container(name, minute),
                "acted": int(act),
                "scale_event": int(extra != previous_extra),
                "actual_demand": actual,
                "planned_demand": planned,
                "peak_cpu": cpu.max(),
                "overloaded_containers": int(overloaded.sum()),
                "excess_calls": float(np.maximum(loads - limit_calls, 0.0).sum()),
                "jain_fairness": (cpu.sum() ** 2) / (len(cpu) * (cpu ** 2).sum()) if (cpu ** 2).sum() > 0 else 1.0,
                "routing_change": 0.5 * np.abs(fractions - previous).sum() if previous is not None else 0.0
            })
        previous_by_container = dict(zip(container_ids, fractions))
        previous_extra = extra
    return minute_rows


allocation_rows = []
for multiplier in DEMAND_MULTIPLIERS:
    for configuration in CONFIGURATIONS:
        sources = ["Persistence (calls)"] if configuration in ("Observed routing", "Round Robin + reactive scale-out") else SOURCE_LABELS
        for source in sources:
            for name in gnn_callees:
                for row in simulate_allocation(name, source, multiplier, configuration):
                    row.update({"source": source, "multiplier": multiplier, "configuration": configuration})
                    allocation_rows.append(row)
    print(f"finished demand x {multiplier}")
allocation_minutes = pd.DataFrame(allocation_rows)
allocation_minutes.to_csv(os.path.join(OUTPUT_DIR, "allocation_minutes.csv"), index=False)
print(f"simulated callee minutes: {len(allocation_minutes):,}")

# COMMAND ----------

def summarise_allocation(rows):
    per_minute = rows.groupby("t_idx")
    return pd.Series({
        "mean_excess_calls_per_second": per_minute.excess_calls.sum().mean(),
        "overloaded_container_minutes_percent": 100 * rows.overloaded_containers.sum() / rows.containers.sum(),
        "added_container_minutes": rows.extra_containers.sum(),
        "added_memory_share": rows.added_memory_share.sum(),
        "jain_fairness": rows.jain_fairness.mean(),
        "routing_change": rows.routing_change.mean(),
        "scale_events": rows.scale_event.sum(),
        "acted_percent": 100 * rows.acted.mean()
    })


allocation_summary = (
    allocation_minutes.groupby(["multiplier", "configuration", "source"]).apply(summarise_allocation).reset_index()
)
show_table(allocation_summary.round(3), "allocation_summary")

stress = DEMAND_MULTIPLIERS[-1]
main_configuration = "Linear Programming + classifier" if "Linear Programming + classifier" in CONFIGURATIONS else "Linear Programming + capacity check"
stressed = allocation_summary[allocation_summary.multiplier == stress]
main = stressed[stressed.configuration == main_configuration].set_index("source")
conventional = stressed[stressed.configuration == "Round Robin + reactive scale-out"].iloc[0]
observed_reference = stressed[stressed.configuration == "Observed routing"].iloc[0]
tell(
    f"At {stress:g} times the demand, the trace routing leaves {calls(observed_reference.mean_excess_calls_per_second)} excess calls per second.",
    f"Conventional Round Robin with reactive scale-out leaves {calls(conventional.mean_excess_calls_per_second)} "
    f"and uses {conventional.added_container_minutes:,.0f} extra container minutes.",
    f"{main_configuration} with perfect demand information (the oracle) leaves {calls(main.loc['Oracle (calls)', 'mean_excess_calls_per_second'])}.",
    "No forecast does better here, so the remaining excess comes from the gate and the routing, not from forecast error."
    if main.loc["Oracle (calls)", "mean_excess_calls_per_second"] <= main.mean_excess_calls_per_second.min() + 1e-9
    else "Some forecasts do slightly better than the oracle here. With a gate, the outcome also depends on which minutes trigger action, not only on forecast error.",
    f"With persistence it leaves {calls(main.loc['Persistence (CPU)', 'mean_excess_calls_per_second'])}." if "Persistence (CPU)" in main.index else "",
    f"The GNN source chosen on calibration minutes, {best_gnn}, leaves {calls(main.loc[best_gnn, 'mean_excess_calls_per_second'])} "
    f"and uses {main.loc[best_gnn, 'added_container_minutes']:,.0f} extra container minutes." if best_gnn else ""
)

# COMMAND ----------

plot_configurations = [name for name in CONFIGURATIONS if name not in ("Observed routing", "Round Robin + reactive scale-out")]
fig, axes = plt.subplots(len(plot_configurations), 1, figsize=(13, 2.9 * len(plot_configurations) + 1.5), sharex=True, squeeze=False)
for panel, (ax, configuration) in enumerate(zip(axes[:, 0], plot_configurations)):
    rows = stressed[stressed.configuration == configuration].set_index("source").reindex(SOURCE_LABELS)
    colors = [TEXT_MUTED if "Oracle" in label or "Persistence" in label else SERIES_BLUE for label in SOURCE_LABELS]
    ax.bar(range(len(SOURCE_LABELS)), rows.mean_excess_calls_per_second, color=colors, width=0.7)
    ax.axhline(conventional.mean_excess_calls_per_second, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.3, label="Round Robin + reactive scale-out")
    ax.axhline(observed_reference.mean_excess_calls_per_second, color=TEXT_PRIMARY, linestyle="dotted", linewidth=1.3, label="Observed routing")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_title(configuration, loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(THOUSANDS)
    if panel == 0:
        ax.legend(loc="upper right", ncol=2, fontsize=9)
axes[-1, 0].set_xticks(range(len(SOURCE_LABELS)))
axes[-1, 0].set_xticklabels(SOURCE_LABELS, rotation=35, ha="right")
finish_figure(fig, f"Excess Calls by Demand Source at Demand x {stress:g}", "Demand source",
              "Calls per second above safe capacity (symmetric log scale)", "excess_by_demand_source.png")

classifier_configuration = "Linear Programming + classifier"
capacity_configuration = "Linear Programming + capacity check"
if classifier_configuration in CONFIGURATIONS:
    with_classifier = stressed[stressed.configuration == classifier_configuration].set_index("source")
    without_classifier = stressed[stressed.configuration == capacity_configuration].set_index("source")
    reference_source = best_gnn or "Persistence (calls)"
    tell(
        f"Grey bars are the oracle and persistence. Blue bars are GNN forecasts. The dashed line is conventional balancing.",
        f"For {reference_source}, the classifier gate acts in {with_classifier.loc[reference_source, 'acted_percent']:.1f}% of callee minutes "
        f"and the capacity check alone acts in {without_classifier.loc[reference_source, 'acted_percent']:.1f}%.",
        f"Excess calls are {calls(with_classifier.loc[reference_source, 'mean_excess_calls_per_second'])} with the classifier and "
        f"{calls(without_classifier.loc[reference_source, 'mean_excess_calls_per_second'])} without it.",
        "If the two are close, the classifier saves interventions without losing protection. If the classifier is worse, it misses real overloads."
    )

# COMMAND ----------

labels = list(main.index)[::-1]
colors = [TEXT_MUTED if "Oracle" in label or "Persistence" in label else SERIES_BLUE for label in labels]
fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(labels) + 2.2), sharey=True)
axes[0].barh(labels, main.loc[labels, "mean_excess_calls_per_second"], color=colors, height=0.65)
axes[0].axvline(conventional.mean_excess_calls_per_second, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.3, label="Round Robin + reactive scale-out")
axes[0].set_xscale("symlog", linthresh=1)
axes[0].set_xlim(left=0)
axes[0].set_xlabel("Calls per second above safe capacity (symmetric log scale)")
axes[0].set_ylabel("Demand source")
axes[0].set_title("Excess Calls", fontsize=12)
axes[0].legend(loc="lower right")
axes[1].barh(labels, main.loc[labels, "added_container_minutes"], color=colors, height=0.65)
axes[1].axvline(conventional.added_container_minutes, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.3)
axes[1].set_xlabel("Added container minutes")
axes[1].set_title("Container Cost", fontsize=12)
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(THOUSANDS)
finish_figure(fig, f"Excess Calls and Container Cost, {main_configuration}", None, None, "excess_and_cost.png")

dominated_by_persistence = [
    label for label in gnn_labels if "Persistence (CPU)" in main.index
    and main.loc[label, "mean_excess_calls_per_second"] <= main.loc["Persistence (CPU)", "mean_excess_calls_per_second"]
    and main.loc[label, "added_container_minutes"] <= main.loc["Persistence (CPU)", "added_container_minutes"]
]
tell(
    "Short bars on both sides are best: few excess calls and few extra containers.",
    f"GNN sources that beat persistence on both risk and cost: {callee_list(dominated_by_persistence)}." if dominated_by_persistence
    else "No GNN source beats persistence on both risk and cost at the same time.",
    "A source with a long cost bar buys safety with more containers. A source with a long excess bar saves containers but lets overload through."
)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(main) + 2.2), sharey=True)
labels = list(main.index)[::-1]
colors = [TEXT_MUTED if "Oracle" in label or "Persistence" in label else SERIES_BLUE for label in labels]
axes[0].barh(labels, main.loc[labels, "jain_fairness"], color=colors, height=0.65)
axes[0].axvline(conventional.jain_fairness, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.3, label="Round Robin + reactive scale-out")
axes[0].set_xlim(min(0.9, float(main.jain_fairness.min()) - 0.01), 1.0)
axes[0].set_xlabel("Mean Jain fairness index")
axes[0].set_ylabel("Demand source")
axes[0].set_title("Fairness", fontsize=12)
axes[0].legend(loc="lower left")
axes[1].barh(labels, main.loc[labels, "scale_events"], color=colors, height=0.65)
axes[1].axvline(conventional.scale_events, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.3)
axes[1].set_xlabel("Scale-out changes in the test window")
axes[1].set_title("Scaling Changes", fontsize=12)
for ax in axes:
    ax.grid(axis="y", visible=False)
finish_figure(fig, f"Fairness and Scaling Changes, {main_configuration}", None, None, "fairness_and_scaling_changes.png")

steadiest = main.scale_events.idxmin()
tell(
    f"Fairness stays between {main.jain_fairness.min():.3f} and {main.jain_fairness.max():.3f} across demand sources, "
    f"against {conventional.jain_fairness:.3f} for conventional balancing.",
    f"{steadiest} changes the number of containers least often ({main.loc[steadiest, 'scale_events']:.0f} times). "
    f"Conventional balancing changes it {conventional.scale_events:.0f} times.",
    "Fewer scaling changes mean fewer container starts and a calmer system."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.1 Are the differences real?
# MAGIC
# MAGIC Each test minute gives one value per demand source: the total excess calls over all callees, and the extra containers in use.
# MAGIC Two sources are compared with the paired difference of these per minute values. Minutes are autocorrelated, so the standard error uses the
# MAGIC equal weighted cosine long run variance (Lazarus, Lewis, Stock and Watson 2018) with a t distribution. Holm's method adjusts the p values
# MAGIC for the number of comparisons. Each GNN source is compared with persistence on CPU (same conversion, same margin rule) and with conventional balancing.
# MAGIC
# MAGIC The **primary family** is fixed in advance: excess calls, at the highest demand multiplier, against Persistence (CPU), in the main configuration.
# MAGIC `p_holm` adjusts within each family of comparisons and `p_holm_all` adjusts over every test in the table. Conclusions about one GNN being better
# MAGIC should come from the primary family. Comparisons with conventional balancing change the balancer, the gate and the demand source at once,
# MAGIC so they describe the whole system, not the forecast alone. Persistence (calls) in the main configuration changes only the demand source. A p value is left empty when every minute has the same difference.

# COMMAND ----------

def ewc_terms(n_minutes):
    return max(2, int(math.floor(0.4 * n_minutes ** (2 / 3))))


def ewc_long_run_variance(values, n_terms):
    centred = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centred)
    t = np.arange(1, n + 1)
    projections = [math.sqrt(2 / n) * np.dot(centred, np.cos(math.pi * j * (t - 0.5) / n)) for j in range(1, n_terms + 1)]
    return float(np.mean(np.square(projections)))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    order = valid[np.argsort(p[valid], kind="stable")]
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(order) - rank) * p[index]))
        adjusted[index] = running_max
    return adjusted


def per_minute(configuration, source, column, multiplier):
    rows = allocation_minutes[(allocation_minutes.configuration == configuration) & (allocation_minutes.source == source)
                              & (allocation_minutes.multiplier == multiplier)]
    return rows.groupby("t_idx")[column].sum()


def paired_difference(first, second):
    aligned = pd.concat([first, second], axis=1, join="inner").dropna()
    difference = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy()
    n = len(difference)
    terms = ewc_terms(n)
    standard_error = math.sqrt(max(ewc_long_run_variance(difference, terms), 0.0) / n) if n > 2 else float("nan")
    mean = float(difference.mean()) if n else float("nan")
    if not np.isfinite(standard_error) or standard_error == 0:
        return {"minutes": n, "mean_difference": mean, "low": mean, "high": mean, "p": float("nan")}
    quantile = float(stats.t.ppf(0.975, terms))
    return {"minutes": n, "mean_difference": mean, "low": mean - quantile * standard_error, "high": mean + quantile * standard_error,
            "p": float(2 * stats.t.sf(abs(mean / standard_error), terms))}


test_rows = []
for multiplier in DEMAND_MULTIPLIERS:
    for label in gnn_labels:
        for reference_name, reference_configuration, reference_source in (
            ("Persistence (CPU)", main_configuration, "Persistence (CPU)"),
            ("Persistence (calls)", main_configuration, "Persistence (calls)"),
            ("Conventional", "Round Robin + reactive scale-out", "Persistence (calls)")
        ):
            if reference_source not in SOURCE_LABELS:
                continue
            for metric, column in (("excess calls per second", "excess_calls"), ("extra containers", "extra_containers")):
                result = paired_difference(per_minute(main_configuration, label, column, multiplier),
                                           per_minute(reference_configuration, reference_source, column, multiplier))
                test_rows.append({"multiplier": multiplier, "source": label, "compared_with": reference_name, "metric": metric, **result})
paired_tests = pd.DataFrame(test_rows)
if not paired_tests.empty:
    paired_tests["p_holm"] = np.nan
    for _, index in paired_tests.groupby(["multiplier", "compared_with", "metric"]).groups.items():
        paired_tests.loc[index, "p_holm"] = holm_adjust(paired_tests.loc[index, "p"].to_numpy())
    paired_tests["p_holm_all"] = holm_adjust(paired_tests.p.to_numpy())
    paired_tests["primary_family"] = ((paired_tests.multiplier == stress) & (paired_tests.compared_with == "Persistence (CPU)")
                                      & (paired_tests.metric == "excess calls per second"))
    show_table(paired_tests.round(4), "paired_tests")

    excess_versus_persistence = paired_tests[(paired_tests.multiplier == stress) & (paired_tests.compared_with == "Persistence (CPU)")
                                             & (paired_tests.metric == "excess calls per second")]
    significantly_better = excess_versus_persistence[(excess_versus_persistence.p_holm < 0.05) & (excess_versus_persistence.mean_difference < 0)]
    significantly_worse = excess_versus_persistence[(excess_versus_persistence.p_holm < 0.05) & (excess_versus_persistence.mean_difference > 0)]
    tell(
        "A negative difference means the GNN source leaves fewer excess calls or uses fewer containers than the one it is compared with.",
        f"At {stress:g} times the demand, {len(significantly_better)} of {len(excess_versus_persistence)} GNN sources leave significantly fewer excess calls than persistence after Holm adjustment."
        + (f" They are {callee_list(significantly_better.source)}." if len(significantly_better) else ""),
        f"{len(significantly_worse)} are significantly worse." if len(significantly_worse) else "None is significantly worse.",
        "A difference that is not significant means the test window is too short to tell the two apart, not that they are equal."
    )
else:
    print("No GNN forecasts were found, so there is nothing to test.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Conclusion
# MAGIC
# MAGIC The cell below writes the story of the whole experiment from the results above.

# COMMAND ----------

normal = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[0])]
stressed = summary[(summary.source == "oracle") & (summary.multiplier == DEMAND_MULTIPLIERS[-1])]
persistence_stressed = summary[(summary.source == "persistence") & (summary.multiplier == DEMAND_MULTIPLIERS[-1])]
smart_balancers = ["Weighted Routing", "Linear Programming"]
tell(
    f"1. We studied {len(balanced_callees)} busy callees with {environment.containers.sum():,} containers and {len(connections):,} connections.",
    f"2. A simple model tells how many calls each container can safely take. It was right about {100 * capacity_validation.coverage_with_margin.median():.0f}% of the time on new minutes.",
    f"3. At normal demand, the trace routing overloads some containers ({calls(excess(normal, 'Observed'))} excess calls per second).",
    "   Sharing calls more evenly removes this. The containers were enough. The sharing was the problem."
    if min(excess(normal, name) for name in BALANCERS) < 1 else "   Better sharing reduces this, but some overload remains.",
    f"4. At {DEMAND_MULTIPLIERS[-1]:g} times the demand, the best routing leaves {calls(min(excess(stressed, name) for name in BALANCERS))} excess calls per second.",
    "   Sharing alone is not enough here." if min(excess(stressed, name) for name in BALANCERS) >= 1 else "   Sharing alone almost handles it.",
    f"5. Smart routing plus scale-out leaves {calls(min(excess(stressed, name, True) for name in smart_balancers))}.",
    f"   It costs {policy_value(stressed, 'Linear Programming', True, 'added_container_minutes'):,.0f} extra container minutes.",
    f"6. Linear Programming gives the fairest and steadiest result. Weighted Routing is almost as good and needs no solver.",
    f"7. Knowing the future matters most at the start of bursts. Last minute's demand leaves "
    f"{calls(policy_value(persistence_stressed, 'Linear Programming', True, 'mean_excess_calls_per_second'))} excess calls per second, the oracle leaves "
    f"{calls(excess(stressed, 'Linear Programming', True))}.",
    f"8. With forecasts, {main_configuration} and the best GNN source ({best_gnn}) leave "
    f"{calls(main.loc[best_gnn, 'mean_excess_calls_per_second'])} excess calls per second at {stress:g} times the demand, "
    f"using {main.loc[best_gnn, 'added_container_minutes']:,.0f} extra container minutes." if best_gnn
    else "8. No GNN forecasts were found. Run the forecasting notebooks to add them.",
    f"   Conventional Round Robin with reactive scale-out leaves {calls(conventional.mean_excess_calls_per_second)}.",
    f"   Balancing every minute with Linear Programming and {best_gnn} leaves "
    f"{calls(allocation_summary[allocation_summary.multiplier == stress].set_index(['configuration', 'source']).loc[('Linear Programming + every minute', best_gnn), 'mean_excess_calls_per_second'])} "
    f"with {allocation_summary[allocation_summary.multiplier == stress].set_index(['configuration', 'source']).loc[('Linear Programming + every minute', best_gnn), 'added_container_minutes']:,.0f} extra container minutes."
    if best_gnn and ("Linear Programming + every minute", best_gnn) in allocation_summary[allocation_summary.multiplier == stress].set_index(["configuration", "source"]).index else "",
    f"9. The classifier flags {len(FLAGGED_CALLEE_MINUTES):,} callee minutes. It decides where to look. The capacity check decides whether to act."
    if FLAGGED_CALLEE_MINUTES else "9. No classifier predictions were used."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Methods and references
# MAGIC
# MAGIC | Method | Where | Reference | Role in this notebook |
# MAGIC |---|---|---|---|
# MAGIC | Ratio estimator for connection shares | Section 2 | Horvitz, D. G. and Thompson, D. J. (1952). *JASA*, 47(260), 663 to 685. | The call graph is a 0.5% sample. Shares need no scaling, absolute counts would. |
# MAGIC | Theil Sen slope | Section 3 | Theil, H. (1950). *Indagationes Mathematicae*, 12. Sen, P. K. (1968). *JASA*, 63(324), 1379 to 1389. | Robust CPU per call, not pulled by outlier minutes. |
# MAGIC | Quantile safety margin | Section 3 | Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33 to 50. | Capacity covers 90% of observed CPU, not only the average. |
# MAGIC | Greedy least loaded assignment | Section 4 | Graham, R. L. (1969). Bounds on multiprocessing timing anomalies. *SIAM Journal on Applied Mathematics*, 17(2), 416 to 429. | Each chunk goes to the least loaded container. |
# MAGIC | Min-max linear program | Section 4 | Huangfu, Q. and Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10, 119 to 142 (HiGHS). | Lowest possible peak CPU for the planned demand, with a penalty on routing changes. |
# MAGIC | Capacity based scale-out | Section 5 | Luo, S. et al. (2023). Erms: efficient resource management for shared microservices with SLA guarantees. *ASPLOS 2023*. | Containers are added from demand divided by per container capacity. |
# MAGIC | Scale-in delay | Section 5 | Kubernetes documentation, Horizontal Pod Autoscaler, stabilisation window. | Avoids adding and removing containers every minute. |
# MAGIC | Jain fairness index | Section 7 | Jain, R., Chiu, D. M. and Hawe, W. R. (1984). *A quantitative measure of fairness and discrimination for resource allocation in shared computer systems*. DEC Technical Report TR-301. | Single number for how evenly CPU is spread. |
# MAGIC | Oracle reference | Section 6 | Herbst, N. et al. (2018). Quantifying cloud performance and dependability. *ACM TOMPECS*, 3(4). | Perfect information sets the best result a forecast can reach. |
# MAGIC | Split conformal safety margin | Section 8 | Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. Lei, J. et al. (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094 to 1111. | Every forecast, the GNNs and persistence alike, gets a margin in calls from its own errors on held out calibration minutes. Coverage is exact only if calibration and test errors are exchangeable, so the test coverage is reported. |
# MAGIC | Forecast to demand conversion | Section 8 | Inversion of the capacity model in section 3 | A CPU forecast becomes planned calls. Oracle (CPU) measures the error of this step alone. |
# MAGIC | Paired tests with a long run variance | Section 8.1 | Lazarus, E., Lewis, D. J., Stock, J. H. and Watson, M. W. (2018). HAR inference: recommendations for practice. *JBES*, 36(4), 541 to 559. | Minutes are autocorrelated, so a plain t test would be too confident. |
# MAGIC | Holm adjustment | Section 8.1 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65 to 70. | Many sources are compared at once, so p values are adjusted. |
# MAGIC
# MAGIC **Limits.** CPU under a new routing is modelled, not measured, because the trace cannot be replayed on real containers.
# MAGIC The trace has only average response time, so overload is measured in CPU and excess calls, not in tail latency.
# MAGIC Connection shares are fixed over the test window because the call graph sample is too sparse per minute.
# MAGIC The GNN models forecast CPU, not calls. Calls are derived from CPU with the capacity model, so its error adds to the forecast error.
# MAGIC The edge classifier predicts unusually high traffic on a connection, not capacity overload, so a capacity check follows it.
# MAGIC The paired tests use the first training seed only, so they do not include training randomness. The seed spread of the forecast error is shown in section 8.
# MAGIC The simulation assumes the platform knows which containers run in a minute. Only the first training seed of each model drives the allocation.