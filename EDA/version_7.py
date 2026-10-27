# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis: Alibaba Microservices Trace v2021
# MAGIC
# MAGIC This notebook explores the three tables used in the forecasting work: **MSResource** (CPU and memory per container),
# MAGIC **MSRTQps** (call rate and response time per container) and **MSCallGraph** (sampled calls between microservices).
# MAGIC
# MAGIC Traffic is analysed at two levels.
# MAGIC
# MAGIC * **Instance level.** One container (`msinstanceid`) in one minute.
# MAGIC * **Microservice level.** All containers of one microservice (`msname`) in one minute, summed or averaged as explained in each section.
# MAGIC
# MAGIC It follows the structure of the earlier EDA notebook and corrects its statistical issues. Section 10 lists every
# MAGIC statistical method used, its reference and how it changes the conclusions of this EDA.
# MAGIC
# MAGIC | Section | Content |
# MAGIC |---|---|
# MAGIC | 1 | Settings and chart helper |
# MAGIC | 2 | Load the tables and check time coverage |
# MAGIC | 3 | Data quality |
# MAGIC | 4 | MSResource: CPU and memory |
# MAGIC | 5 | MSRTQps: instance level traffic |
# MAGIC | 6 | MSRTQps: microservice level traffic |
# MAGIC | 7 | MSCallGraph: calls and dependencies |
# MAGIC | 8 | Relationships across tables |
# MAGIC | 9 | Experiments for graph based forecasting |
# MAGIC | 10 | Statistical methods, references and their impact |
# MAGIC | 11 | Summary of findings |

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
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd
from pyspark.sql import functions as F
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore", category=FutureWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings and chart helper
# MAGIC
# MAGIC The paths point to the corrected MSRTQps output and to the ETL tables used by the feature engineering notebooks. `WORK_PATH` stores intermediate tables.

# COMMAND ----------

RESOURCE_PATH = "s3://research-s20426/etl_output_root/resource"
MSRTQPS_PATH = "s3://research-s20426/msrtqps_parquet/parquet_final/"
CALLGRAPH_PATH = "s3://research-s20426/etl_output_root/callgraph_1"
WORK_PATH = "s3://research-s20426/eda_work"
OUTPUT_DIR = "./eda_outputs"

FIRST_MINUTE = 0
LAST_MINUTE = 719
CALL_GRAPH_SAMPLING_RATE = 0.005
SAMPLE_ROWS = 500_000
MIN_MINUTES_PER_SERVICE = 30

MCR_COLUMNS = ["providerRPC_MCR", "HTTP_MCR", "consumerMQ_MCR", "consumerRPC_MCR"]
RT_COLUMNS = ["providerRPC_RT", "HTTP_RT", "consumerMQ_RT", "consumerRPC_RT"]
INCOMING_MCR_COLUMNS = ["providerRPC_MCR", "HTTP_MCR", "consumerMQ_MCR"]
OUTGOING_MCR_COLUMNS = ["consumerRPC_MCR"]
MISSING_SERVICE_NAMES = ["(?)", "NAN", "nan", "NaN", "", "UNKNOWN", "UNAVAILABLE"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# COMMAND ----------

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE_COLOR = "#fcfcfb"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#ef8a89", "#c23535"]
)
CORRELATION_BAR_COLORS = LinearSegmentedColormap.from_list(
    "correlation_bar_colors", ["#1c5cab", "#6da7ec", "#b9b8ae", "#ef8a89", "#c23535"]
)
CATEGORY_COLORS = [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA, "#eda100", "#6250d6"]

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

THOUSANDS = FuncFormatter(lambda value, position: f"{value:,.0f}")


def plain_number(value, position=None):
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def show_table(frame, name):
    frame.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
    if frame.empty:
        print("no rows")
        return
    try:
        display(frame)
    except NameError:
        print(frame.to_string(index=False))


def format_log_axis(axis, limits):
    low, high = limits
    decades = math.log10(high / low) if low > 0 else 99
    if decades < 1:
        axis.set_major_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
    elif decades < 3:
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(plain_number))
    axis.set_minor_formatter(NullFormatter())


def format_axes(ax, count_axis=None):
    if count_axis == "y":
        ax.yaxis.set_major_formatter(THOUSANDS)
    if count_axis == "x":
        ax.xaxis.set_major_formatter(THOUSANDS)
    if ax.get_xscale() == "log":
        format_log_axis(ax.xaxis, ax.get_xlim())
    if ax.get_yscale() == "log":
        format_log_axis(ax.yaxis, ax.get_ylim())


def save_and_show(fig, file_name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def finish_chart(fig, ax, title, x_label, y_label, file_name, count_axis=None):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    format_axes(ax, count_axis)
    save_and_show(fig, file_name)


def finish_figure(fig, title, x_label, y_label, file_name):
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    if x_label:
        fig.supxlabel(x_label, fontsize=11, color=TEXT_SECONDARY)
    if y_label:
        fig.supylabel(y_label, fontsize=11, color=TEXT_SECONDARY)
    save_and_show(fig, file_name)


PIE_COLORS = [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA, "#eda100", "#6250d6", "#c23535", "#86b6ef", TEXT_MUTED]


def draw_pie(ax, labels, values, title):
    values = np.asarray(values, dtype=float)
    shares = 100 * values / values.sum()
    wedges, _, percent_texts = ax.pie(
        values, colors=PIE_COLORS[:len(values)], startangle=90, counterclock=False, pctdistance=0.78,
        autopct=lambda percent: f"{percent:.1f}%" if percent >= 4 else "",
        wedgeprops={"width": 0.45, "edgecolor": SURFACE_COLOR, "linewidth": 2}
    )
    for text in percent_texts:
        text.set_color("white")
        text.set_fontsize(10)
        text.set_fontweight("bold")
    ax.legend(wedges, [f"{label} ({share:.1f}%)" for label, share in zip(labels, shares)],
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(title)
    ax.set_aspect("equal")


def draw_boxes(ax, labels, groups, color=SERIES_BLUE):
    positions = np.arange(1, len(groups) + 1)
    ax.boxplot(
        groups, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
        boxprops={"facecolor": to_rgba(color, 0.3), "edgecolor": color, "linewidth": 1.2},
        medianprops={"color": "#0d366b", "linewidth": 2},
        whiskerprops={"color": TEXT_MUTED, "linewidth": 1.2},
        capprops={"color": TEXT_MUTED, "linewidth": 1.2}
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def tell(*sentences):
    print("\nInterpretation")
    for sentence in sentences:
        if sentence:
            print("  " + sentence)


def log_bins(values, bins_per_decade=10):
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    low = math.floor(math.log10(positive.min()) * bins_per_decade) / bins_per_decade
    high = math.ceil(math.log10(positive.max()) * bins_per_decade) / bins_per_decade
    return np.logspace(low, high, max(1, round((high - low) * bins_per_decade)) + 1)


def draw_distribution(ax, edges, counts, color=SERIES_BLUE, label=None):
    heights = np.append(np.asarray(counts, dtype=float), counts[-1])
    ax.fill_between(edges, heights, step="post", color=color, alpha=0.22, linewidth=0)
    ax.step(edges, heights, where="post", color=color, linewidth=1.6, label=label)
    ax.set_ylim(bottom=0)


def mark_median(ax, median_value):
    ax.axvline(median_value, color=TEXT_PRIMARY, linestyle="dashed", linewidth=1.3, label="Median")


def plot_correlation_histogram(values, title, file_name):
    edges = np.linspace(-1, 1, 41)
    counts, _ = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(centers, counts, width=edges[1] - edges[0], color=CORRELATION_BAR_COLORS((centers + 1) / 2))
    mark_median(ax, np.median(values))
    ax.set_xlim(-1, 1)
    ax.legend(loc="upper left")
    finish_chart(fig, ax, title, "Spearman correlation", "Microservices", file_name)


def draw_violins(ax, labels, groups):
    positions = np.arange(1, len(groups) + 1)
    trimmed = []
    for values in groups:
        values = np.asarray(values, dtype=float)
        low, high = np.percentile(values, [0.5, 99.5])
        trimmed.append(values[(values >= low) & (values <= high)])
    parts = ax.violinplot(trimmed, positions=positions, widths=0.82, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9ec5f4")
        body.set_edgecolor(SERIES_BLUE)
        body.set_linewidth(1)
        body.set_alpha(0.9)
    for position, values in zip(positions, trimmed):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.vlines(position, q1, q3, color="#0d366b", linewidth=6, zorder=3)
        ax.scatter([position], [median], color="white", edgecolor="#0d366b", s=28, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def draw_box_plot(ax, labels, percentile_rows):
    boxes = []
    for label, (p05, p25, p50, p75, p95) in zip(labels, percentile_rows):
        boxes.append({"label": label, "whislo": p05, "q1": p25, "med": p50, "q3": p75, "whishi": p95, "fliers": []})
    ax.bxp(
        boxes, showfliers=False, patch_artist=True, widths=0.55,
        boxprops={"facecolor": "#9ec5f4", "edgecolor": SERIES_BLUE, "linewidth": 1.2},
        medianprops={"color": "#0d366b", "linewidth": 2},
        whiskerprops={"color": TEXT_MUTED, "linewidth": 1.2},
        capprops={"color": TEXT_MUTED, "linewidth": 1.2}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the tables and check time coverage
# MAGIC
# MAGIC Every table gets the same minute index, `t_idx = floor(timestamp / 60000)`. Minute 720 is dropped because it only holds the last partial interval.

# COMMAND ----------

def add_minute_index(table):
    for column_name, data_type in table.dtypes:
        if data_type in ("double", "float"):
            table = table.withColumn(column_name, F.when(~F.isnan(F.col(column_name)), F.col(column_name)))
    table = table.withColumn("t_idx", F.floor(F.col("timestamp") / 60000).cast("int"))
    return table.filter(F.col("t_idx").between(FIRST_MINUTE, LAST_MINUTE))


resource = add_minute_index(spark.read.parquet(RESOURCE_PATH))
msrtqps = add_minute_index(spark.read.parquet(MSRTQPS_PATH))
callgraph = add_minute_index(spark.read.parquet(CALLGRAPH_PATH))

MCR_COLUMNS = [column for column in MCR_COLUMNS if column in msrtqps.columns]
RT_COLUMNS = [column for column in RT_COLUMNS if column in msrtqps.columns]
INCOMING_MCR_COLUMNS = [column for column in INCOMING_MCR_COLUMNS if column in msrtqps.columns]
OUTGOING_MCR_COLUMNS = [column for column in OUTGOING_MCR_COLUMNS if column in msrtqps.columns]

overview_rows = []
for table_name, table in (("MSResource", resource), ("MSRTQps", msrtqps), ("MSCallGraph", callgraph)):
    overview_rows.append({"table": table_name, "rows": table.count(), "columns": ", ".join(table.columns)})
overview = pd.DataFrame(overview_rows)
show_table(overview, "table_overview")
print("MSRTQps traffic columns:", MCR_COLUMNS + RT_COLUMNS)

# COMMAND ----------

def rows_per_minute(table):
    return table.groupBy("t_idx").count().orderBy("t_idx").toPandas()


resource_minutes = rows_per_minute(resource)
msrtqps_minutes = rows_per_minute(msrtqps)
callgraph_minutes = rows_per_minute(callgraph)

for table_name, minutes in (("MSResource", resource_minutes), ("MSRTQps", msrtqps_minutes), ("MSCallGraph", callgraph_minutes)):
    missing = sorted(set(range(FIRST_MINUTE, LAST_MINUTE + 1)) - set(minutes.t_idx))
    thin = minutes[minutes["count"] < 0.5 * minutes["count"].median()]
    print(f"{table_name}: {len(minutes)} of {LAST_MINUTE - FIRST_MINUTE + 1} minutes present, "
          f"{len(missing)} missing, {len(thin)} with less than half the median row count")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(11, 4.5))
table_minutes = (("MSResource", resource_minutes), ("MSRTQps", msrtqps_minutes), ("MSCallGraph", callgraph_minutes))
for (table_name, minutes), color, width in zip(table_minutes, CATEGORY_COLORS, (3.2, 2.0, 1.4)):
    relative_count = 100 * minutes["count"] / minutes["count"].median()
    ax.plot(minutes.t_idx, relative_count, color=color, linewidth=width, label=table_name)
ax.axhline(100, color=TEXT_MUTED, linewidth=1, linestyle="dotted")
ax.set_xlim(FIRST_MINUTE, LAST_MINUTE)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", ncol=3)
finish_chart(fig, ax, "Records per Minute", "Minute of trace", "Records relative to median (%)", "records_per_minute.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** MSResource and MSRTQps are complete in all 720 minutes. MSCallGraph is not. It falls to almost zero between minutes 220 and 330
# MAGIC and has 140 minutes with less than half the usual records. Edge features built from call graph counts are unreliable in that window.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data quality
# MAGIC
# MAGIC Four checks decide how later sections aggregate the data.
# MAGIC
# MAGIC 1.
# MAGIC 2.
# MAGIC per container and minute before any statistic. Otherwise containers with more records would count more.
# MAGIC 3.
# MAGIC 4.
# MAGIC If that is common, those RT columns are not real response times and are left out.

# COMMAND ----------

def missing_share(table, table_name):
    total = table.count()
    expressions = [F.sum(F.col(column).isNull().cast("int")).alias(column) for column in table.columns]
    counts = table.select(*expressions).first().asDict()
    return pd.DataFrame([{"table": table_name, "column": column, "missing_share": counts[column] / total} for column in table.columns])


missing_values = pd.concat([
    missing_share(resource, "MSResource"),
    missing_share(msrtqps, "MSRTQps"),
    missing_share(callgraph, "MSCallGraph")
], ignore_index=True)
show_table(missing_values[missing_values.missing_share > 0].round(4), "missing_values")

# COMMAND ----------

records_per_instance_minute = (
    resource.groupBy("msinstanceid", "t_idx").count()
    .groupBy("count").agg(F.count("*").alias("instance_minutes"))
    .orderBy("count").toPandas()
    .rename(columns={"count": "records_in_minute"})
)
print("MSResource records per container and minute:")
show_table(records_per_instance_minute, "records_per_instance_minute")

utilisation_ranges = resource.select(
    F.mean((F.col("cpu_utilization") > 1).cast("double")).alias("cpu_above_100_percent_share"),
    F.mean((F.col("memory_utilization") > 1).cast("double")).alias("memory_above_100_percent_share"),
    F.mean((F.col("memory_utilization") < 0).cast("double")).alias("memory_below_zero_share")
).first().asDict()
show_table(pd.DataFrame([utilisation_ranges]).round(6), "utilisation_ranges")

# COMMAND ----------

equal_value_rows = []
for paradigm in ("providerRPC", "consumerRPC", "HTTP", "consumerMQ"):
    rate_column, time_column = f"{paradigm}_MCR", f"{paradigm}_RT"
    if rate_column in msrtqps.columns and time_column in msrtqps.columns:
        both = msrtqps.filter(F.col(rate_column).isNotNull() & F.col(time_column).isNotNull())
        result = both.select(
            F.count("*").alias("rows_with_both"),
            F.mean((F.col(rate_column) == F.col(time_column)).cast("double")).alias("share_rt_equals_mcr")
        ).first().asDict()
        equal_value_rows.append({"paradigm": paradigm, **result})
rt_equals_mcr = pd.DataFrame(equal_value_rows)
show_table(rt_equals_mcr, "response_time_equals_call_rate")

SUSPECT_RT_COLUMNS = [f"{row.paradigm}_RT" for row in rt_equals_mcr.itertuples() if (row.share_rt_equals_mcr or 0) > 0.5]
RT_COLUMNS = [column for column in RT_COLUMNS if column not in SUSPECT_RT_COLUMNS]
print("RT columns left out because they mostly equal the call rate:", SUSPECT_RT_COLUMNS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MSResource: CPU and memory
# MAGIC
# MAGIC All statistics in this section use **instance minutes**: the mean of the records of one container in one minute. Utilisation is shown in percent.

# COMMAND ----------

instance_minute_resource = resource.groupBy("msname", "msinstanceid", "t_idx").agg(
    F.mean("cpu_utilization").alias("cpu_utilization"),
    F.mean("memory_utilization").alias("memory_utilization")
)
instance_minute_resource.write.mode("overwrite").parquet(f"{WORK_PATH}/instance_minute_resource")
instance_minute_resource = spark.read.parquet(f"{WORK_PATH}/instance_minute_resource")


def summary_statistics(table, column, scale=1.0):
    row = table.select(
        F.count(column).alias("count"),
        F.mean(column).alias("mean"),
        F.stddev(column).alias("standard_deviation"),
        F.percentile_approx(column, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 10000).alias("percentiles"),
        F.skewness(column).alias("skewness"),
        F.kurtosis(column).alias("excess_kurtosis")
    ).first()
    p05, p25, p50, p75, p95, p99 = [value * scale for value in row.percentiles]
    return {
        "column": column, "count": row["count"], "mean": row["mean"] * scale,
        "standard_deviation": row["standard_deviation"] * scale, "p5": p05, "p25": p25, "median": p50,
        "p75": p75, "p95": p95, "p99": p99, "iqr": p75 - p25,
        "skewness": row["skewness"], "excess_kurtosis": row["excess_kurtosis"]
    }


resource_summary = pd.DataFrame([
    summary_statistics(instance_minute_resource, "cpu_utilization", 100.0),
    summary_statistics(instance_minute_resource, "memory_utilization", 100.0)
])
print("Instance minute utilisation in percent:")
show_table(resource_summary.round(3), "resource_summary")

# COMMAND ----------

def spark_histogram(table, column, low, high, bins, scale=1.0):
    width = (high - low) / bins
    scaled = F.col(column) * scale
    present = table.filter(F.col(column).isNotNull())
    inside = present.filter((scaled >= low) & (scaled <= high))
    counts = (
        inside.withColumn("bucket", F.least(F.floor((scaled - low) / width).cast("int"), F.lit(bins - 1)))
        .groupBy("bucket").count().toPandas()
    )
    total, kept = present.count(), int(counts["count"].sum()) if len(counts) else 0
    print(f"{column}: {1 - kept / max(total, 1):.3%} of values lie outside {low:g} to {high:g} and are not drawn")
    full = pd.DataFrame({"bucket": range(bins)}).merge(counts, on="bucket", how="left").fillna(0)
    full["left_edge"] = low + full.bucket * width
    return full, width


def display_range(table, column, scale=1.0, lower_quantile=0.001, upper_quantile=0.999, step=5):
    low, high = table.select(F.percentile_approx(column, [lower_quantile, upper_quantile], 10000)).first()[0]
    low = math.floor(low * scale / step) * step
    high = math.ceil(high * scale / step) * step
    return low, max(high, low + step)


BIN_WIDTH_PERCENT = 0.5


def plot_utilisation_distribution(column, color, median_percent, title, x_label, file_name):
    low, high = display_range(instance_minute_resource, column, 100.0)
    bins = int(round((high - low) / BIN_WIDTH_PERCENT))
    histogram, width = spark_histogram(instance_minute_resource, column, low, high, bins, 100.0)
    edges = low + np.arange(bins + 1) * width
    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw_distribution(ax, edges, histogram["count"].to_numpy(), color=color)
    mark_median(ax, median_percent)
    ax.set_xlim(low, high)
    ax.legend(loc="upper right")
    finish_chart(fig, ax, title, x_label, "Instance minutes", file_name, count_axis="y")


plot_utilisation_distribution("cpu_utilization", SERIES_BLUE, resource_summary.loc[0, "median"],
                              "CPU Utilization Distribution", "CPU utilization (%)", "cpu_distribution.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** CPU is low and right skewed. The median is 15.9% and the 95th percentile is 43.4%. A small second mode near 55 to 65% marks a group of busy containers.
# MAGIC Most containers are over provisioned, which leaves room for resource optimisation.

# COMMAND ----------

plot_utilisation_distribution("memory_utilization", SERIES_AQUA, resource_summary.loc[1, "median"],
                              "Memory Utilization Distribution", "Memory utilization (%)", "memory_distribution.png")

# COMMAND ----------

LEVEL_EDGES = [0, 10, 30, 50, 70, 1000]
LEVEL_LABELS = ["under 10%", "10 to 30%", "30 to 50%", "50 to 70%", "over 70%"]


def level_counts(column):
    percent = F.col(column) * 100
    level = F.when(percent < 10, 0).when(percent < 30, 1).when(percent < 50, 2).when(percent < 70, 3).otherwise(4)
    counts = instance_minute_resource.filter(F.col(column).isNotNull()).groupBy(level.alias("level")).count().toPandas()
    return counts.set_index("level")["count"].reindex(range(len(LEVEL_LABELS)), fill_value=0)


cpu_levels = level_counts("cpu_utilization")
memory_levels = level_counts("memory_utilization")
fig, (cpu_axis, memory_axis) = plt.subplots(1, 2, figsize=(14, 5.5))
draw_pie(cpu_axis, LEVEL_LABELS, cpu_levels, "CPU")
draw_pie(memory_axis, LEVEL_LABELS, memory_levels, "Memory")
fig.suptitle("Instance Minutes by Utilization Level", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
save_and_show(fig, "utilization_levels.png")

tell(
    f"{100 * cpu_levels.iloc[:2].sum() / cpu_levels.sum():.0f}% of container minutes use less than 30% CPU.",
    f"Only {100 * cpu_levels.iloc[4] / cpu_levels.sum():.1f}% run above 70% CPU.",
    f"For memory, {100 * memory_levels.iloc[3:].sum() / memory_levels.sum():.0f}% of container minutes sit above 50%.",
    "CPU is mostly idle capacity. Memory is the tighter resource."
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Memory sits much higher than CPU, with a median of 66.8%. The sharp peaks come from services that hold a fixed memory level.
# MAGIC Memory is the tighter resource, but it changes little over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Microservice level table
# MAGIC
# MAGIC One row per microservice and minute. CPU and memory are the mean over containers.

# COMMAND ----------

service_minute_resource = instance_minute_resource.groupBy("msname", "t_idx").agg(
    F.count("*").alias("instances"),
    F.mean("cpu_utilization").alias("cpu_mean"),
    F.mean("memory_utilization").alias("memory_mean")
)

# COMMAND ----------

service_resource_pandas = service_minute_resource.toPandas()


def draw_percentile_bands(ax, column, light_color, dark_color, y_label):
    by_minute = service_resource_pandas.groupby("t_idx")[column].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack() * 100
    ax.fill_between(by_minute.index, by_minute[0.1], by_minute[0.9], color=light_color, alpha=0.35, linewidth=0, label="10th to 90th percentile")
    ax.fill_between(by_minute.index, by_minute[0.25], by_minute[0.75], color=light_color, alpha=0.75, linewidth=0, label="25th to 75th percentile")
    ax.plot(by_minute.index, by_minute[0.5], color=dark_color, linewidth=1.8, label="Median")
    ax.set_xlim(FIRST_MINUTE, LAST_MINUTE)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))


fig, (cpu_axis, memory_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
draw_percentile_bands(cpu_axis, "cpu_mean", "#86b6ef", "#0d366b", "CPU utilization (%)")
draw_percentile_bands(memory_axis, "memory_mean", "#8fd9bd", "#0b6b48", "Memory utilization (%)")
finish_figure(fig, "CPU and Memory Utilization Over Time", "Minute of trace", None, "utilization_over_time.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Median service CPU stays near 11 to 12% and drops to about 9% after minute 600, where the upper band also narrows.
# MAGIC Memory stays flat near 62% for all 12 hours. CPU has a mild drift and no strong cycle, while memory is almost constant.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 CPU and memory relationship
# MAGIC
# MAGIC Three correlations are compared.
# MAGIC
# MAGIC * **Pooled Pearson** over all instance minutes.
# MAGIC * **Pooled Spearman** on a random sample of instance minutes.
# MAGIC * **Within service Spearman**, computed separately for each microservice over its minutes and then summarised.
# MAGIC
# MAGIC Pooled values mix differences between services with changes inside a service. Only the within service value describes how CPU and memory move together for one microservice.

# COMMAND ----------

pooled_pearson = instance_minute_resource.select(F.corr("cpu_utilization", "memory_utilization")).first()[0]

total_instance_minutes = instance_minute_resource.count()
sample_fraction = min(1.0, SAMPLE_ROWS / max(total_instance_minutes, 1))
resource_sample = (
    instance_minute_resource.sample(fraction=sample_fraction, seed=7)
    .select("msname", "cpu_utilization", "memory_utilization").dropna().toPandas()
)
pooled_spearman = stats.spearmanr(resource_sample.cpu_utilization, resource_sample.memory_utilization).correlation


def within_service_spearman(frame, first_column, second_column):
    results = []
    for service_name, rows in frame.groupby("msname"):
        rows = rows[[first_column, second_column]].dropna()
        if len(rows) < MIN_MINUTES_PER_SERVICE or rows[first_column].nunique() < 2 or rows[second_column].nunique() < 2:
            continue
        results.append({"msname": service_name, "spearman": stats.spearmanr(rows[first_column], rows[second_column]).correlation})
    return pd.DataFrame(results)


cpu_memory_by_service = within_service_spearman(service_resource_pandas, "cpu_mean", "memory_mean")
print(f"pooled Pearson over instance minutes: {pooled_pearson:.3f}")
print(f"pooled Spearman on a sample of {len(resource_sample):,} instance minutes: {pooled_spearman:.3f}")
print(f"within service Spearman: median {cpu_memory_by_service.spearman.median():.3f}, "
      f"interquartile range {cpu_memory_by_service.spearman.quantile(0.25):.3f} to {cpu_memory_by_service.spearman.quantile(0.75):.3f}, "
      f"{len(cpu_memory_by_service)} microservices")

# COMMAND ----------

plot_correlation_histogram(cpu_memory_by_service.spearman, "CPU and Memory Correlation per Microservice",
                           "cpu_memory_correlation_per_service.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The pooled Pearson correlation is 0.281, but the median within service Spearman is -0.167 (interquartile range -0.437 to 0.126).
# MAGIC The pooled value comes from differences between services, not from changes inside a service. CPU and memory should be forecast as separate targets.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Spread per microservice
# MAGIC
# MAGIC The ten microservices with the most containers, drawn from up to 20,000 sampled instance minutes each. Boxes show the 25th to 75th percentile and the median.

# COMMAND ----------

largest_services = (
    instance_minute_resource.groupBy("msname").agg(F.countDistinct("msinstanceid").alias("containers"))
    .orderBy(F.desc("containers")).limit(10).toPandas()
)
ROWS_PER_SERVICE = 20_000
largest_rows = instance_minute_resource.filter(F.col("msname").isin(list(largest_services.msname)))
row_counts = dict(largest_rows.groupBy("msname").count().toPandas().values)
sample_fractions = {name: min(1.0, ROWS_PER_SERVICE / count) for name, count in row_counts.items()}
service_sample = (
    largest_rows.sampleBy("msname", sample_fractions, seed=7)
    .select("msname", "cpu_utilization", "memory_utilization").dropna().toPandas()
)
service_labels = [f"Service {rank}" for rank in range(1, len(largest_services) + 1)]
show_table(pd.DataFrame({"label": service_labels, "msname": largest_services.msname, "containers": largest_services.containers}), "largest_services")


def service_groups(column):
    groups, labels = [], []
    for label, service_name in zip(service_labels, largest_services.msname):
        values = service_sample.loc[service_sample.msname == service_name, column].to_numpy() * 100
        if len(values) >= 2:
            groups.append(values)
            labels.append(label)
    return labels, groups

# COMMAND ----------

fig, (cpu_axis, memory_axis) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
cpu_labels, cpu_groups = service_groups("cpu_utilization")
memory_labels, memory_groups = service_groups("memory_utilization")
draw_boxes(cpu_axis, cpu_labels, cpu_groups, SERIES_BLUE)
draw_boxes(memory_axis, memory_labels, memory_groups, SERIES_AQUA)
cpu_axis.set_ylabel("CPU utilization (%)")
memory_axis.set_ylabel("Memory utilization (%)")
memory_axis.tick_params(axis="x", rotation=30)
finish_figure(fig, "CPU and Memory Utilization by Microservice", "Microservice", None, "utilization_by_service.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The ten largest services differ much more from each other than within themselves. Service 1 runs near 60% CPU and Service 2 near 9%.
# MAGIC Each service holds a narrow memory band between 60% and 85%. Scaling thresholds must be set per service, not once for the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Outliers
# MAGIC
# MAGIC Tukey fences are applied **per microservice** to its minute level mean CPU. A minute is an outlier when it lies more than 3 interquartile ranges outside the quartiles of that same service.

# COMMAND ----------

def tukey_outlier_flags(frame, column, multiplier=3.0):
    quartiles = frame.groupby("msname")[column].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["q1", "q3"]
    merged = frame.merge(quartiles, left_on="msname", right_index=True)
    spread = merged.q3 - merged.q1
    return (merged[column] < merged.q1 - multiplier * spread) | (merged[column] > merged.q3 + multiplier * spread)


service_resource_pandas["cpu_outlier"] = tukey_outlier_flags(service_resource_pandas, "cpu_mean").to_numpy()
outlier_share_by_minute = service_resource_pandas.groupby("t_idx").cpu_outlier.mean() * 100
print(f"share of service minutes flagged as CPU outliers: {service_resource_pandas.cpu_outlier.mean():.2%}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(outlier_share_by_minute.index, outlier_share_by_minute.values, color=SERIES_ORANGE, alpha=0.2, linewidth=0)
ax.plot(outlier_share_by_minute.index, outlier_share_by_minute.values, color=SERIES_ORANGE, linewidth=1.4)
ax.set_ylim(bottom=0)
ax.set_xlim(FIRST_MINUTE, LAST_MINUTE)
finish_chart(fig, ax, "CPU Outlier Share Over Time", "Minute of trace", "Outlier service minutes (%)", "cpu_outliers_over_time.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Only 0.84% of service minutes are CPU outliers under per service 3 IQR fences. They arrive in short bursts and become more frequent after minute 650.
# MAGIC Forecast errors are expected to peak around these bursts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MSRTQps: instance level traffic
# MAGIC
# MAGIC MCR is the number of calls per second and RT is the mean response time in milliseconds (README of the trace). Each container and minute becomes one row: values from several records in the same minute are averaged, because a rate is not additive in time.
# MAGIC
# MAGIC * **Incoming call rate** = providerRPC + HTTP + consumerMQ calls per second.
# MAGIC * **Outgoing call rate** = consumerRPC calls per second.
# MAGIC
# MAGIC A paradigm that is missing in a row is counted as zero calls for the rate sums. Response times stay missing when there are no calls.

# COMMAND ----------

def null_safe_sum(columns):
    total = sum(F.coalesce(F.col(column), F.lit(0.0)) for column in columns)
    any_present = None
    for column in columns:
        present = F.col(column).isNotNull()
        any_present = present if any_present is None else (any_present | present)
    return F.when(any_present, total)


instance_minute_traffic = (
    msrtqps.groupBy("msname", "msinstanceid", "t_idx")
    .agg(*[F.mean(column).alias(column) for column in MCR_COLUMNS + RT_COLUMNS])
    .withColumn("incoming_mcr", null_safe_sum(INCOMING_MCR_COLUMNS))
    .withColumn("outgoing_mcr", null_safe_sum(OUTGOING_MCR_COLUMNS))
)
instance_minute_traffic.write.mode("overwrite").parquet(f"{WORK_PATH}/instance_minute_traffic")
instance_minute_traffic = spark.read.parquet(f"{WORK_PATH}/instance_minute_traffic")

traffic_summary_rows = []
for column in MCR_COLUMNS + ["incoming_mcr", "outgoing_mcr"] + RT_COLUMNS:
    row = summary_statistics(instance_minute_traffic.filter(F.col(column) > 0), column)
    row["zero_share"] = instance_minute_traffic.select(F.mean((F.col(column) == 0).cast("double"))).first()[0]
    row["present_share"] = instance_minute_traffic.select(F.mean(F.col(column).isNotNull().cast("double"))).first()[0]
    traffic_summary_rows.append(row)
traffic_summary = pd.DataFrame(traffic_summary_rows)
print("Instance minute traffic. Percentiles and moments use positive values only.")
show_table(traffic_summary[["column", "present_share", "zero_share", "median", "p95", "p99", "mean", "skewness"]].round(3), "traffic_summary")

# COMMAND ----------

def spark_log_histogram(table, column, bins_per_decade=20, lower_quantile=0.001, upper_quantile=0.999):
    positive = table.filter(F.col(column) > 0)
    low, high = positive.select(F.percentile_approx(column, [lower_quantile, upper_quantile], 10000)).first()[0]
    bucket = F.floor(F.log10(F.col(column)) * bins_per_decade).cast("int")
    counts = positive.withColumn("bucket", bucket).groupBy("bucket").count().toPandas()
    first_bucket = math.floor(math.log10(low) * bins_per_decade)
    last_bucket = math.floor(math.log10(high) * bins_per_decade)
    kept = counts[counts.bucket.between(first_bucket, last_bucket)]
    print(f"{column}: {1 - kept['count'].sum() / counts['count'].sum():.3%} of positive values lie outside {low:.3g} to {high:.3g} and are not drawn")
    full = pd.DataFrame({"bucket": range(first_bucket, last_bucket + 1)}).merge(kept, on="bucket", how="left").fillna(0)
    edges = 10 ** (np.arange(first_bucket, last_bucket + 2) / bins_per_decade)
    return edges, full["count"].to_numpy()


incoming_edges, incoming_counts = spark_log_histogram(instance_minute_traffic, "incoming_mcr")
outgoing_edges, outgoing_counts = spark_log_histogram(instance_minute_traffic, "outgoing_mcr")

fig, ax = plt.subplots(figsize=(10, 4.8))
draw_distribution(ax, incoming_edges, incoming_counts, color=SERIES_BLUE, label="Incoming calls")
draw_distribution(ax, outgoing_edges, outgoing_counts, color=SERIES_ORANGE, label="Outgoing calls")
ax.set_xscale("log")
ax.set_xlim(min(incoming_edges[0], outgoing_edges[0]), max(incoming_edges[-1], outgoing_edges[-1]))
ax.legend(loc="upper left")
finish_chart(fig, ax, "Instance Call Rate Distribution", "Calls per second (log scale)", "Instance minutes",
             "instance_call_rate_distribution.png", count_axis="y")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Container call rates span five orders of magnitude. The incoming median is 52 calls per second, the 99th percentile is 1,034 and the skewness is 124.
# MAGIC Outgoing rates are a little lower, with a median of 33. Traffic targets need a log transform before training.

# COMMAND ----------

paradigm_share = pd.DataFrame({
    "paradigm": INCOMING_MCR_COLUMNS,
    "calls_per_second": [instance_minute_traffic.select(F.sum(column)).first()[0] or 0.0 for column in INCOMING_MCR_COLUMNS]
})
paradigm_share["share_percent"] = 100 * paradigm_share.calls_per_second / paradigm_share.calls_per_second.sum()
show_table(paradigm_share.round(2), "incoming_calls_by_paradigm")

ordered_share = paradigm_share.sort_values("share_percent", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5.5))
draw_pie(ax, [name.replace("_MCR", "") for name in ordered_share.paradigm], ordered_share.calls_per_second,
         "Incoming Calls by Communication Paradigm")
save_and_show(fig, "incoming_calls_by_paradigm.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** providerRPC carries 71.8% of incoming calls, consumerMQ 18.2% and HTTP 10.0%. HTTP_RT and consumerMQ_RT copy the call rate and are left out.
# MAGIC Response time features therefore come only from providerRPC.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Coverage between MSResource and MSRTQps
# MAGIC
# MAGIC The forecasting features join the two tables on container and minute. This check shows how many resource instance minutes have traffic data.

# COMMAND ----------

joined_coverage = (
    instance_minute_resource.select("msinstanceid", "t_idx")
    .join(instance_minute_traffic.select("msinstanceid", "t_idx", F.lit(1).alias("has_traffic")), ["msinstanceid", "t_idx"], "left")
    .select(F.mean(F.coalesce(F.col("has_traffic"), F.lit(0)).cast("double"))).first()[0]
)
print(f"share of resource instance minutes with traffic data: {joined_coverage:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MSRTQps: microservice level traffic
# MAGIC
# MAGIC * **Microservice call rate** = the sum of the incoming call rates of its containers in that minute.
# MAGIC * **Microservice response time** = the call weighted mean of container response times,
# MAGIC * **Load imbalance** = the coefficient of variation (standard deviation divided by mean) of incoming call rate across the

# COMMAND ----------

response_time_paradigm = "providerRPC" if "providerRPC_RT" in RT_COLUMNS else (RT_COLUMNS[0].replace("_RT", "") if RT_COLUMNS else None)
weighted_rt_expressions = []
if response_time_paradigm is not None:
    rate, time = F.col(f"{response_time_paradigm}_MCR"), F.col(f"{response_time_paradigm}_RT")
    weight = F.when(time.isNotNull() & (rate > 0), rate)
    weighted_rt_expressions = [F.try_divide(F.sum(weight * time), F.sum(weight)).alias("response_time_ms")]

service_minute_traffic = instance_minute_traffic.groupBy("msname", "t_idx").agg(
    F.count("*").alias("traffic_instances"),
    F.sum("incoming_mcr").alias("incoming_mcr"),
    F.sum("outgoing_mcr").alias("outgoing_mcr"),
    F.try_divide(F.stddev_pop("incoming_mcr"), F.mean("incoming_mcr")).alias("load_imbalance"),
    *weighted_rt_expressions
)
service_minutes = service_minute_resource.join(service_minute_traffic, ["msname", "t_idx"], "full_outer").toPandas()
print(f"service minutes: {len(service_minutes):,}, microservices: {service_minutes.msname.nunique():,}")
print(f"response time column used: {response_time_paradigm}_RT")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Concentration of traffic
# MAGIC
# MAGIC The Lorenz curve orders microservices from least to most traffic and plots the cumulative share of calls. The Gini coefficient is 0 when all services get the same traffic and close to 1 when a few services get almost all of it.

# COMMAND ----------

def lorenz_curve(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.cumsum(ordered) / ordered.sum()
    population = np.arange(1, len(ordered) + 1) / len(ordered)
    return np.insert(population, 0, 0.0), np.insert(cumulative, 0, 0.0)


def gini_coefficient(values):
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * ordered)) / (n * ordered.sum()) - (n + 1) / n)


service_total_calls = service_minutes.groupby("msname").incoming_mcr.sum().fillna(0)
population_share, call_share = lorenz_curve(service_total_calls)
ordered_calls = np.sort(service_total_calls.to_numpy())[::-1]
top_1_percent = ordered_calls[: max(1, len(ordered_calls) // 100)].sum() / ordered_calls.sum()
top_10_percent = ordered_calls[: max(1, len(ordered_calls) // 10)].sum() / ordered_calls.sum()
traffic_gini = gini_coefficient(service_total_calls)
print(f"Gini coefficient of incoming calls across microservices: {traffic_gini:.3f}")
print(f"top 1% of microservices receive {top_1_percent:.1%} of incoming calls, top 10% receive {top_10_percent:.1%}")

fig, ax = plt.subplots(figsize=(6.8, 6.5))
ax.fill_between(population_share * 100, call_share * 100, population_share * 100, color="#86b6ef", alpha=0.35, linewidth=0)
ax.plot(population_share * 100, call_share * 100, color=SERIES_BLUE, linewidth=2.4, label="Observed")
ax.plot([0, 100], [0, 100], color=TEXT_MUTED, linestyle="dashed", linewidth=1.5, label="Equal traffic")
for population_point, top_share, label in ((90, top_10_percent, "Top 10%"), (99, top_1_percent, "Top 1%")):
    point = (population_point, 100 * (1 - top_share))
    ax.scatter(*point, color=SERIES_ORANGE, s=45, zorder=4)
    ax.annotate(label, point, textcoords="offset points", xytext=(-58, 4), color=TEXT_SECONDARY, fontsize=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.legend()
finish_chart(fig, ax, "Traffic Concentration Across Microservices", "Share of microservices (%)", "Share of incoming calls (%)",
             "traffic_concentration.png")

# COMMAND ----------

ranked_calls = service_total_calls.sort_values(ascending=False).to_numpy()
service_count = len(ranked_calls)
group_edges = [0, max(1, service_count // 100), max(2, service_count // 10), max(3, service_count // 2), service_count]
group_labels = ["Top 1% of services", "Next 9%", "Next 40%", "Bottom 50%"]
group_calls = [ranked_calls[start:end].sum() for start, end in zip(group_edges[:-1], group_edges[1:])]

fig, ax = plt.subplots(figsize=(8, 5.5))
draw_pie(ax, group_labels, group_calls, "Traffic Share by Microservice Group")
save_and_show(fig, "traffic_share_by_group.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Traffic is highly concentrated. The Gini coefficient is 0.871, the top 1% of services receive 34.8% of calls and the top 10% receive 79.9%.
# MAGIC Error metrics pooled over services will be dominated by a few hubs, so evaluation should be per service or weighted.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Total traffic and burst minutes
# MAGIC
# MAGIC A burst minute is a minute whose total call rate is more than `BURST_FACTOR` times the 15 minute rolling median. For each burst minute the table names the service whose call rate rose the most above its own median.

# COMMAND ----------

BURST_FACTOR = 3.0

total_rate_by_minute = service_minutes.groupby("t_idx").incoming_mcr.sum().sort_index()
rolling_median = total_rate_by_minute.rolling(15, center=True, min_periods=1).median()
is_burst_minute = total_rate_by_minute > BURST_FACTOR * rolling_median
cleaned_total_rate = total_rate_by_minute.where(~is_burst_minute, rolling_median)

rate_grid_all = service_minutes.pivot_table(index="t_idx", columns="msname", values="incoming_mcr")
excess_over_median = rate_grid_all - rate_grid_all.median()
service_rank = {name: rank for rank, name in enumerate(service_total_calls.sort_values(ascending=False).index, start=1)}
burst_rows = []
for minute in total_rate_by_minute.index[is_burst_minute]:
    top_service = excess_over_median.loc[minute].idxmax()
    burst_rows.append({
        "minute": int(minute),
        "total_calls_per_second": total_rate_by_minute[minute],
        "rolling_median": rolling_median[minute],
        "service_rank_by_volume": service_rank.get(top_service),
        "service_share_of_excess": excess_over_median.loc[minute, top_service] / (total_rate_by_minute[minute] - rolling_median[minute])
    })
burst_minutes = pd.DataFrame(burst_rows)
print(f"burst minutes: {len(burst_minutes)}")
show_table(burst_minutes.round(3), "burst_minutes")

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(total_rate_by_minute.index, total_rate_by_minute.values, color="#9ec5f4", linewidth=1.2, label="Per minute")
ax.plot(rolling_median.index, rolling_median.values, color="#0d366b", linewidth=2.2, label="15 minute rolling median")
ax.scatter(total_rate_by_minute.index[is_burst_minute], total_rate_by_minute[is_burst_minute], color=SERIES_ORANGE, s=40, zorder=4,
           label="Burst minute")
ax.set_yscale("log")
ax.set_xlim(FIRST_MINUTE, LAST_MINUTE)
ax.legend(loc="upper right", ncol=3)
finish_chart(fig, ax, "Total Incoming Call Rate Over Time", "Minute of trace", "Incoming calls per second (log scale)",
             "total_call_rate_over_time.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The baseline stays near 10 to 12 million calls per second for the whole trace. The large spikes near minutes 83 and 627 are burst minutes.
# MAGIC The table shows how much of each spike one service explains. They are single service events, not cluster wide surges.

# COMMAND ----------

busiest_services = service_total_calls.sort_values(ascending=False).index[:5]
fig, axes = plt.subplots(len(busiest_services), 1, figsize=(13, 11), sharex=True)
for rank, (ax, service_name, color) in enumerate(zip(axes, busiest_services, CATEGORY_COLORS), start=1):
    rows = service_minutes[(service_minutes.msname == service_name) & (service_minutes.incoming_mcr > 0)].sort_values("t_idx")
    floor_value = rows.incoming_mcr.min() * 0.8
    ax.fill_between(rows.t_idx, rows.incoming_mcr, floor_value, color=color, alpha=0.18, linewidth=0)
    ax.plot(rows.t_idx, rows.incoming_mcr, color=color, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor_value)
    ax.set_title(f"Service {rank}", loc="left", fontsize=11, color=color, pad=4)
    format_axes(ax)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
axes[-1].set_xlim(FIRST_MINUTE, LAST_MINUTE)
finish_figure(fig, "Call Rate of the Five Busiest Microservices", "Minute of trace", "Incoming calls per second (log scale)",
              "busiest_services_call_rate.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The five busiest services behave in five different ways. Service 1 drifts slowly, Service 2 is noisy with one burst and Service 3 is flat with isolated spikes.
# MAGIC Service 4 has frequent deep drops and Service 5 changes level in steps. One global forecasting rule will not fit all services.

# COMMAND ----------

busiest_ten = service_total_calls.sort_values(ascending=False).index[:10]
rate_groups = [service_minutes.loc[(service_minutes.msname == name) & (service_minutes.incoming_mcr > 0), "incoming_mcr"].to_numpy()
               for name in busiest_ten]
fig, ax = plt.subplots(figsize=(12, 5))
draw_boxes(ax, [f"Service {rank}" for rank in range(1, len(busiest_ten) + 1)], rate_groups, SERIES_ORANGE)
ax.set_yscale("log")
ax.tick_params(axis="x", rotation=30)
finish_chart(fig, ax, "Call Rate by Microservice", "Microservice", "Incoming calls per second (log scale)", "call_rate_by_service.png")

spread = [np.log10(np.percentile(values, 75) / np.percentile(values, 25)) for values in rate_groups if len(values)]
tell(
    "Each box is one busy service over all its minutes.",
    f"The middle half of a service's minutes spans on average {np.mean(spread):.2f} decades of call rate.",
    "Wide boxes mean bursty services. Narrow boxes mean steady services that are easy to forecast."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Call rate heatmap
# MAGIC
# MAGIC Each row is one of the busiest services and each column is a 5 minute bin (the median of its minutes), which removes single minute flicker. Colour is the log2 ratio to the service's own median, with symmetric limits taken from the 98th percentile of the data.

# COMMAND ----------

HEATMAP_SERVICES = 30
HEATMAP_BIN_MINUTES = 5
heatmap_services = service_total_calls.sort_values(ascending=False).index[:HEATMAP_SERVICES]
rank_of_service = {name: rank for rank, name in enumerate(heatmap_services, start=1)}
heatmap_rows = service_minutes[service_minutes.msname.isin(heatmap_services)].copy()
heatmap_rows["time_bin"] = heatmap_rows.t_idx // HEATMAP_BIN_MINUTES * HEATMAP_BIN_MINUTES

zero_share = heatmap_rows.assign(no_calls=heatmap_rows.incoming_mcr.fillna(0) <= 0).groupby("msname").no_calls.mean()
print("services in the heatmap with more than 10% of minutes without calls:")
show_table(zero_share[zero_share > 0.1].rename(lambda name: f"Service {rank_of_service[name]}").round(3).reset_index(), "heatmap_zero_minutes")

call_rate_grid = (
    heatmap_rows.pivot_table(index="msname", columns="time_bin", values="incoming_mcr", aggfunc="median")
    .reindex(index=heatmap_services, columns=range(FIRST_MINUTE, LAST_MINUTE + 1, HEATMAP_BIN_MINUTES))
)
positive_grid = call_rate_grid.where(call_rate_grid > 0)
log_ratio = np.log2(positive_grid.div(positive_grid.median(axis=1), axis=0))
colour_limit = float(np.clip(np.nanpercentile(np.abs(log_ratio.to_numpy()), 98), 0.5, 3))
row_order = leaves_list(linkage(log_ratio.fillna(0).to_numpy(), method="ward"))
ordered_ratio = log_ratio.iloc[row_order]

heatmap_colors = DIVERGING_BLUE_RED.copy()
heatmap_colors.set_bad(GRID_COLOR)
fig, ax = plt.subplots(figsize=(12, 8))
image = ax.imshow(ordered_ratio.to_numpy(), aspect="auto", cmap=heatmap_colors, vmin=-colour_limit, vmax=colour_limit,
                  interpolation="nearest", extent=(FIRST_MINUTE, LAST_MINUTE + 1, len(ordered_ratio) + 0.5, 0.5))
ax.set_yticks(range(1, len(ordered_ratio) + 1))
ax.set_yticklabels([f"Service {rank_of_service[name]}" for name in ordered_ratio.index], fontsize=8)
ax.grid(False)
tick_values = np.linspace(-colour_limit, colour_limit, 5)
colorbar = fig.colorbar(image, ax=ax, ticks=tick_values, pad=0.02)
colorbar.ax.minorticks_off()
colorbar.ax.set_yticklabels([f"{2 ** value:.2g}x" for value in tick_values])
colorbar.set_label("Call rate relative to service median")
finish_chart(fig, ax, f"Call Rate Heatmap of the {len(ordered_ratio)} Busiest Microservices", "Minute of trace", "Microservice",
             "call_rate_heatmap.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The busiest services do not move together. Some follow slow level shifts lasting hours, while others change bin by bin.
# MAGIC Clusters of rows with the same shape point to services that share upstream callers. Rows listed in the zero minutes table have reporting gaps, not real traffic drops.

# COMMAND ----------

imbalance = service_minutes[(service_minutes.traffic_instances >= 2)].load_imbalance.dropna()
print(f"load imbalance across containers: median {imbalance.median():.3f}, 90th percentile {imbalance.quantile(0.9):.3f}")
print("the chart shows values up to the 95th percentile")

imbalance_limit = imbalance.quantile(0.95)
imbalance_edges = np.linspace(0, imbalance_limit, 121)
imbalance_counts, _ = np.histogram(imbalance[imbalance <= imbalance_limit], bins=imbalance_edges)
fig, ax = plt.subplots(figsize=(10, 4.5))
draw_distribution(ax, imbalance_edges, imbalance_counts, color="#6250d6")
mark_median(ax, imbalance.median())
ax.set_xlim(0, imbalance_limit)
ax.legend(loc="upper right")
finish_chart(fig, ax, "Load Imbalance Across Instances", "Coefficient of variation of incoming call rate", "Service minutes",
             "load_imbalance.png", count_axis="y")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The median coefficient of variation across containers is 0.071, so most services spread load evenly. The 90th percentile is 0.550.
# MAGIC A tail of service minutes has strongly uneven load, which a service level forecast cannot see. Those services need instance level features.

# COMMAND ----------

if "response_time_ms" in service_minutes.columns:
    response_times = service_minutes.response_time_ms.dropna()
    response_times = response_times[response_times > 0]
    low, high = response_times.quantile([0.001, 0.999])
    shown_times = response_times[response_times.between(low, high)]
    time_edges = log_bins(shown_times, bins_per_decade=20)
    time_counts, _ = np.histogram(shown_times, bins=time_edges)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw_distribution(ax, time_edges, time_counts, color=SERIES_ORANGE)
    mark_median(ax, response_times.median())
    ax.set_xscale("log")
    ax.set_xlim(time_edges[0], time_edges[-1])
    ax.legend(loc="upper right")
    finish_chart(fig, ax, "Microservice Response Time Distribution", "Call weighted response time (ms, log scale)", "Service minutes",
                 "service_response_time_distribution.png", count_axis="y")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The call weighted providerRPC response time centres near 13 ms and ranges from under 1 ms to about 1 s.
# MAGIC The shape is close to log normal, so response time targets also need a log scale.

# COMMAND ----------

if "response_time_ms" in service_minutes.columns:
    rate_and_time = within_service_spearman(service_minutes, "incoming_mcr", "response_time_ms")
    print(f"call rate vs response time per microservice: median Spearman {rate_and_time.spearman.median():.3f} "
          f"over {len(rate_and_time)} microservices")
    plot_correlation_histogram(rate_and_time.spearman, "Call Rate and Response Time Correlation per Microservice",
                               "rate_response_time_correlation.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The median within service Spearman is 0.028 over 1,194 services, and the values spread across the full range.
# MAGIC Most services show no queueing effect at the observed load. Call rate alone cannot predict response time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Time series analysis
# MAGIC
# MAGIC The total call rate uses the cleaned series, with burst minutes replaced by the rolling median. ADF tests for a unit root and KPSS tests for stationarity, so both are reported.

# COMMAND ----------

def stationarity_tests(series, name):
    values = np.asarray(series, dtype=float)
    adf_statistic, adf_p_value = adfuller(values, autolag="AIC")[:2]
    kpss_statistic, kpss_p_value = kpss(values, regression="c", nlags="auto")[:2]
    return {"series": name, "adf_statistic": adf_statistic, "adf_p_value": adf_p_value,
            "kpss_statistic": kpss_statistic, "kpss_p_value": kpss_p_value}

# COMMAND ----------

cluster_series_over_time = pd.DataFrame({
    "Median CPU (%)": 100 * service_minutes.groupby("t_idx").cpu_mean.median(),
    "Median memory (%)": 100 * service_minutes.groupby("t_idx").memory_mean.median(),
    "Total calls per second": cleaned_total_rate,
    "Median response time (ms)": service_minutes.groupby("t_idx").response_time_ms.median()
}).sort_index()

fig, axes = plt.subplots(len(cluster_series_over_time.columns), 1, figsize=(13, 10), sharex=True)
for ax, column, color in zip(axes, cluster_series_over_time.columns, CATEGORY_COLORS):
    series = cluster_series_over_time[column]
    ax.plot(series.index, series.values, color=color, linewidth=1.2)
    ax.plot(series.index, series.rolling(30, center=True, min_periods=1).mean(), color=TEXT_PRIMARY, linewidth=1.6)
    ax.set_ylabel(column, fontsize=9)
axes[0].legend(["Per minute", "30 minute rolling mean"], loc="upper right", ncol=2)
axes[-1].set_xlim(FIRST_MINUTE, LAST_MINUTE)
finish_figure(fig, "Cluster Time Series", "Minute of trace", None, "cluster_time_series.png")

first_half = cluster_series_over_time.loc[:LAST_MINUTE // 2].mean()
second_half = cluster_series_over_time.loc[LAST_MINUTE // 2 + 1:].mean()
changes = 100 * (second_half - first_half) / first_half
tell(*[f"{column}: the second half of the trace is {change:+.1f}% against the first half." for column, change in changes.items()],
     "Large changes point to drift, which forecasting models must follow.")

# COMMAND ----------

ROLLING_WINDOW = 30
rolling_mean = cleaned_total_rate.rolling(ROLLING_WINDOW, center=True, min_periods=ROLLING_WINDOW // 2).mean()
rolling_std = cleaned_total_rate.rolling(ROLLING_WINDOW, center=True, min_periods=ROLLING_WINDOW // 2).std()

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(cleaned_total_rate.index, cleaned_total_rate.values, color="#9ec5f4", linewidth=1.1, label="Total call rate")
ax.fill_between(rolling_mean.index, rolling_mean - 2 * rolling_std, rolling_mean + 2 * rolling_std, color=SERIES_BLUE, alpha=0.15,
                linewidth=0, label="Rolling mean plus or minus 2 standard deviations")
ax.plot(rolling_mean.index, rolling_mean.values, color="#0d366b", linewidth=2, label=f"{ROLLING_WINDOW} minute rolling mean")
ax.set_xlim(FIRST_MINUTE, LAST_MINUTE)
ax.legend(loc="upper right")
finish_chart(fig, ax, "Rolling Mean and Standard Deviation of Total Call Rate", "Minute of trace", "Incoming calls per second",
             "rolling_statistics.png", count_axis="y")

variation = rolling_std / rolling_mean
tell(
    f"The rolling mean moves between {rolling_mean.min():,.0f} and {rolling_mean.max():,.0f} calls per second.",
    f"The rolling standard deviation is {100 * variation.median():.1f}% of the mean in a typical window.",
    "A flat mean and a steady band support a stationary series." if rolling_mean.max() / rolling_mean.min() < 1.2
    else "The mean shifts over the trace, so a forecasting model needs recent history, not only a long average."
)

# COMMAND ----------

traffic_levels = cleaned_total_rate.to_numpy()
traffic_changes = np.diff(traffic_levels)
stationarity = pd.DataFrame([
    stationarity_tests(traffic_levels, "total call rate"),
    stationarity_tests(traffic_changes, "minute to minute change")
])
show_table(stationarity.round(4), "stationarity_tests")
level_row = stationarity.iloc[0]
level_is_stationary = level_row.adf_p_value < 0.05 and level_row.kpss_p_value > 0.05
tell(
    "ADF rejects a unit root when its p value is below 0.05. KPSS rejects stationarity when its p value is below 0.05.",
    "Both tests agree that the total call rate is stationary." if level_is_stationary
    else "The tests do not agree that the level is stationary, so models should work with changes or a detrended series.",
    "KPSS p values are limited to the range 0.01 to 0.10 by its lookup table."
)

# COMMAND ----------

detrended = traffic_levels - pd.Series(traffic_levels).rolling(61, center=True, min_periods=1).mean().to_numpy()
frequencies, power = periodogram(detrended, detrend="linear")
valid = (frequencies > 0) & (1 / np.where(frequencies > 0, frequencies, 1) <= len(detrended) / 3)
periods, power = 1 / frequencies[valid], power[valid]
dominant_period = float(periods[np.argmax(power)])

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(periods, power, color=SERIES_BLUE, linewidth=1.4)
ax.fill_between(periods, power, color=SERIES_BLUE, alpha=0.15, linewidth=0)
ax.axvline(dominant_period, color=SERIES_ORANGE, linestyle="dashed", linewidth=1.4, label=f"Strongest period: {dominant_period:.0f} minutes")
ax.set_xscale("log")
ax.legend(loc="upper right")
finish_chart(fig, ax, "Periodogram of Total Call Rate", "Period (minutes, log scale)", "Spectral power", "periodogram.png")

power_share = power.max() / power.sum()
tell(
    f"The strongest repeating cycle has a period of about {dominant_period:.0f} minutes.",
    f"It holds {100 * power_share:.1f}% of the spectral power.",
    "That is a clear cycle worth modelling." if power_share > 0.15 else "No single cycle dominates, so the trace has no strong repeating pattern within 12 hours."
)

# COMMAND ----------

SEASON_PERIOD = int(min(max(round(dominant_period), 5), 180))
decomposition = STL(pd.Series(traffic_levels, index=cleaned_total_rate.index), period=SEASON_PERIOD).fit()
parts = {"Observed": decomposition.observed, "Trend": decomposition.trend, "Cycle": decomposition.seasonal, "Residual": decomposition.resid}

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, (name, series), color in zip(axes, parts.items(), [SERIES_BLUE, "#0d366b", SERIES_AQUA, SERIES_ORANGE]):
    if name == "Residual":
        ax.vlines(series.index, 0, series.values, color=color, linewidth=0.8)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    else:
        ax.plot(series.index, series.values, color=color, linewidth=1.3)
    ax.set_ylabel(name)
    ax.yaxis.set_major_formatter(THOUSANDS)
axes[-1].set_xlim(FIRST_MINUTE, LAST_MINUTE)
finish_figure(fig, f"STL Decomposition of Total Call Rate (cycle {SEASON_PERIOD} minutes)", "Minute of trace", None, "stl_decomposition.png")

trend_strength = max(0.0, 1 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid))
cycle_strength = max(0.0, 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid))
tell(
    f"Trend strength is {trend_strength:.2f} and cycle strength is {cycle_strength:.2f}, on a scale from 0 to 1.",
    "The trend explains most of the slow movement." if trend_strength > 0.6 else "The trend explains little of the variation. Most movement is short term.",
    "The cycle is strong enough to use as a feature." if cycle_strength > 0.6 else "The cycle does not repeat steadily, so a fixed periodic feature adds little.",
    "The residual holds the short bursts that a forecast cannot know in advance."
)

# COMMAND ----------

MAX_LAG = 40
autocorrelation, autocorrelation_band = acf(traffic_changes, nlags=MAX_LAG, alpha=0.05, fft=True)
partial_autocorrelation, partial_band = pacf(traffic_changes, nlags=MAX_LAG, alpha=0.05, method="ywm")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, values, band, title in ((axes[0], autocorrelation, autocorrelation_band, "Autocorrelation"),
                                (axes[1], partial_autocorrelation, partial_band, "Partial Autocorrelation")):
    lags = np.arange(1, len(values))
    half_width = band[1:, 1] - values[1:]
    outside = np.abs(values[1:]) > half_width
    ax.fill_between(lags, -half_width, half_width, color=AXIS_COLOR, alpha=0.45, linewidth=0, label="95% band")
    ax.vlines(lags, 0, values[1:], colors=np.where(outside, SERIES_ORANGE, SERIES_BLUE), linewidth=1.5)
    ax.scatter(lags, values[1:], color=np.where(outside, SERIES_ORANGE, SERIES_BLUE), s=18, zorder=3)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Lag (minutes)")
axes[0].set_ylabel("Correlation")
axes[0].legend(loc="upper right")
finish_figure(fig, "Autocorrelation of Call Rate Changes", None, None, "acf_pacf_call_rate_changes.png")

significant_partial = [lag for lag in range(1, len(partial_autocorrelation))
                       if abs(partial_autocorrelation[lag]) > partial_band[lag, 1] - partial_autocorrelation[lag]]
tell(
    f"The lag 1 autocorrelation of changes is {autocorrelation[1]:.2f}.",
    "A negative value means a jump is partly reversed in the next minute." if autocorrelation[1] < 0 else "A positive value means changes carry on into the next minute.",
    f"Partial autocorrelation is significant at lags {significant_partial[:6]}." if significant_partial else "No partial autocorrelation is significant.",
    f"An input window of about {max(significant_partial[:6]) if significant_partial else 1} minutes holds the direct temporal signal."
)

# COMMAND ----------

service_minutes["hour_of_trace"] = service_minutes.t_idx // 60
hours = sorted(service_minutes.hour_of_trace.dropna().astype(int).unique())
cpu_by_hour = [100 * service_minutes.loc[service_minutes.hour_of_trace == hour, "cpu_mean"].dropna().to_numpy() for hour in hours]
calls_by_hour = [cleaned_total_rate[(cleaned_total_rate.index // 60) == hour].to_numpy() for hour in hours]

fig, (cpu_axis, calls_axis) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
draw_boxes(cpu_axis, [str(hour) for hour in hours], cpu_by_hour, SERIES_BLUE)
draw_boxes(calls_axis, [str(hour) for hour in hours], calls_by_hour, SERIES_ORANGE)
cpu_axis.set_ylabel("Microservice CPU (%)")
calls_axis.set_ylabel("Total calls per second")
calls_axis.yaxis.set_major_formatter(THOUSANDS)
finish_figure(fig, "CPU and Call Rate by Hour of Trace", "Hour of trace", None, "hourly_box_plots.png")

hourly_medians = pd.Series([np.median(values) for values in calls_by_hour], index=hours)
tell(
    f"The busiest hour is hour {hourly_medians.idxmax()} and the quietest is hour {hourly_medians.idxmin()}.",
    f"Their median call rates differ by {100 * (hourly_medians.max() / hourly_medians.min() - 1):.1f}%.",
    "The trace covers 12 hours, so these boxes describe this trace only. They cannot show a daily cycle."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MSCallGraph: calls and dependencies
# MAGIC
# MAGIC Three properties of this table change how it must be counted.
# MAGIC
# MAGIC * **Sampling.** Call graphs are sampled at 0.5% and cover more than ten clusters, so counts are sample counts.
# MAGIC * **Duplicate records.** A remote call is recorded twice with the same `rpcid`, once by the caller and once by the callee.
# MAGIC * **Signed response time.** The caller record has a positive `rt` and the callee record has a negative one.

# COMMAND ----------

named_calls = callgraph.filter(
    F.col("UM").isNotNull() & F.col("DM").isNotNull()
    & ~F.col("UM").isin(MISSING_SERVICE_NAMES) & ~F.col("DM").isin(MISSING_SERVICE_NAMES)
)
callgraph_quality = callgraph.select(
    F.count("*").alias("records"),
    F.mean((F.col("UM").isNull() | F.col("UM").isin(MISSING_SERVICE_NAMES)).cast("double")).alias("missing_caller_share"),
    F.mean((F.col("DM").isNull() | F.col("DM").isin(MISSING_SERVICE_NAMES)).cast("double")).alias("missing_callee_share"),
    F.mean((F.col("rt") > 0).cast("double")).alias("positive_rt_share"),
    F.mean((F.col("rt") < 0).cast("double")).alias("negative_rt_share"),
    F.mean((F.col("rt") == 0).cast("double")).alias("zero_rt_share")
).first().asDict()
show_table(pd.DataFrame([callgraph_quality]).round(4), "callgraph_quality")

unique_calls = named_calls.dropDuplicates(["traceid", "rpcid"])
unique_calls.write.mode("overwrite").parquet(f"{WORK_PATH}/unique_calls")
unique_calls = spark.read.parquet(f"{WORK_PATH}/unique_calls")
caller_side_calls = named_calls.filter(F.col("rt") >= 0)
unique_call_count = unique_calls.count()
print(f"named call records: {named_calls.count():,}, unique calls: {unique_call_count:,}")

# COMMAND ----------

calls_by_type = unique_calls.groupBy("rpctype").count().toPandas()
calls_by_type["share_percent"] = 100 * calls_by_type["count"] / calls_by_type["count"].sum()

response_time_by_type = (
    caller_side_calls.groupBy("rpctype")
    .agg(F.percentile_approx("rt", [0.05, 0.25, 0.5, 0.75, 0.95], 10000).alias("percentiles"), F.count("*").alias("calls"))
    .toPandas()
)
response_time_by_type["median_ms"] = [row[2] for row in response_time_by_type.percentiles]
rpc_types = calls_by_type.merge(response_time_by_type, on="rpctype", how="inner").sort_values("median_ms").reset_index(drop=True)
show_table(rpc_types[["rpctype", "count", "share_percent", "median_ms", "percentiles"]].round(2), "rpc_type_summary")

RESPONSE_TIME_FLOOR_MS = 0.5
type_names = rpc_types.rpctype.astype(str).tolist()
fig, (share_axis, time_axis) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1.3]})
by_share = rpc_types.sort_values("count", ascending=False)
draw_pie(share_axis, by_share.rpctype.astype(str).tolist(), by_share["count"], "Share of Calls")
time_axis.set_title("Response Time", fontsize=12)
time_axis.set_ylabel("RPC type")

floored = [np.maximum(np.array(row, dtype=float), RESPONSE_TIME_FLOOR_MS) for row in rpc_types.percentiles]
time_boxes = [
    {"label": name, "whislo": row[0], "q1": row[1], "med": row[2], "q3": row[3], "whishi": row[4], "fliers": []}
    for name, row in zip(type_names, floored)
]
time_axis.bxp(time_boxes, vert=False, showfliers=False, patch_artist=True, widths=0.55,
              boxprops={"facecolor": "#9ec5f4", "edgecolor": SERIES_BLUE, "linewidth": 1.2},
              medianprops={"color": "#0d366b", "linewidth": 2},
              whiskerprops={"color": TEXT_MUTED, "linewidth": 1.2},
              capprops={"color": TEXT_MUTED, "linewidth": 1.2})
time_axis.set_xscale("log")
time_axis.set_xlabel("Response time (ms, log scale)")
format_axes(time_axis)
print(f"values under 1 ms are stored as 0 in the trace and are drawn at {RESPONSE_TIME_FLOOR_MS} ms")
finish_figure(fig, "Calls and Response Time by RPC Type", None, None, "rpc_type_calls_and_response_time.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Cache (mc) calls are 48.6% of all calls and rpc calls are 33.2%. mq, mc and db finish in about 1 ms at the median.
# MAGIC rpc has a median of 2 ms but a 95th percentile of 131 ms, and http (21 ms) and userDefined (25 ms) are the slowest.
# MAGIC Latency risk sits on rpc and userDefined edges, so the call type should be an edge feature in the GNN.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Dependency structure
# MAGIC
# MAGIC Out degree is the number of **distinct** microservices a service calls. In degree is the number of distinct services that call it.

# COMMAND ----------

edges = unique_calls.select("UM", "DM").distinct()
out_degree = edges.groupBy("UM").count().toPandas()["count"].to_numpy()
in_degree = edges.groupBy("DM").count().toPandas()["count"].to_numpy()
print(f"distinct caller to callee edges: {edges.count():,}")
print(f"out degree: median {np.median(out_degree):.0f}, 99th percentile {np.percentile(out_degree, 99):.0f}, max {out_degree.max()}")
print(f"in degree: median {np.median(in_degree):.0f}, 99th percentile {np.percentile(in_degree, 99):.0f}, max {in_degree.max()}")


fig, ax = plt.subplots(figsize=(9, 5.5))
for degrees, label, color in ((out_degree, "Out-degree (distinct callees)", SERIES_BLUE), (in_degree, "In-degree (distinct callers)", SERIES_ORANGE)):
    ordered = np.sort(degrees)
    share_at_least = 1.0 - np.arange(len(ordered)) / len(ordered)
    ax.step(ordered, share_at_least * 100, where="post", color=color, linewidth=2, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="lower left")
finish_chart(fig, ax, "Degree Distribution", "Degree (log scale)", "Microservices with at least this degree (%)", "degree_distribution.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** There are 53,960 distinct edges. The median out degree is 5 and the median in degree is 2, but a few hubs reach 485 callees and 447 callers.
# MAGIC The graph is sparse with heavy tailed hubs, so GNN message passing should normalise by degree.

# COMMAND ----------

top_edges = (
    unique_calls.groupBy("UM", "DM").count().orderBy(F.desc("count")).limit(20).toPandas()
    .rename(columns={"count": "sampled_calls"})
)
top_edges["estimated_calls_per_minute"] = top_edges.sampled_calls / CALL_GRAPH_SAMPLING_RATE / (LAST_MINUTE - FIRST_MINUTE + 1)
edge_services = list(pd.unique(top_edges[["UM", "DM"]].to_numpy().ravel()))
service_number = {name: number for number, name in enumerate(edge_services, start=1)}
top_edges["label"] = [f"Service {service_number[caller]} → Service {service_number[callee]}" for caller, callee in zip(top_edges.UM, top_edges.DM)]
show_table(top_edges[["label", "UM", "DM", "sampled_calls", "estimated_calls_per_minute"]], "top_call_edges")

edge_values = top_edges.estimated_calls_per_minute[::-1]
edge_colors = SEQUENTIAL_BLUE(0.3 + 0.7 * edge_values / edge_values.max())
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.barh(top_edges.label[::-1], edge_values, color=edge_colors, height=0.7)
ax.grid(axis="y", visible=False)
finish_chart(fig, ax, "Top 20 Call Edges", "Estimated calls per minute", "Caller to callee", "top_call_edges.png", count_axis="x")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** Two callers dominate. Service 1 appears in 6 of the top 20 edges and Service 3 in 8. Two of the top edges are self calls and one caller is EXTERNAL.
# MAGIC Self loops and the external entry node need explicit handling when the graph is built.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Relationships across tables
# MAGIC
# MAGIC **Cluster level.** Four minute level series are compared: median microservice CPU, median microservice memory, total incoming call rate (burst minutes removed) and median microservice response time. Two series that both trend over the 12 hours can be strongly correlated without any link between them.
# MAGIC
# MAGIC **Microservice level.** For each microservice, the Spearman correlation between its call rate and its CPU over its minutes shows whether traffic drives utilisation inside that service.

# COMMAND ----------

cluster_series = pd.DataFrame({
    "CPU median": service_minutes.groupby("t_idx").cpu_mean.median(),
    "Memory median": service_minutes.groupby("t_idx").memory_mean.median(),
    "Total call rate": cleaned_total_rate,
    "Median response time": service_minutes.groupby("t_idx").response_time_ms.median()
}).sort_index().dropna()
print(f"white noise threshold for one minute changes: {1.96 / np.sqrt(len(cluster_series) - 1):.3f}")
print("sampled calls are left out because the call graph has a collection gap from minute 220 to 330")

level_correlation = cluster_series.corr(method="spearman")
change_correlation = cluster_series.diff().dropna().corr(method="spearman")
print("Spearman correlation of minute level values:")
show_table(level_correlation.round(3).reset_index(names="series"), "correlation_levels")
print("\nSpearman correlation of minute to minute changes:")
show_table(change_correlation.round(3).reset_index(names="series"), "correlation_changes")

# COMMAND ----------

def draw_correlation_triangle(ax, matrix, title):
    lower = matrix.iloc[1:, :-1].copy()
    upper_mask = np.triu(np.ones(lower.shape, dtype=bool), k=1)
    values = lower.to_numpy(dtype=float, copy=True)
    values[upper_mask] = np.nan
    image = ax.imshow(values, cmap=DIVERGING_BLUE_RED, vmin=-1, vmax=1)
    ax.set_xticks(range(len(lower.columns)))
    ax.set_xticklabels(lower.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(lower.index)))
    ax.set_yticklabels(lower.index)
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(lower.index)):
        for column in range(len(lower.columns)):
            if not upper_mask[row, column]:
                value = values[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if abs(value) > 0.6 else TEXT_PRIMARY)
    return image


fig, (level_axis, change_axis) = plt.subplots(1, 2, figsize=(14, 5.8))
draw_correlation_triangle(level_axis, level_correlation, "Values")
image = draw_correlation_triangle(change_axis, change_correlation, "Changes")
change_axis.set_yticklabels([])
level_axis.set_ylabel("Series")
level_axis.set_xlabel("Series")
change_axis.set_xlabel("Series")
fig.colorbar(image, ax=[level_axis, change_axis], label="Spearman correlation", shrink=0.85)
fig.suptitle("Correlation of Minute-Level Values and Changes", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_values_and_changes.png"), dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

# COMMAND ----------

white_noise_threshold = 1.96 / np.sqrt(len(cluster_series) - 1)
level_value = level_correlation.loc["Total call rate", "CPU median"]
change_value = change_correlation.loc["Total call rate", "CPU median"]
strong_changes = [
    f"{first} and {second} ({change_correlation.loc[first, second]:.2f})"
    for row, first in enumerate(change_correlation.index) for second in change_correlation.columns[:row]
    if abs(change_correlation.loc[first, second]) > white_noise_threshold
]
print(f"Interpretation: CPU median and total call rate correlate at {level_value:.2f} on levels but {change_value:.2f} on changes. "
      "A level correlation that disappears after differencing is a shared drift, not a minute to minute link.")
print("Change correlations above the white noise threshold:", strong_changes if strong_changes else "none")

# COMMAND ----------

traffic_and_cpu = within_service_spearman(service_minutes, "incoming_mcr", "cpu_mean")
traffic_and_memory = within_service_spearman(service_minutes, "incoming_mcr", "memory_mean")
print(f"call rate vs CPU per microservice: median Spearman {traffic_and_cpu.spearman.median():.3f}, "
      f"{(traffic_and_cpu.spearman > 0.5).mean():.1%} of {len(traffic_and_cpu)} microservices above 0.5")
print(f"call rate vs memory per microservice: median Spearman {traffic_and_memory.spearman.median():.3f}, "
      f"{(traffic_and_memory.spearman > 0.5).mean():.1%} of {len(traffic_and_memory)} microservices above 0.5")

plot_correlation_histogram(traffic_and_cpu.spearman, "Call Rate and CPU Correlation per Microservice", "rate_cpu_correlation.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The median within service Spearman is 0.596, and 58.2% of 1,297 services are above 0.5. Traffic is a strong driver of CPU for most services.
# MAGIC This supports forecasting CPU from traffic features.

# COMMAND ----------

plot_correlation_histogram(traffic_and_memory.spearman, "Call Rate and Memory Correlation per Microservice", "rate_memory_correlation.png")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation.** The median is -0.064 and only 7.9% of services are above 0.5. Memory does not follow traffic.
# MAGIC Memory forecasting should rely on its own lagged history rather than traffic features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Experiments for graph based forecasting
# MAGIC
# MAGIC Five experiments test assumptions behind a GNN traffic forecaster and a resource optimiser.
# MAGIC
# MAGIC | Experiment | Question |
# MAGIC |---|---|
# MAGIC | 9.1 Caller to callee cross correlation | Do connected services move together more than unconnected ones? |
# MAGIC | 9.2 CPU elasticity | How strongly does CPU scale with calls per container? |
# MAGIC | 9.3 Traffic burstiness | Which services are bursty and hard to forecast? |
# MAGIC | 9.4 Lag 1 autocorrelation | How strong is the persistence baseline a model must beat? |
# MAGIC | 9.5 Response time by CPU | At what utilisation does latency start to rise? |
# MAGIC
# MAGIC All five use one minute changes or per service values, so the shared drift found in section 8 does not create false structure.

# COMMAND ----------

full_minutes = range(FIRST_MINUTE, LAST_MINUTE + 1)
rate_by_service = service_minutes.pivot_table(index="t_idx", columns="msname", values="incoming_mcr").reindex(full_minutes)
rate_changes = np.log1p(rate_by_service).diff()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.1 Caller to callee cross correlation
# MAGIC
# MAGIC For the 300 busiest caller to callee edges, the correlation between one minute changes of log call rate is computed at lags of minus 5 to 5 minutes. A positive lag means the caller moves first.

# COMMAND ----------

EDGE_COUNT = 300
MAX_LAG_MINUTES = 5
NON_SERVICE_NAMES = MISSING_SERVICE_NAMES + ["EXTERNAL", "USER"]
lags = list(range(-MAX_LAG_MINUTES, MAX_LAG_MINUTES + 1))

dependency_edges = (
    unique_calls.filter((F.col("UM") != F.col("DM")) & ~F.col("UM").isin(NON_SERVICE_NAMES) & ~F.col("DM").isin(NON_SERVICE_NAMES))
    .groupBy("UM", "DM").count().orderBy(F.desc("count")).limit(EDGE_COUNT * 3).toPandas()
)
dependency_edges = dependency_edges[dependency_edges.UM.isin(rate_changes.columns) & dependency_edges.DM.isin(rate_changes.columns)].head(EDGE_COUNT)


def lagged_correlations(caller, callee):
    return [rate_changes[callee].corr(rate_changes[caller].shift(lag)) for lag in lags]


edge_correlations = pd.DataFrame(
    [lagged_correlations(caller, callee) for caller, callee in zip(dependency_edges.UM, dependency_edges.DM)], columns=lags
)

all_edge_pairs = set(map(tuple, edges.toPandas().to_numpy()))
candidate_services = list(rate_changes.columns[rate_changes.notna().sum() >= MIN_MINUTES_PER_SERVICE])
random_generator = np.random.default_rng(7)
random_pairs = []
attempts = 0
while len(random_pairs) < len(dependency_edges) and attempts < 100 * max(len(dependency_edges), 1):
    attempts += 1
    first, second = random_generator.choice(candidate_services, 2, replace=False)
    if (first, second) not in all_edge_pairs and (second, first) not in all_edge_pairs:
        random_pairs.append((first, second))
random_correlations = pd.DataFrame([lagged_correlations(first, second) for first, second in random_pairs], columns=lags)

best_lag = edge_correlations.abs().idxmax(axis=1)
mann_whitney = stats.mannwhitneyu(edge_correlations[0].dropna(), random_correlations[0].dropna(), alternative="greater")
connected_median = edge_correlations[0].median()
unconnected_median = random_correlations[0].median()
print(f"edges: {len(dependency_edges)}, random pairs: {len(random_pairs)}")
print(f"median lag 0 correlation: connected {connected_median:.3f}, unconnected {unconnected_median:.3f}, "
      f"Mann Whitney p value {mann_whitney.pvalue:.2g}")
print(f"strongest lag: caller first {(best_lag > 0).mean():.1%}, same minute {(best_lag == 0).mean():.1%}, callee first {(best_lag < 0).mean():.1%}")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.fill_between(lags, edge_correlations.quantile(0.25), edge_correlations.quantile(0.75), color="#86b6ef", alpha=0.35, linewidth=0,
                label="Connected pairs, 25th to 75th percentile")
ax.plot(lags, edge_correlations.median(), color=SERIES_BLUE, marker="o", label="Connected pairs, median")
ax.plot(lags, random_correlations.median(), color=TEXT_MUTED, marker="o", linestyle="dashed", label="Unconnected pairs, median")
ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
ax.set_xticks(lags)
ax.legend(loc="upper right")
finish_chart(fig, ax, "Caller to Callee Cross-Correlation", "Lag in minutes (positive means caller leads)",
             "Correlation of call rate changes", "caller_callee_cross_correlation.png")

if mann_whitney.pvalue < 0.01 and connected_median > unconnected_median:
    print("Interpretation: connected services move together far more than unconnected ones. The call graph carries forecasting signal, which supports a GNN.")
else:
    print("Interpretation: connected services do not move together more than random pairs at one minute resolution. The graph adds little same minute signal.")
if (best_lag == 0).mean() >= 0.5:
    print("Most edges peak at lag 0. Calls propagate within the minute, so the value of the graph is co-movement rather than caller lead time.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.2 CPU elasticity to call rate
# MAGIC
# MAGIC For each service, the Theil Sen slope of log CPU on log calls per container. A slope of 1 means CPU grows in proportion to traffic.

# COMMAND ----------

service_minutes["calls_per_instance"] = service_minutes.incoming_mcr / service_minutes.traffic_instances


def cpu_elasticity(rows):
    rows = rows[(rows.calls_per_instance > 0) & (rows.cpu_mean > 0)]
    if len(rows) < MIN_MINUTES_PER_SERVICE or rows.calls_per_instance.nunique() < 2:
        return np.nan
    return stats.theilslopes(np.log(rows.cpu_mean), np.log(rows.calls_per_instance))[0]


elasticity = service_minutes.groupby("msname")[["calls_per_instance", "cpu_mean"]].apply(cpu_elasticity).dropna()
print(f"CPU elasticity: median {elasticity.median():.2f}, interquartile range {elasticity.quantile(0.25):.2f} to {elasticity.quantile(0.75):.2f}, "
      f"{len(elasticity)} microservices")

elasticity_edges = np.linspace(-0.5, 1.5, 81)
elasticity_counts, _ = np.histogram(elasticity.clip(-0.5, 1.5), bins=elasticity_edges)
fig, ax = plt.subplots(figsize=(10, 4.5))
draw_distribution(ax, elasticity_edges, elasticity_counts, color=SERIES_BLUE)
mark_median(ax, elasticity.median())
ax.axvline(1, color=SERIES_ORANGE, linewidth=1.3, label="Proportional scaling")
ax.set_xlim(-0.5, 1.5)
ax.legend(loc="upper right")
finish_chart(fig, ax, "CPU Elasticity to Call Rate", "Elasticity (log-log slope)", "Microservices", "cpu_elasticity.png")

share_proportional = (elasticity > 0.8).mean()
print(f"Interpretation: the median elasticity is {elasticity.median():.2f} and {share_proportional:.1%} of services are above 0.8. "
      + ("CPU is mostly a fixed baseline, so scaling containers by traffic alone over provisions most services."
         if elasticity.median() < 0.6 else "CPU scales close to proportionally with traffic, so traffic forecasts translate directly into CPU needs."))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.3 Traffic burstiness
# MAGIC
# MAGIC Burstiness B = (standard deviation minus mean) / (standard deviation plus mean) of a service's call rate over its minutes. B is -1 for constant traffic, 0 for Poisson like traffic and above 0 for bursty traffic.

# COMMAND ----------

rate_statistics = service_minutes.groupby("msname").incoming_mcr.agg(["mean", "std", "count"])
rate_statistics = rate_statistics[(rate_statistics["count"] >= MIN_MINUTES_PER_SERVICE) & (rate_statistics["mean"] > 0)].copy()
rate_statistics["burstiness"] = (rate_statistics["std"] - rate_statistics["mean"]) / (rate_statistics["std"] + rate_statistics["mean"])
volume_burstiness_spearman = stats.spearmanr(rate_statistics["mean"], rate_statistics.burstiness).correlation
print(f"burstiness: median {rate_statistics.burstiness.median():.2f}, share above 0: {(rate_statistics.burstiness > 0).mean():.1%}")
print(f"Spearman of mean rate and burstiness: {volume_burstiness_spearman:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(rate_statistics["mean"], rate_statistics.burstiness, s=18, color=SERIES_BLUE, alpha=0.45, linewidths=0, label="Microservice")
ax.axhline(0, color=SERIES_ORANGE, linewidth=1.3, linestyle="dashed", label="Poisson like")
ax.set_xscale("log")
ax.set_ylim(-1, 1)
ax.legend(loc="upper right")
finish_chart(fig, ax, "Traffic Burstiness by Microservice Volume", "Mean incoming calls per second (log scale)", "Burstiness",
             "traffic_burstiness.png")

print(f"Interpretation: {(rate_statistics.burstiness > 0).mean():.1%} of services are burstier than Poisson traffic. "
      + ("Busy services are smoother than small ones, so forecast errors will concentrate in low volume services and should be reported by volume tier."
         if volume_burstiness_spearman < -0.1 else "Burstiness does not fall with volume, so busy services are as hard to forecast as small ones."))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.4 Lag 1 autocorrelation per service
# MAGIC
# MAGIC The lag 1 autocorrelation of each service's log call rate and of its CPU. A high value means the last minute already predicts the next one well, so persistence is a strong baseline and model skill must be measured against it.

# COMMAND ----------

cpu_by_service = service_minutes.pivot_table(index="t_idx", columns="msname", values="cpu_mean").reindex(full_minutes)


def lag_one_autocorrelation(series):
    return series.autocorr(lag=1) if series.notna().sum() >= MIN_MINUTES_PER_SERVICE and series.nunique() > 1 else np.nan


call_rate_persistence = np.log1p(rate_by_service).apply(lag_one_autocorrelation).dropna()
cpu_persistence = cpu_by_service.apply(lag_one_autocorrelation).dropna()
print(f"lag 1 autocorrelation median: call rate {call_rate_persistence.median():.3f}, CPU {cpu_persistence.median():.3f}")

persistence_edges = np.linspace(-0.5, 1, 61)
fig, ax = plt.subplots(figsize=(10, 4.5))
for values, color, label in ((call_rate_persistence, SERIES_BLUE, "Call rate"), (cpu_persistence, SERIES_AQUA, "CPU utilization")):
    counts, _ = np.histogram(values.clip(-0.5, 1), bins=persistence_edges)
    draw_distribution(ax, persistence_edges, counts, color=color, label=label)
ax.set_xlim(-0.5, 1)
ax.legend(loc="upper left")
finish_chart(fig, ax, "Lag-1 Autocorrelation per Microservice", "Lag-1 autocorrelation", "Microservices", "lag_one_autocorrelation.png")

print(f"Interpretation: {(call_rate_persistence > 0.8).mean():.1%} of services have call rate persistence above 0.8 and "
      f"{(cpu_persistence > 0.8).mean():.1%} have CPU persistence above 0.8. "
      "Where persistence is high the model must beat the persistence baseline. Services with low persistence gain most from neighbour information.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9.5 Response time by CPU utilisation
# MAGIC
# MAGIC Response time is divided by each service's own median, so services with different speeds can be compared. Service minutes are grouped into 5 point CPU bins with at least 200 minutes each.

# COMMAND ----------

latency_rows = service_minutes[["msname", "cpu_mean", "response_time_ms"]].dropna()
latency_rows = latency_rows[latency_rows.response_time_ms > 0].copy()
latency_rows["relative_response_time"] = latency_rows.response_time_ms / latency_rows.groupby("msname").response_time_ms.transform("median")
latency_rows["cpu_bin"] = (latency_rows.cpu_mean * 100) // 5 * 5 + 2.5
binned_latency = latency_rows.groupby("cpu_bin").relative_response_time.describe(percentiles=[0.25, 0.5, 0.75])
binned_latency = binned_latency[binned_latency["count"] >= 200]
show_table(binned_latency.round(3).reset_index(), "response_time_by_cpu")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.fill_between(binned_latency.index, binned_latency["25%"], binned_latency["75%"], color="#f6b89f", alpha=0.45, linewidth=0,
                label="25th to 75th percentile")
ax.plot(binned_latency.index, binned_latency["50%"], color=SERIES_ORANGE, marker="o", label="Median")
ax.axhline(1, color=TEXT_MUTED, linewidth=1, linestyle="dotted")
ax.legend(loc="upper left")
finish_chart(fig, ax, "Response Time by CPU Utilization", "CPU utilization (%)", "Response time relative to service median",
             "response_time_by_cpu.png")

rising_bins = binned_latency.index[binned_latency["50%"] > 1.2]
if len(rising_bins):
    print(f"Interpretation: median response time first rises 20% above the service median at about {rising_bins.min():.0f}% CPU. "
          "This marks the queueing knee and a safe utilisation ceiling for the optimiser.")
else:
    print("Interpretation: response time stays within 20% of the service median at every CPU level. "
          "Latency is driven by dependencies rather than local CPU, which argues for graph aware optimisation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Statistical methods, references and their impact on this EDA
# MAGIC
# MAGIC | Method | Where | Reference | Impact on this EDA |
# MAGIC |---|---|---|---|
# MAGIC | Averaging records per container and minute before analysis | Sections 3 to 5 | Basic weighting principle: each sampling unit counts once | MSResource has more than one record per container and minute. Without this step, containers with more records dominate every statistic. |
# MAGIC | Median, interquartile range and percentiles instead of mean and standard deviation | Sections 4 to 7 | Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison Wesley. | Utilisation, call rates and response times are strongly skewed (skewness printed in sections 4 and 5). Mean plus or minus one standard deviation gives impossible negative values and hides the tail. |
# MAGIC | Tukey fences, applied per microservice | Section 4.4 | Tukey (1977) | Pooled fences flag busy services as outliers. Per service fences flag unusual minutes of a service, which is what matters for forecasting. |
# MAGIC | Log scaled histograms | Sections 5 to 7 | Standard practice for data spanning several orders of magnitude (Tukey 1977) | Call rates span several decades. A linear axis puts almost all data in the first bar. |
# MAGIC | Spearman rank correlation next to Pearson | Sections 4.2, 6, 8 | Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72 to 101. Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. *Proceedings of the Royal Society of London*, 58, 240 to 242. | Pearson is dominated by a few extreme values in long tailed data. Spearman measures any monotonic relation. The earlier binned Pearson with counts capped at 1,000 was biased and is replaced by an exact Pearson and a sampled Spearman. |
# MAGIC | Within service correlation instead of only pooled correlation | Sections 4.2, 6, 8 | Robinson, W. S. (1950). Ecological correlations and the behavior of individuals. *American Sociological Review*, 15(3), 351 to 357. Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *JRSS Series B*, 13(2), 238 to 241. | A pooled correlation mixes differences between services with changes inside a service and can even have the opposite sign. The forecasting models predict each service over time, so the within service value is the relevant one. |
# MAGIC | Correlating minute to minute changes instead of levels | Section 8 | Yule, G. U. (1926). Why do we sometimes get nonsense correlations between time series? *JRSS*, 89(1), 1 to 63. Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111 to 120. | Series that trend over the 12 hours look correlated even when unrelated. The earlier cross table heatmap used levels only. |
# MAGIC | Augmented Dickey Fuller and KPSS tests | Section 6.2 | Dickey, D. A. and Fuller, W. A. (1979). *JASA*, 74(366), 427 to 431. Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. and Shin, Y. (1992). *Journal of Econometrics*, 54(1 to 3), 159 to 178. | Decides whether autocorrelation of levels is meaningful or mostly reflects a trend. If the level series is not stationary, the ACF of changes is the one to use for choosing window sizes. |
# MAGIC | Bartlett band for the autocorrelation function | Section 6.2 | Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time series. *JRSS Supplement*, 8(1), 27 to 41. | The fixed band of 1.96 divided by the square root of n assumes white noise and marks too many lags as significant for persistent series. |
# MAGIC | No daily seasonality from 12 hours of data | Sections 4 and 6 | Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition. OTexts. | A seasonal pattern needs several full cycles to be estimated. The earlier `t_idx mod 24` profile was a 24 minute cycle labelled as hours and is removed. |
# MAGIC | Lorenz curve and Gini coefficient | Section 6.1 | Lorenz, M. O. (1905). Methods of measuring the concentration of wealth. *Publications of the American Statistical Association*, 9(70), 209 to 219. Gini, C. (1912). *Variabilita e mutabilita*. | Quantifies how unevenly traffic is spread over microservices. A high concentration means aggregate metrics are driven by a few services and per service evaluation is needed. |
# MAGIC | Call weighted mean response time | Section 6 | Definition of a mean over calls | A plain mean over containers gives an idle container the same weight as a busy one. |
# MAGIC | Coefficient of variation for load imbalance | Section 6 | Scale free spread measure (standard deviation divided by mean) | Lets services with very different call rates be compared. |
# MAGIC | Horvitz Thompson scaling of sampled counts | Section 7 | Horvitz, D. G. and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *JASA*, 47(260), 663 to 685. | Call graph counts are from a 0.5% sample of call graphs. Dividing by the sampling rate gives an unbiased estimate of totals. Shares and rankings need no scaling. |
# MAGIC | Counting distinct neighbours for degree, one record per call, caller side response time | Section 7 | Trace README (Luo et al. 2021, *Characterizing Microservice Dependency and Performance: Alibaba Trace Analysis*, SoCC 2021) | Remote calls are recorded twice with opposite signs of `rt`. Without these rules call counts double and mean response time is pulled towards zero. |
# MAGIC | Burst minutes from a rolling median filter | Section 6.2 | Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA*, 69(346), 383 to 393. | One service spike at minute 83 or 627 would distort the stationarity tests, the autocorrelation and the cluster correlations. Replacing burst minutes with the rolling median keeps those statistics about the whole cluster. |
# MAGIC | Ward hierarchical clustering of heatmap rows | Section 6.3 | Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301), 236 to 244. | Groups services with similar time profiles, so shared upstream patterns become visible. |
# MAGIC | Cross correlation of differenced series and the Mann Whitney U test | Section 9.1 | Box, G. E. P. and Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden Day. Mann, H. B. and Whitney, D. R. (1947). *Annals of Mathematical Statistics*, 18(1), 50 to 60. | Tests whether call graph neighbours share traffic changes. This is the empirical case for using a GNN. |
# MAGIC | Theil Sen slope | Section 9.2 | Theil, H. (1950). *Indagationes Mathematicae*, 12, 85 to 91. Sen, P. K. (1968). *JASA*, 63(324), 1379 to 1389. | A robust elasticity of CPU to traffic per service. It shows how much CPU a traffic forecast can explain. |
# MAGIC | Burstiness coefficient | Section 9.3 | Goh, K. I. and Barabasi, A. L. (2008). Burstiness and memory in complex systems. *EPL*, 81(4), 48002. | Separates smooth and bursty services, which predicts where forecast errors will be largest. |
# MAGIC | Lag 1 autocorrelation as a persistence baseline | Section 9.4 | Hyndman and Athanasopoulos (2021) | Shows how strong the naive baseline is, so model skill is judged against it. |
# MAGIC | Response time against utilisation | Section 9.5 | Kleinrock, L. (1975). *Queueing Systems, Volume 1: Theory*. Wiley. | Locates the utilisation where latency starts to rise, which sets the target for the optimiser. |
# MAGIC | Approximate percentiles in Spark | All sections | Greenwald, M. and Khanna, S. (2001). Space efficient online computation of quantile summaries. *SIGMOD 2001*. Spark's `percentile_approx` uses a summary of this family. | Percentiles have a small rank error, controlled by the accuracy setting of 10000. This is far below what matters for the charts. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary of findings
# MAGIC
# MAGIC The cell below collects the main numbers computed above, so the summary always matches the data that was loaded.

# COMMAND ----------

findings = {
    "msrtqps_minutes_present": int(len(msrtqps_minutes)),
    "resource_instance_minutes_with_traffic": float(joined_coverage),
    "cpu_median_percent": float(resource_summary.loc[0, "median"]),
    "cpu_p95_percent": float(resource_summary.loc[0, "p95"]),
    "memory_median_percent": float(resource_summary.loc[1, "median"]),
    "cpu_memory_pooled_pearson": float(pooled_pearson),
    "cpu_memory_pooled_spearman": float(pooled_spearman),
    "cpu_memory_within_service_median_spearman": float(cpu_memory_by_service.spearman.median()),
    "traffic_gini": traffic_gini,
    "top_1_percent_services_call_share": float(top_1_percent),
    "top_10_percent_services_call_share": float(top_10_percent),
    "load_imbalance_median": float(imbalance.median()),
    "call_rate_cpu_within_service_median_spearman": float(traffic_and_cpu.spearman.median()),
    "call_rate_memory_within_service_median_spearman": float(traffic_and_memory.spearman.median()),
    "suspect_rt_columns": SUSPECT_RT_COLUMNS,
    "unique_sampled_calls": int(unique_call_count),
    "burst_minutes": int(is_burst_minute.sum()),
    "connected_pairs_lag_0_median_correlation": float(connected_median),
    "unconnected_pairs_lag_0_median_correlation": float(unconnected_median),
    "cpu_elasticity_median": float(elasticity.median()),
    "burstiness_median": float(rate_statistics.burstiness.median()),
    "call_rate_lag_1_autocorrelation_median": float(call_rate_persistence.median()),
    "cpu_lag_1_autocorrelation_median": float(cpu_persistence.median())
}
readable_findings = [
    {"finding": name, "value": f"{value:.4f}" if isinstance(value, float) else str(value)} for name, value in findings.items()
]
show_table(pd.DataFrame(readable_findings), "eda_findings")

with open(os.path.join(OUTPUT_DIR, "eda_findings.json"), "w") as handle:
    json.dump(findings, handle, indent=2)