"""
eda.py
------
Four analysis functions, each returning:
  { "title": str, "finding": str, "chart_path": str | None }

"finding" is a plain-English sentence fed to the Claude prompt.
"chart_path" is a saved PNG shown in the Streamlit UI.
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
PALETTE = ["#1F4E79", "#2E75B6", "#5BA3D9", "#A8C8E8", "#D6E8F5"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig, name: str) -> str:
    path = os.path.join(CHARTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 1. Revenue / primary metric trend over time ───────────────────────────────
def revenue_trend(df: pd.DataFrame, summary: dict) -> dict:
    """
    Line chart of the primary numeric metric grouped by month.
    Works with any dataset that has a date column + at least one numeric column.
    Tries to find a 'sales' or 'revenue' column first, else uses the first numeric col.
    """
    date_col = summary["date_column"]
    num_cols  = summary["numeric_columns"]

    if not date_col or not num_cols:
        return {"title": "Revenue Trend", "finding": "No date or numeric column found for trend analysis.", "chart_path": None}

    # Pick best metric column
    metric_col = _pick_metric(num_cols, ["sales", "revenue", "amount", "total"])

    temp = df[[date_col, metric_col]].dropna().copy()
    temp["_month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = temp.groupby("_month")[metric_col].sum().reset_index()

    if len(monthly) < 2:
        return {"title": "Revenue Trend", "finding": "Not enough date range for a trend.", "chart_path": None}

    # Stats for finding sentence
    peak_row  = monthly.loc[monthly[metric_col].idxmax()]
    total     = monthly[metric_col].sum()
    first_val = monthly[metric_col].iloc[0]
    last_val  = monthly[metric_col].iloc[-1]
    pct_chg   = ((last_val - first_val) / first_val * 100) if first_val else 0

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(monthly["_month"], monthly[metric_col], color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(monthly["_month"], monthly[metric_col], alpha=0.12, color=PALETTE[1])
    ax.set_title(f"{metric_col.title()} Over Time (Monthly)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"{x:,.2f}"))
    ax.tick_params(axis="x", rotation=30)
    chart_path = _save(fig, "revenue_trend")

    direction = "up" if pct_chg >= 0 else "down"
    finding = (
        f"{metric_col.title()} totaled ${total:,.0f} over the full period. "
        f"The peak month was {peak_row['_month'].strftime('%B %Y')} at ${peak_row[metric_col]:,.0f}. "
        f"Overall the trend moved {direction} {abs(pct_chg):.1f}% from the first to the last recorded month."
    )
    return {"title": f"{metric_col.title()} Trend Over Time", "finding": finding, "chart_path": chart_path}


# ── 2. Top/bottom performers by category ─────────────────────────────────────
def category_breakdown(df: pd.DataFrame, summary: dict) -> dict:
    """
    Horizontal bar chart of total metric by best categorical column.
    """
    cat_cols = summary["categorical_columns"]
    num_cols = summary["numeric_columns"]

    if not cat_cols or not num_cols:
        return {"title": "Category Breakdown", "finding": "No categorical or numeric columns for breakdown.", "chart_path": None}

    metric_col = _pick_metric(num_cols, ["sales", "revenue", "amount", "profit", "total"])
    group_col  = _pick_category(cat_cols, ["category", "segment", "region", "department", "product", "type"])

    temp = df[[group_col, metric_col]].dropna()
    grouped = temp.groupby(group_col)[metric_col].sum().sort_values(ascending=True)

    top    = grouped.idxmax()
    bottom = grouped.idxmin()
    top_v  = grouped.max()
    bot_v  = grouped.min()
    gap_pct = ((top_v - bot_v) / bot_v * 100) if bot_v else 0

    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.55)))
    bars = ax.barh(grouped.index, grouped.values, color=PALETTE[1], edgecolor="white")
    # Highlight top bar
    bars[-1].set_color(PALETTE[0])
    ax.set_title(f"{metric_col.title()} by {group_col.title()}", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"{x:.2f}"))
    chart_path = _save(fig, "category_breakdown")

    finding = (
        f"By {group_col}, '{top}' led with ${top_v:,.0f} in {metric_col}, "
        f"while '{bottom}' was lowest at ${bot_v:,.0f} — a {gap_pct:.0f}% gap between best and worst performers."
    )
    return {"title": f"{metric_col.title()} by {group_col.title()}", "finding": finding, "chart_path": chart_path}


# ── 3. Profit / secondary metric distribution ─────────────────────────────────
def profit_distribution(df: pd.DataFrame, summary: dict) -> dict:
    """
    Histogram of a secondary numeric metric (profit, margin, score, etc.)
    """
    num_cols = summary["numeric_columns"]
    if not num_cols:
        return {"title": "Distribution", "finding": "No numeric columns for distribution analysis.", "chart_path": None}

    metric_col = _pick_metric(num_cols, ["profit", "margin", "score", "discount", "quantity", "units"])

    data = df[metric_col].dropna()
    mean_v   = data.mean()
    median_v = data.median()
    pct_neg  = (data < 0).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(data, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(mean_v,   color=PALETTE[0], linewidth=2,   linestyle="--", label=f"Mean: {mean_v:,.2f}")
    ax.axvline(median_v, color="#E8A020", linewidth=2, linestyle=":",  label=f"Median: {median_v:,.2f}")
    ax.legend(fontsize=9)
    ax.set_title(f"{metric_col.title()} Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(metric_col.title())
    ax.set_ylabel("Count")
    chart_path = _save(fig, "profit_distribution")

    skew_desc = "right-skewed (a few large values pulling the mean up)" if mean_v > median_v else \
                "left-skewed (a few large losses pulling the mean down)" if mean_v < median_v else "symmetric"
    neg_note  = f" {pct_neg:.1f}% of records show negative {metric_col}." if pct_neg > 0 else ""

    finding = (
        f"{metric_col.title()} has a mean of {mean_v:,.2f} and median of {median_v:,.2f}, "
        f"indicating a {skew_desc} distribution.{neg_note}"
    )
    return {"title": f"{metric_col.title()} Distribution", "finding": finding, "chart_path": chart_path}


# ── 4. Correlation heatmap ────────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame, summary: dict) -> dict:
    """
    Heatmap of correlations between numeric columns.
    """
    num_cols = summary["numeric_columns"]
    usable   = [c for c in num_cols if df[c].nunique() > 5]

    if len(usable) < 2:
        return {"title": "Correlation Heatmap", "finding": "Not enough numeric columns for correlation analysis.", "chart_path": None}

    corr = df[usable].corr()

    fig, ax = plt.subplots(figsize=(min(8, len(usable) * 1.4), min(6, len(usable) * 1.2)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("Correlation Between Numeric Variables", fontsize=13, fontweight="bold", pad=12)
    chart_path = _save(fig, "correlation_heatmap")

    import numpy as np
    mask = np.eye(len(corr), dtype=bool)
    corr_unstacked = corr.where(~mask).stack()
    if len(corr_unstacked):
        strongest = corr_unstacked.abs().idxmax()
        strength  = corr_unstacked[strongest]
        finding = (
            f"The strongest relationship is between '{strongest[0]}' and '{strongest[1]}' "
            f"(r = {strength:.2f}), suggesting {'a strong positive' if strength > 0.6 else 'a moderate' if strength > 0.3 else 'a weak'} association."
        )
    else:
        finding = "Correlation matrix computed across all numeric variables."

    return {"title": "Numeric Correlations", "finding": finding, "chart_path": chart_path}


# ── Runner: execute all four analyses ────────────────────────────────────────
def run_all(df: pd.DataFrame, summary: dict) -> list[dict]:
    analyses = [
        revenue_trend(df, summary),
        category_breakdown(df, summary),
        profit_distribution(df, summary),
        correlation_heatmap(df, summary),
    ]
    return [a for a in analyses if a["finding"]]  # drop any that fully failed


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pick_metric(cols: list[str], keywords: list[str]) -> str:
    for kw in keywords:
        for col in cols:
            if kw in col.lower():
                return col
    return cols[0]


def _pick_category(cols: list[str], keywords: list[str]) -> str:
    for kw in keywords:
        for col in cols:
            if kw in col.lower():
                return col
    # Fall back to col with reasonable cardinality
    for col in cols:
        n = len(set(col))
        if 2 <= n <= 30:
            return col
    return cols[0]
