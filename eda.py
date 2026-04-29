"""
eda.py
------
Analysis functions, each returning:
  { "title", "finding", "chart_path", "plotly_fig" }

"finding" is fed to the Claude prompt. "chart_path" is a saved PNG; "plotly_fig" is for Streamlit.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

warnings.filterwarnings("ignore")

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
PALETTE = ["#1F4E79", "#2E75B6", "#5BA3D9", "#A8C8E8", "#D6E8F5"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
)


def _save(fig, name: str, chart_prefix: str = "") -> str:
    safe = f"{chart_prefix}{name}".strip("_")
    path = os.path.join(CHARTS_DIR, f"{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_effective_summary(
    df: pd.DataFrame, summary: dict, overrides: dict[str, str | None] | None
) -> dict:
    """
    Merge user overrides with heuristics. Keys in overrides: date_column, metric_column,
    category_column, distribution_column (each optional).
    """
    o = overrides or {}
    num_cols = list(summary.get("numeric_columns") or [])
    cat_cols = list(summary.get("categorical_columns") or [])

    date_col = o.get("date_column") or summary.get("date_column")
    if date_col and date_col not in df.columns:
        date_col = summary.get("date_column")

    def resolve_numeric(name: str | None, fallback_keywords: list[str]) -> str | None:
        if name and name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
        if num_cols:
            return _pick_metric(num_cols, fallback_keywords)
        return None

    metric_col = resolve_numeric(o.get("metric_column"), ["sales", "revenue", "amount", "total"])
    dist_col = resolve_numeric(
        o.get("distribution_column"), ["profit", "margin", "score", "discount", "quantity", "units"]
    )
    if dist_col is None and num_cols:
        dist_col = _pick_metric(num_cols, ["profit", "margin", "score", "discount", "quantity", "units"])

    group_col = o.get("category_column")
    if group_col and group_col in df.columns:
        if df[group_col].nunique() <= 1:
            group_col = None
    else:
        group_col = None
    if not group_col and cat_cols:
        group_col = _pick_category(df, cat_cols, ["category", "segment", "region", "department", "product", "type"])

    out = {**summary, "date_column": date_col, "eda_metric": metric_col, "eda_category": group_col, "eda_distribution": dist_col}
    return out


# ── 1. Revenue / primary metric trend over time ───────────────────────────────
def revenue_trend(df: pd.DataFrame, s: dict, chart_prefix: str = "") -> dict:
    date_col = s.get("date_column")
    metric_col = s.get("eda_metric")

    if not date_col or not metric_col or metric_col not in df.columns:
        return {
            "title": "Revenue Trend",
            "finding": "No date or numeric column found for trend analysis.",
            "chart_path": None,
            "plotly_fig": None,
        }

    temp = df[[date_col, metric_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(temp[date_col]):
        return {
            "title": "Revenue Trend",
            "finding": "Date column could not be used for trend analysis.",
            "chart_path": None,
            "plotly_fig": None,
        }

    temp["_month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = temp.groupby("_month", as_index=False)[metric_col].sum()

    if len(monthly) < 2:
        return {
            "title": "Revenue Trend",
            "finding": "Not enough date range for a trend.",
            "chart_path": None,
            "plotly_fig": None,
        }

    peak_row = monthly.loc[monthly[metric_col].idxmax()]
    total = monthly[metric_col].sum()
    first_val = monthly[metric_col].iloc[0]
    last_val = monthly[metric_col].iloc[-1]
    pct_chg = ((last_val - first_val) / first_val * 100) if first_val else 0

    fig_mpl, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(monthly["_month"], monthly[metric_col], color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(monthly["_month"], monthly[metric_col], alpha=0.12, color=PALETTE[1])
    ax.set_title(f"{metric_col.title()} Over Time (Monthly)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"{x:,.2f}")
    )
    ax.tick_params(axis="x", rotation=30)
    chart_path = _save(fig_mpl, "revenue_trend", chart_prefix)

    fig_pl = go.Figure()
    fig_pl.add_trace(
        go.Scatter(
            x=monthly["_month"],
            y=monthly[metric_col],
            mode="lines+markers",
            line=dict(color=PALETTE[0], width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(46, 117, 182, 0.12)",
            name=metric_col,
        )
    )
    fig_pl.update_layout(
        title=dict(text=f"{metric_col.title()} Over Time (Monthly)", font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=60),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(tickformat=",.0f"),
        height=380,
    )

    direction = "up" if pct_chg >= 0 else "down"
    finding = (
        f"{metric_col.title()} totaled ${total:,.0f} over the full period. "
        f"The peak month was {peak_row['_month'].strftime('%B %Y')} at ${peak_row[metric_col]:,.0f}. "
        f"Overall the trend moved {direction} {abs(pct_chg):.1f}% from the first to the last recorded month."
    )
    return {
        "title": f"{metric_col.title()} Trend Over Time",
        "finding": finding,
        "chart_path": chart_path,
        "plotly_fig": fig_pl,
    }


# ── 2. Top/bottom performers by category ─────────────────────────────────────
def category_breakdown(df: pd.DataFrame, s: dict, chart_prefix: str = "") -> dict:
    group_col = s.get("eda_category")
    metric_col = s.get("eda_metric")

    if not group_col or not metric_col or group_col not in df.columns or metric_col not in df.columns:
        return {
            "title": "Category Breakdown",
            "finding": "No categorical or numeric columns for breakdown.",
            "chart_path": None,
            "plotly_fig": None,
        }

    temp = df[[group_col, metric_col]].dropna()
    grouped = temp.groupby(group_col)[metric_col].sum().sort_values(ascending=True)

    top = grouped.idxmax()
    bottom = grouped.idxmin()
    top_v = grouped.max()
    bot_v = grouped.min()
    gap_pct = ((top_v - bot_v) / bot_v * 100) if bot_v else 0

    fig_mpl, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.55)))
    bars = ax.barh(grouped.index.astype(str), grouped.values, color=PALETTE[1], edgecolor="white")
    bars[-1].set_color(PALETTE[0])
    ax.set_title(f"{metric_col.title()} by {group_col.title()}", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"{x:.2f}")
    )
    chart_path = _save(fig_mpl, "category_breakdown", chart_prefix)

    fig_pl = go.Figure(
        go.Bar(
            x=grouped.values,
            y=[str(i) for i in grouped.index],
            orientation="h",
            marker_color=[PALETTE[0] if str(i) == str(top) else PALETTE[1] for i in grouped.index],
        )
    )
    fig_pl.update_layout(
        title=dict(text=f"{metric_col.title()} by {group_col.title()}", font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=120, r=20, t=50, b=40),
        height=max(360, len(grouped) * 28),
        xaxis=dict(tickformat=",.0f"),
    )

    finding = (
        f"By {group_col}, '{top}' led with ${top_v:,.0f} in {metric_col}, "
        f"while '{bottom}' was lowest at ${bot_v:,.0f} — a {gap_pct:.0f}% gap between best and worst performers."
    )
    return {
        "title": f"{metric_col.title()} by {group_col.title()}",
        "finding": finding,
        "chart_path": chart_path,
        "plotly_fig": fig_pl,
    }


# ── 3. Distribution ───────────────────────────────────────────────────────────
def profit_distribution(df: pd.DataFrame, s: dict, chart_prefix: str = "") -> dict:
    metric_col = s.get("eda_distribution")
    num_cols = s.get("numeric_columns") or []

    if not metric_col or metric_col not in df.columns:
        return {
            "title": "Distribution",
            "finding": "No numeric columns for distribution analysis.",
            "chart_path": None,
            "plotly_fig": None,
        }

    data = df[metric_col].dropna()
    mean_v = data.mean()
    median_v = data.median()
    pct_neg = (data < 0).mean() * 100

    fig_mpl, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(data, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(mean_v, color=PALETTE[0], linewidth=2, linestyle="--", label=f"Mean: {mean_v:,.2f}")
    ax.axvline(median_v, color="#E8A020", linewidth=2, linestyle=":", label=f"Median: {median_v:,.2f}")
    ax.legend(fontsize=9)
    ax.set_title(f"{metric_col.title()} Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(metric_col.title())
    ax.set_ylabel("Count")
    chart_path = _save(fig_mpl, "profit_distribution", chart_prefix)

    fig_pl = go.Figure()
    fig_pl.add_trace(go.Histogram(x=data, nbinsx=40, marker_color=PALETTE[1], name="Count"))
    fig_pl.add_vline(x=mean_v, line_dash="dash", line_color=PALETTE[0], annotation_text=f"Mean {mean_v:,.2f}")
    fig_pl.add_vline(x=median_v, line_dash="dot", line_color="#E8A020", annotation_text=f"Median {median_v:,.2f}")
    fig_pl.update_layout(
        title=dict(text=f"{metric_col.title()} Distribution", font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        height=380,
        xaxis=dict(title=metric_col.title()),
        yaxis=dict(title="Count"),
    )

    skew_desc = (
        "right-skewed (a few large values pulling the mean up)"
        if mean_v > median_v
        else "left-skewed (a few large losses pulling the mean down)"
        if mean_v < median_v
        else "symmetric"
    )
    neg_note = f" {pct_neg:.1f}% of records show negative {metric_col}." if pct_neg > 0 else ""

    finding = (
        f"{metric_col.title()} has a mean of {mean_v:,.2f} and median of {median_v:,.2f}, "
        f"indicating a {skew_desc} distribution.{neg_note}"
    )
    return {
        "title": f"{metric_col.title()} Distribution",
        "finding": finding,
        "chart_path": chart_path,
        "plotly_fig": fig_pl,
    }


# ── 4. Correlation heatmap ────────────────────────────────────────────────────
def correlation_heatmap(df: pd.DataFrame, s: dict, chart_prefix: str = "") -> dict:
    num_cols = s.get("numeric_columns") or []
    usable = [c for c in num_cols if df[c].nunique() > 5]

    if len(usable) < 2:
        return {
            "title": "Correlation Heatmap",
            "finding": "Not enough numeric columns for correlation analysis.",
            "chart_path": None,
            "plotly_fig": None,
        }

    corr = df[usable].corr()

    fig_mpl, ax = plt.subplots(figsize=(min(8, len(usable) * 1.4), min(6, len(usable) * 1.2)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Between Numeric Variables", fontsize=13, fontweight="bold", pad=12)
    chart_path = _save(fig_mpl, "correlation_heatmap", chart_prefix)

    z_text = np.round(corr.values.astype(float), 2).astype(str).tolist()
    fig_pl = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    fig_pl.update_layout(
        title=dict(text="Correlation Between Numeric Variables", font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=80, r=40, t=50, b=80),
        height=min(520, 120 + len(usable) * 50),
    )

    mask = np.eye(len(corr), dtype=bool)
    corr_unstacked = corr.where(~mask).stack()
    if len(corr_unstacked):
        strongest = corr_unstacked.abs().idxmax()
        strength = corr_unstacked[strongest]
        finding = (
            f"The strongest relationship is between '{strongest[0]}' and '{strongest[1]}' "
            f"(r = {strength:.2f}), suggesting {'a strong positive' if strength > 0.6 else 'a moderate' if strength > 0.3 else 'a weak'} association."
        )
    else:
        finding = "Correlation matrix computed across all numeric variables."

    return {
        "title": "Numeric Correlations",
        "finding": finding,
        "chart_path": chart_path,
        "plotly_fig": fig_pl,
    }


# ── 5. Pareto (cumulative contribution) ───────────────────────────────────────
def pareto_chart(df: pd.DataFrame, s: dict, chart_prefix: str = "") -> dict:
    group_col = s.get("eda_category")
    metric_col = s.get("eda_metric")

    if not group_col or not metric_col or group_col not in df.columns or metric_col not in df.columns:
        return {
            "title": "Pareto Analysis",
            "finding": "Need both a category and numeric metric for Pareto analysis.",
            "chart_path": None,
            "plotly_fig": None,
        }

    temp = df[[group_col, metric_col]].dropna()
    totals = temp.groupby(group_col)[metric_col].sum().sort_values(ascending=False)
    top_n = min(20, len(totals))
    s_slice = totals.head(top_n)
    cum_pct = (s_slice.cumsum() / s_slice.sum() * 100) if s_slice.sum() else s_slice * 0
    total_all = totals.sum()
    share_top = (s_slice.iloc[0] / total_all * 100) if total_all else 0

    fig_mpl, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(len(s_slice)), s_slice.values, color=PALETTE[1], edgecolor="white")
    ax1.set_xticks(range(len(s_slice)))
    ax1.set_xticklabels([str(x) for x in s_slice.index], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(metric_col.title())
    ax1.set_title(f"Pareto: {metric_col.title()} by {group_col.title()} (top {top_n})", fontsize=13, fontweight="bold", pad=12)

    ax2 = ax1.twinx()
    ax2.plot(range(len(s_slice)), cum_pct.values, color=PALETTE[0], marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative % of shown categories")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    chart_path = _save(fig_mpl, "pareto", chart_prefix)

    fig_pl = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pl.add_trace(
        go.Bar(x=[str(x) for x in s_slice.index], y=s_slice.values, name=metric_col, marker_color=PALETTE[1]),
        secondary_y=False,
    )
    fig_pl.add_trace(
        go.Scatter(
            x=[str(x) for x in s_slice.index],
            y=cum_pct.values,
            mode="lines+markers",
            name="Cumulative %",
            line=dict(color=PALETTE[0], width=2),
        ),
        secondary_y=True,
    )
    fig_pl.update_xaxes(tickangle=45)
    fig_pl.update_layout(
        title=dict(
            text=f"Pareto: {metric_col.title()} by {group_col.title()} (top {top_n})",
            font=dict(size=14),
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=120),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_pl.update_yaxes(title_text=metric_col.title(), secondary_y=False)
    fig_pl.update_yaxes(title_text="Cumulative % (of top categories)", range=[0, 105], secondary_y=True)

    finding = (
        f"Pareto view of {metric_col} by {group_col}: the largest segment '{s_slice.index[0]}' accounts for "
        f"about {share_top:.1f}% of the top-{top_n} total; cumulative share crosses most of the concentration in the ranked bars."
    )
    return {
        "title": f"Pareto: {metric_col.title()} by {group_col.title()}",
        "finding": finding,
        "chart_path": chart_path,
        "plotly_fig": fig_pl,
    }


# ── Runner ────────────────────────────────────────────────────────────────────
def run_all(
    df: pd.DataFrame,
    summary: dict,
    overrides: dict[str, str | None] | None = None,
    chart_prefix: str = "",
) -> list[dict[str, Any]]:
    eff = build_effective_summary(df, summary, overrides)
    analyses = [
        revenue_trend(df, eff, chart_prefix),
        category_breakdown(df, eff, chart_prefix),
        profit_distribution(df, eff, chart_prefix),
        correlation_heatmap(df, eff, chart_prefix),
        pareto_chart(df, eff, chart_prefix),
    ]
    return [a for a in analyses if a.get("finding")]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pick_metric(cols: list[str], keywords: list[str]) -> str:
    for kw in keywords:
        for col in cols:
            if kw in col.lower():
                return col
    return cols[0]


def _pick_category(df: pd.DataFrame, cols: list[str], keywords: list[str]) -> str:
    for kw in keywords:
        for col in cols:
            if kw in col.lower():
                return col
    for col in cols:
        n = df[col].nunique()
        if 2 <= n <= 30:
            return col
    return cols[0]
