"""
app.py
------
Streamlit UI — tabbed flow: Upload, Explore, AI, Export.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import html as html_module
import io
import json
import os

import pandas as pd
import streamlit as st

from ai_narrative import (
    answer_question_streaming,
    generate_memo_streaming,
    generate_structured_insights,
)
from data_loader import describe_dataset, load_and_clean
from eda import run_all
from report_html import render_html_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Design tokens + components ────────────────────────────────────────────────
st.markdown(
    """
<style>
    :root {
        --brand-primary: #1F4E79;
        --brand-secondary: #2E75B6;
        --brand-muted: #5BA3D9;
        --surface-page: #f8fafc;
        --surface-card: #ffffff;
        --surface-accent: #EBF4FF;
        --border-default: #d0e4f7;
        --border-strong: #a8c8e8;
        --text-body: #1a1a2e;
        --text-soft: #475569;
        --shadow-card: 0 2px 8px rgba(31, 78, 121, 0.06);
        --radius-sm: 6px;
        --radius-md: 10px;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --font-ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    }
    .main { background-color: var(--surface-page); font-family: var(--font-ui); }
    .stApp > header { background: transparent; }
    h1 { color: var(--brand-primary); font-weight: 650; letter-spacing: -0.02em; }
    h2, h3 { color: var(--brand-secondary); font-weight: 600; }
    .app-lede {
        font-size: 1.05rem;
        color: var(--text-soft);
        max-width: 42rem;
        line-height: 1.5;
        margin: 0 0 var(--space-md) 0;
    }
    .tab-hint {
        font-size: 0.9rem;
        color: var(--text-soft);
        margin-bottom: var(--space-md);
        max-width: 40rem;
    }
    .empty-panel {
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-md);
        padding: var(--space-lg);
        background: var(--surface-card);
        color: var(--text-soft);
        max-width: 36rem;
    }
    .chart-card {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: var(--space-md) var(--space-md) var(--space-lg);
        margin-bottom: var(--space-lg);
        background: var(--surface-card);
        box-shadow: var(--shadow-card);
    }
    .chart-card h4 { margin: 0 0 var(--space-sm) 0; font-size: 1rem; color: var(--brand-primary); }
    .finding-box {
        background: var(--surface-accent);
        border-left: 4px solid var(--brand-secondary);
        padding: 12px 16px;
        border-radius: var(--radius-sm);
        margin: var(--space-sm) 0 0 0;
        font-size: 0.92rem;
        color: var(--text-body);
        line-height: 1.5;
    }
    .memo-box {
        background: var(--surface-card);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: 24px 28px;
        font-size: 1rem;
        line-height: 1.75;
        color: var(--text-body);
        box-shadow: var(--shadow-card);
        max-width: 48rem;
    }
    .insight-card {
        background: var(--surface-card);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .prose-memo p { margin: 0 0 0.75em 0; }
    [data-testid="stTabs"] { margin-top: var(--space-md); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str):
    return load_and_clean(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Building charts and summary stats…")
def cached_run_all(file_bytes: bytes, filename: str, overrides_key: str):
    df, summary = cached_load(file_bytes, filename)
    o = json.loads(overrides_key)
    return df, summary, run_all(df, summary, overrides=o if o else None)


def resolve_api_key(sidebar_key: str) -> str:
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return sidebar_key.strip() or env_key


def _clear_ai_session_for_new_file(upload_name: str | None) -> None:
    if upload_name is None:
        return
    prev = st.session_state.get("_uploaded_file_key")
    if prev != upload_name:
        st.session_state["_uploaded_file_key"] = upload_name
        st.session_state.pop("last_memo", None)
        st.session_state.pop("last_insights", None)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("AI Business Analytics")
st.markdown(
    '<p class="app-lede">Upload business data, explore automated charts, then generate AI summaries and exports.</p>',
    unsafe_allow_html=True,
)

# ── Sidebar: Setup (always) ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")
    env_hint = "Use **ANTHROPIC_API_KEY** in the environment, or paste a key below."
    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        help=f"{env_hint} Keys: console.anthropic.com",
        placeholder="sk-ant-…",
    )
    st.caption(env_hint)
    st.divider()
    st.subheader("Sample data")
    st.markdown("[Superstore on Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)")
    st.divider()
    st.caption("Python · Pandas · Plotly · Claude · Streamlit")

tab_upload, tab_explore, tab_ai, tab_export = st.tabs(["Upload", "Explore", "AI", "Export"])

uploaded = None
with tab_upload:
    st.markdown('<p class="tab-hint">Step 1 — Choose a CSV. Sales, HR, finance, or operations data all work.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        st.markdown(
            '<div class="empty-panel">No file yet. Pick a <code>.csv</code> above, then open '
            '<strong>Explore</strong> to see metrics and charts.</div>',
            unsafe_allow_html=True,
        )

# Resolve upload / load
file_bytes: bytes = b""
filename = ""
df: pd.DataFrame | None = None
summary: dict | None = None
load_error: str | None = None

if uploaded is not None:
    _clear_ai_session_for_new_file(uploaded.name)
    file_bytes = uploaded.getvalue()
    filename = uploaded.name
    try:
        df, summary = cached_load(file_bytes, filename)
    except Exception as e:
        load_error = str(e)

# ── Sidebar: Column mapping (only when data loaded) ───────────────────────────
overrides: dict[str, str | None] = {}
if df is not None and summary is not None and load_error is None:
    num_cols = summary.get("numeric_columns") or []
    cat_cols = summary.get("categorical_columns") or []
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_candidates = list(
        dict.fromkeys(
            ([summary["date_column"]] if summary.get("date_column") else []) + datetime_cols
        )
    )

    with st.sidebar:
        st.subheader("Column mapping")
        st.caption("Override auto-detected columns for all charts.")
        date_opts = ["Auto"] + [c for c in date_candidates if c in df.columns]
        date_choice = st.selectbox("Date column", date_opts, index=0)
        metric_choice = st.selectbox("Primary metric", ["Auto"] + num_cols, index=0)
        cat_choice = st.selectbox("Breakdown dimension", ["Auto"] + cat_cols, index=0)
        dist_choice = st.selectbox("Distribution metric", ["Auto"] + num_cols, index=0)

        if date_choice != "Auto":
            overrides["date_column"] = date_choice
        if metric_choice != "Auto":
            overrides["metric_column"] = metric_choice
        if cat_choice != "Auto":
            overrides["category_column"] = cat_choice
        if dist_choice != "Auto":
            overrides["distribution_column"] = dist_choice
elif uploaded is not None and load_error is None:
    with st.sidebar:
        st.subheader("Column mapping")
        st.caption("Available after the file loads successfully.")
elif uploaded is None:
    with st.sidebar:
        st.subheader("Column mapping")
        st.caption("Upload a CSV in the **Upload** tab to configure columns.")
elif load_error:
    with st.sidebar:
        st.subheader("Column mapping")
        st.caption("Fix the error shown in **Explore** before mapping columns.")

overrides_key = json.dumps(overrides, sort_keys=True)

analyses: list = []
dataset_desc = ""

if df is not None and summary is not None and load_error is None and file_bytes:
    try:
        _df_cached, summary_cached, analyses = cached_run_all(file_bytes, filename, overrides_key)
        df = _df_cached
        summary = summary_cached
        dataset_desc = describe_dataset(df, summary)
    except Exception as e:
        load_error = str(e)
        analyses = []

api_key = resolve_api_key(api_key_input)

# ── Explore tab ────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown(
        '<p class="tab-hint">Step 2 — Review data quality and automated charts.</p>',
        unsafe_allow_html=True,
    )
    if uploaded is None:
        st.markdown(
            '<div class="empty-panel">Upload a CSV on the <strong>Upload</strong> tab first.</div>',
            unsafe_allow_html=True,
        )
    elif load_error:
        st.error(f"Could not load file: {load_error}")
    elif df is not None and summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{summary['cleaned_rows']:,}")
        col2.metric("Columns", summary["original_cols"])
        col3.metric(
            "Date range",
            (
                f"{df[summary['date_column']].min().date()} – {df[summary['date_column']].max().date()}"
                if summary.get("date_column")
                else "No date column"
            ),
        )
        col4.metric("Duplicate rows", f"{summary.get('duplicate_rows', 0):,}")

        with st.expander("Data quality"):
            miss = summary.get("missing_pct") or {}
            if miss:
                miss_df = (
                    pd.Series(miss)
                    .sort_values(ascending=False)
                    .head(25)
                    .rename("% missing")
                    .reset_index()
                )
                miss_df.columns = ["Column", "% missing"]
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            else:
                st.write("No missing-value summary.")

        with st.expander("Preview cleaned data"):
            st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Automated analysis")

        if not analyses:
            st.warning(
                "Could not build charts — ensure the CSV has numeric columns (and categories where needed)."
            )
        else:
            pairs = [analyses[i : i + 2] for i in range(0, len(analyses), 2)]
            for pair in pairs:
                cols = st.columns(len(pair))
                for col, analysis in zip(cols, pair):
                    with col:
                        st.markdown(
                            f'<div class="chart-card"><h4>{html_module.escape(analysis["title"])}</h4>',
                            unsafe_allow_html=True,
                        )
                        fig = analysis.get("plotly_fig")
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        elif analysis.get("chart_path") and os.path.exists(analysis["chart_path"]):
                            st.image(analysis["chart_path"], use_container_width=True)
                        st.markdown(
                            f'<div class="finding-box">{html_module.escape(analysis.get("finding", ""))}</div></div>',
                            unsafe_allow_html=True,
                        )

# ── AI tab ───────────────────────────────────────────────────────────────────
with tab_ai:
    st.markdown(
        '<p class="tab-hint">Step 3 — Optional. Requires an API key (sidebar or env). Answers use only loaded context.</p>',
        unsafe_allow_html=True,
    )
    if uploaded is None:
        st.markdown(
            '<div class="empty-panel">Upload a CSV on the <strong>Upload</strong> tab to enable memo, insights, and Q&A.</div>',
            unsafe_allow_html=True,
        )
    elif load_error:
        st.error(f"Fix the load error before using AI: {load_error}")
    elif df is None or not analyses:
        st.markdown(
            '<div class="empty-panel">Charts could not be produced for this file. Check numeric columns in <strong>Explore</strong>.</div>',
            unsafe_allow_html=True,
        )
    else:
        tone = st.radio(
            "Memo tone / audience",
            options=["executive", "board", "operations"],
            horizontal=True,
            help="Adjusts Claude for executive, board, or operations readers.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            gen_memo = st.button("Generate executive memo", type="primary", use_container_width=True)
        with col_b:
            gen_insights = st.button("Generate structured insights", use_container_width=True)

        if not api_key:
            st.warning("Add an API key under **Setup** in the sidebar, or set **ANTHROPIC_API_KEY**.")
        else:
            if gen_memo:
                memo_box = st.empty()
                full_memo = ""
                try:
                    with st.spinner("Claude is writing your memo…"):
                        for chunk in generate_memo_streaming(
                            dataset_desc, analyses, api_key, tone=tone
                        ):
                            full_memo += chunk
                            memo_box.markdown(
                                f'<div class="memo-box">{html_module.escape(full_memo)}</div>',
                                unsafe_allow_html=True,
                            )
                    st.session_state["last_memo"] = full_memo
                except Exception as e:
                    st.error(f"Claude API error: {e}")

            if gen_insights:
                try:
                    with st.spinner("Generating structured insights…"):
                        st.session_state["last_insights"] = generate_structured_insights(
                            dataset_desc, analyses, api_key
                        )
                except Exception as e:
                    st.error(f"Claude API error: {e}")

            if not gen_memo and st.session_state.get("last_memo"):
                st.markdown("**Saved memo**")
                lm = st.session_state["last_memo"]
                st.markdown(
                    f'<div class="memo-box">{html_module.escape(lm)}</div>',
                    unsafe_allow_html=True,
                )

            insights_saved = st.session_state.get("last_insights")
            if insights_saved:
                st.subheader("Structured insights")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown("**Risks**")
                    for item in insights_saved.get("risks") or []:
                        st.markdown(
                            f'<div class="insight-card">{html_module.escape(str(item))}</div>',
                            unsafe_allow_html=True,
                        )
                with r2:
                    st.markdown("**Opportunities**")
                    for item in insights_saved.get("opportunities") or []:
                        st.markdown(
                            f'<div class="insight-card">{html_module.escape(str(item))}</div>',
                            unsafe_allow_html=True,
                        )
                with r3:
                    st.markdown("**KPIs to watch**")
                    for item in insights_saved.get("kpis") or []:
                        st.markdown(
                            f'<div class="insight-card">{html_module.escape(str(item))}</div>',
                            unsafe_allow_html=True,
                        )

        st.divider()
        st.subheader("Ask a question")
        sample_csv = df.head(15).to_csv(index=False)
        q = st.text_area(
            "Question",
            placeholder="e.g. Which findings matter most for inventory planning?",
            height=100,
            label_visibility="collapsed",
        )
        if api_key and analyses:
            if st.button("Get answer"):
                if not q.strip():
                    st.warning("Enter a question.")
                else:
                    ans_box = st.empty()
                    full = ""
                    try:
                        with st.spinner("Thinking…"):
                            for chunk in answer_question_streaming(
                                dataset_desc, analyses, sample_csv, q.strip(), api_key
                            ):
                                full += chunk
                                ans_box.markdown(
                                    f'<div class="memo-box">{html_module.escape(full)}</div>',
                                    unsafe_allow_html=True,
                                )
                    except Exception as e:
                        st.error(f"Claude API error: {e}")
        elif analyses and not api_key:
            st.caption("Add an API key to ask questions.")

# ── Export tab ─────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown(
        '<p class="tab-hint">Step 4 — Download a text memo or a self-contained HTML report with chart images.</p>',
        unsafe_allow_html=True,
    )
    if uploaded is None:
        st.markdown(
            '<div class="empty-panel">Upload a CSV on the <strong>Upload</strong> tab first.</div>',
            unsafe_allow_html=True,
        )
    elif load_error:
        st.error(f"Cannot export until the file loads: {load_error}")
    elif not analyses:
        st.markdown(
            '<div class="empty-panel">Generate charts from <strong>Explore</strong> first; export needs chart assets.</div>',
            unsafe_allow_html=True,
        )
    else:
        memo_for_dl = st.session_state.get("last_memo", "")
        st.markdown(
            "**Memo (.txt)** — Plain text from the last generated executive memo on the **AI** tab."
        )
        st.download_button(
            label="Download memo (.txt)",
            data=memo_for_dl or "",
            file_name="executive_memo.txt",
            mime="text/plain",
            disabled=not memo_for_dl,
            use_container_width=True,
        )
        if not memo_for_dl:
            st.caption("Generate a memo on the **AI** tab to populate this download.")

        st.divider()
        st.markdown(
            "**HTML report** — One file with embedded chart images; optional memo text and structured insights if generated."
        )
        html_doc = render_html_report(
            title=f"Analytics — {filename}",
            dataset_description=dataset_desc,
            analyses=analyses,
            memo_text=memo_for_dl or "(Memo section empty — generate a memo on the AI tab.)",
            structured_insights=st.session_state.get("last_insights"),
        )
        st.download_button(
            label="Download HTML report",
            data=html_doc,
            file_name="analytics_report.html",
            mime="text/html",
            use_container_width=True,
        )
