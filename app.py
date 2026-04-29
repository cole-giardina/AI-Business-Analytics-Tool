"""
app.py
------
Streamlit UI — upload a CSV, run EDA, generate an AI executive memo.
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from data_loader import load_and_clean, describe_dataset
from eda import run_all
from ai_narrative import generate_memo_streaming

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Business Analytics",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stApp > header { background: transparent; }
    h1 { color: #1F4E79; }
    h2, h3 { color: #2E75B6; }
    .finding-box {
        background: #EBF4FF;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0 16px 0;
        font-size: 0.92rem;
        color: #1a1a2e;
    }
    .memo-box {
        background: white;
        border: 1px solid #d0e4f7;
        border-radius: 8px;
        padding: 24px 28px;
        font-size: 1rem;
        line-height: 1.75;
        color: #1a1a2e;
        box-shadow: 0 2px 8px rgba(31,78,121,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 AI Business Analytics Tool")
st.markdown("Upload a CSV dataset → get automated EDA charts → receive an AI-generated executive memo.")
st.divider()

# ── Sidebar: API key + upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Get yours at console.anthropic.com",
        placeholder="sk-ant-..."
    )
    st.markdown("---")
    st.markdown("**Try the Superstore dataset:**")
    st.markdown("[Download from Kaggle →](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)")
    st.markdown("---")
    st.caption("Built with Python · Pandas · Seaborn · Claude API · Streamlit")

# ── Main: file upload ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded:
    st.info("👆 Upload a CSV to get started. Any business dataset works — sales, HR, finance, e-commerce.")
    st.stop()

# ── Load & clean ──────────────────────────────────────────────────────────────
with st.spinner("Loading and cleaning data..."):
    try:
        df, summary = load_and_clean(uploaded)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{summary['cleaned_rows']:,}")
col2.metric("Columns", summary["original_cols"])
col3.metric("Date Range", 
    f"{df[summary['date_column']].min().date()} – {df[summary['date_column']].max().date()}"
    if summary["date_column"] else "No date column")

with st.expander("Preview cleaned data"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ── EDA ───────────────────────────────────────────────────────────────────────
st.subheader("🔍 Automated Analysis")

with st.spinner("Running EDA..."):
    analyses = run_all(df, summary)

if not analyses:
    st.warning("Could not generate analyses — check that your CSV has numeric columns.")
    st.stop()

dataset_desc = describe_dataset(df, summary)

# Display charts + findings in a 2-col grid
pairs = [analyses[i:i+2] for i in range(0, len(analyses), 2)]
for pair in pairs:
    cols = st.columns(len(pair))
    for col, analysis in zip(cols, pair):
        with col:
            st.markdown(f"**{analysis['title']}**")
            if analysis["chart_path"] and os.path.exists(analysis["chart_path"]):
                st.image(analysis["chart_path"], use_container_width=True)
            st.markdown(f'<div class="finding-box">{analysis["finding"]}</div>', unsafe_allow_html=True)

st.divider()

# ── AI Memo ───────────────────────────────────────────────────────────────────
st.subheader("🤖 AI Executive Memo")

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to generate the executive memo.")
    st.stop()

if st.button("Generate Executive Memo", type="primary", use_container_width=True):
    memo_placeholder = st.empty()
    full_memo = ""

    with st.spinner("Claude is writing your memo..."):
        try:
            memo_box = st.empty()
            full_memo_lines = []
            for chunk in generate_memo_streaming(dataset_desc, analyses, api_key):
                full_memo_lines.append(chunk)
                full_memo = "".join(full_memo_lines)
                memo_box.markdown(f'<div class="memo-box">{full_memo}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Claude API error: {e}")
            st.stop()

    st.divider()
    st.download_button(
        label="⬇️ Download Memo (.txt)",
        data=full_memo,
        file_name="executive_memo.txt",
        mime="text/plain",
    )
