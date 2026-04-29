# AI Business Analytics Tool

Upload a CSV → get automated EDA charts → receive an AI-generated executive memo.

## Stack
- **Python** + **Pandas** — data loading, cleaning, analysis
- **Matplotlib** / **Seaborn** — chart generation
- **Anthropic Claude API** — natural language executive memo
- **Streamlit** — interactive UI

## Setup

```bash
# 1. Clone / download the project folder
cd analytics_tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Paste your Anthropic API key in the sidebar (get one at console.anthropic.com)
2. Upload any business CSV — the tool auto-detects date, numeric, and categorical columns
3. Review the 4 automated charts and key findings
4. Click **Generate Executive Memo** to get the AI-written memo
5. Download the memo as a .txt file

## Recommended Dataset

**Superstore Sales** from Kaggle:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Has Sales, Profit, Category, Region, and Order Date — enough variety for all 4 analyses.

## Project Structure

```
analytics_tool/
├── app.py            # Streamlit UI
├── data_loader.py    # CSV ingestion + cleaning
├── eda.py            # 4 EDA analysis functions
├── ai_narrative.py   # Claude API integration
├── requirements.txt
├── charts/           # Auto-generated chart PNGs
└── data/             # Put your CSV here (optional)
```

## How It Works

1. `data_loader.py` reads the CSV, strips junk, parses dates, and returns a summary dict
2. `eda.py` runs 4 analyses — trend over time, category breakdown, distribution, correlations — each returning a chart + a plain-English finding sentence
3. `ai_narrative.py` passes those finding sentences to Claude with a structured prompt, and streams back the executive memo
4. `app.py` wires it all together in a Streamlit UI with file upload, chart display, and a download button
