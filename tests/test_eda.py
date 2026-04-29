import io

from data_loader import load_and_clean
from eda import build_effective_summary, run_all


def _sales_csv():
    rows = []
    for i in range(12):
        rows.append(f"2023-{i+1:02d}-15,East,CatA,{100 + i * 10},{20 + i}")
    for i in range(12):
        rows.append(f"2023-{i+1:02d}-15,West,CatB,{50 + i * 5},{5 + i}")
    header = "order_date,region,category,sales,profit\n"
    return header + "\n".join(rows)


def test_run_all_produces_analyses():
    df, summary = load_and_clean(io.BytesIO(_sales_csv().encode("utf-8")))
    analyses = run_all(df, summary)
    assert len(analyses) >= 1
    first = analyses[0]
    assert "title" in first and "finding" in first
    assert "plotly_fig" in first or first.get("plotly_fig") is None


def test_column_overrides_metric():
    df, summary = load_and_clean(io.BytesIO(_sales_csv().encode("utf-8")))
    eff = build_effective_summary(df, summary, {"metric_column": "profit"})
    assert eff.get("eda_metric") == "profit"


def test_run_all_respects_override():
    df, summary = load_and_clean(io.BytesIO(_sales_csv().encode("utf-8")))
    analyses = run_all(df, summary, overrides={"metric_column": "profit"})
    titles = " ".join(a["title"] for a in analyses)
    assert "Profit" in titles or "profit" in titles.lower()
