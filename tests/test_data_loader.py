import io

from data_loader import describe_dataset, load_and_clean


def test_load_and_clean_basic():
    csv = "order_date,region,sales,profit\n2023-01-01,East,100,10\n2023-01-02,West,200,5\n"
    df, summary = load_and_clean(io.BytesIO(csv.encode("utf-8")))

    assert len(df) == 2
    assert "sales" in summary["numeric_columns"]
    assert summary.get("date_column")
    assert summary.get("duplicate_rows", 0) == 0
    assert "missing_pct" in summary
    assert summary.get("date_column") == "order_date"


def test_describe_dataset():
    csv = "x,y\n1,a\n2,b\n"
    df, summary = load_and_clean(io.BytesIO(csv.encode("utf-8")))
    text = describe_dataset(df, summary)
    assert "rows" in text.lower() or "2" in text
