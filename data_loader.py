"""
data_loader.py
--------------
Handles CSV ingestion, cleaning, and producing a structured summary
that gets passed downstream to the EDA and Claude layers.
"""

import pandas as pd
import io


def load_and_clean(file) -> tuple[pd.DataFrame, dict]:
    """
    Accept a file path (str) or file-like object (Streamlit UploadedFile).
    Returns (cleaned_df, summary_dict).
    """
    # --- Load ---
    if isinstance(file, str):
        df = pd.read_csv(file, encoding="latin-1")
    else:
        df = pd.read_csv(io.BytesIO(file.read()), encoding="latin-1")

    original_shape = df.shape

    # --- Basic cleaning ---
    # Drop fully empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Auto-detect and parse date columns
    date_col = _detect_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Re-run date detection now that column is parsed
        date_col = _detect_date_column(df)

    # Auto-detect numeric columns (sometimes loaded as strings due to $ or ,)
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.replace(r"[\$,]", "", regex=True)
        try:
            converted = pd.to_numeric(cleaned, errors="raise")
            df[col] = converted
        except (ValueError, AttributeError):
            pass

    # --- Summary dict ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    null_counts = df.isnull().sum()
    missing_pct = (null_counts / len(df) * 100).round(2).to_dict() if len(df) else {}
    summary = {
        "original_rows": original_shape[0],
        "original_cols": original_shape[1],
        "cleaned_rows": len(df),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "date_column": date_col,
        "null_counts": null_counts.to_dict(),
        "missing_pct": missing_pct,
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df[numeric_cols].describe().to_dict() if numeric_cols else {},
    }

    return df, summary


def _detect_date_column(df: pd.DataFrame) -> str | None:
    """Heuristically find the most likely date column."""
    # First: already-parsed datetime columns
    for col in df.select_dtypes(include=["datetime64"]).columns:
        return col

    date_keywords = ["date", "order", "ship", "time", "period", "month", "year"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in date_keywords):
            sample = df[col].dropna().head(10).astype(str)
            try:
                pd.to_datetime(sample, errors="raise")
                return col
            except Exception:
                continue
    return None


def describe_dataset(df: pd.DataFrame, summary: dict) -> str:
    """
    Returns a plain-English description of the dataset shape and columns.
    This gets included in the Claude prompt context.
    """
    lines = [
        f"Dataset: {summary['cleaned_rows']} rows × {summary['original_cols']} columns.",
        f"Numeric columns: {', '.join(summary['numeric_columns']) or 'none detected'}.",
        f"Categorical columns: {', '.join(summary['categorical_columns']) or 'none detected'}.",
    ]
    if summary["date_column"]:
        min_d = df[summary["date_column"]].min()
        max_d = df[summary["date_column"]].max()
        lines.append(f"Date range: {min_d.date()} to {max_d.date()}.")
    return " ".join(lines)
