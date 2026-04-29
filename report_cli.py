#!/usr/bin/env python3
"""
Batch report generator — load CSV, run EDA, optional Claude memo + HTML export.

Usage:
  python report_cli.py path/to/data.csv --out ./reports/run1
  python report_cli.py data.csv --out ./out --memo   # requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from ai_narrative import generate_memo, generate_structured_insights
from data_loader import describe_dataset, load_and_clean
from eda import run_all
from report_html import write_html_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate charts + optional AI memo and HTML report from a CSV.")
    p.add_argument("csv_path", help="Path to input CSV file")
    p.add_argument("--out", required=True, help="Output directory for report artifacts")
    p.add_argument("--memo", action="store_true", help="Generate executive memo via Claude (needs API key)")
    p.add_argument("--insights", action="store_true", help="Also generate structured JSON insights (implies API call)")
    p.add_argument(
        "--chart-prefix",
        default="",
        help="Prefix for PNG filenames in charts/ (default: derived from CSV basename)",
    )
    args = p.parse_args(argv)

    csv_path = args.csv_path
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    chart_prefix = args.chart_prefix
    if not chart_prefix:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        chart_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) + "_"

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "rb") as f:
        raw = f.read()
    df, summary = load_and_clean(io.BytesIO(raw))

    analyses = run_all(df, summary, overrides=None, chart_prefix=chart_prefix)
    dataset_desc = describe_dataset(df, summary)

    memo_text = ""
    insights: dict | None = None
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if args.memo or args.insights:
        if not api_key:
            print("ANTHROPIC_API_KEY is not set; skipping memo/insights.", file=sys.stderr)
        else:
            if args.memo:
                memo_text = generate_memo(dataset_desc, analyses, api_key, tone="executive")
                memo_path = os.path.join(out_dir, "executive_memo.txt")
                with open(memo_path, "w", encoding="utf-8") as mf:
                    mf.write(memo_text)
                print(f"Wrote {memo_path}")
            if args.insights:
                insights = generate_structured_insights(dataset_desc, analyses, api_key)
                ipath = os.path.join(out_dir, "structured_insights.json")
                with open(ipath, "w", encoding="utf-8") as jf:
                    json.dump(insights, jf, indent=2)
                print(f"Wrote {ipath}")

    html_path = os.path.join(out_dir, "report.html")
    title = os.path.basename(csv_path)
    write_html_report(
        html_path,
        title=f"Analytics report — {title}",
        dataset_description=dataset_desc,
        analyses=analyses,
        memo_text=memo_text or "(Memo not generated — run with --memo and ANTHROPIC_API_KEY)",
        structured_insights=insights,
    )
    print(f"Wrote {html_path}")

    meta = {
        "csv": csv_path,
        "rows": summary["cleaned_rows"],
        "chart_prefix": chart_prefix,
        "analyses": len(analyses),
    }
    with open(os.path.join(out_dir, "report_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
