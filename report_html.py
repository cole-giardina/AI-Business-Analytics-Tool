"""
report_html.py
--------------
Build a standalone HTML report from EDA analyses (embedded PNGs) and memo text.
"""

from __future__ import annotations

import base64
import html
import os
from typing import Any


def _img_data_uri(chart_path: str | None) -> str | None:
    if not chart_path or not os.path.isfile(chart_path):
        return None
    with open(chart_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_html_report(
    title: str,
    dataset_description: str,
    analyses: list[dict[str, Any]],
    memo_text: str,
    structured_insights: dict[str, list[str]] | None = None,
) -> str:
    """Return self-contained HTML document as a string."""
    sections = []
    sections.append(f"<h1>{html.escape(title)}</h1>")
    sections.append("<h2>Dataset</h2>")
    sections.append(f"<p>{html.escape(dataset_description)}</p>")

    sections.append("<h2>Charts</h2>")
    for a in analyses:
        t = html.escape(a.get("title", "Chart"))
        sections.append(f"<h3>{t}</h3>")
        uri = _img_data_uri(a.get("chart_path"))
        if uri:
            sections.append(
                f'<p><img src="{uri}" alt="{t}" style="max-width:100%;height:auto;border:1px solid #d0e4f7;border-radius:8px;" /></p>'
            )
        finding = a.get("finding") or ""
        sections.append(f'<p class="finding">{html.escape(finding)}</p>')

    if structured_insights:
        sections.append("<h2>Structured insights</h2>")
        for label, key in (("Risks", "risks"), ("Opportunities", "opportunities"), ("KPIs to watch", "kpis")):
            items = structured_insights.get(key) or []
            if not items:
                continue
            sections.append(f"<h3>{html.escape(label)}</h3><ul>")
            for item in items:
                sections.append(f"<li>{html.escape(str(item))}</li>")
            sections.append("</ul>")

    sections.append("<h2>Executive memo</h2>")
    memo_html = html.escape(memo_text).replace("\n\n", "</p><p>").replace("\n", "<br/>")
    sections.append(f'<div class="memo"><p>{memo_html}</p></div>')

    body_inner = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
      background: #f8fafc; color: #1a1a2e; }}
    h1 {{ color: #1F4E79; }}
    h2, h3 {{ color: #2E75B6; }}
    .finding {{ background: #EBF4FF; border-left: 4px solid #2E75B6; padding: 12px 16px;
      border-radius: 4px; font-size: 0.95rem; }}
    .memo {{ background: #fff; border: 1px solid #d0e4f7; border-radius: 8px; padding: 24px;
      line-height: 1.75; box-shadow: 0 2px 8px rgba(31,78,121,0.06); }}
  </style>
</head>
<body>
{body_inner}
</body>
</html>
"""


def write_html_report(
    out_path: str,
    title: str,
    dataset_description: str,
    analyses: list[dict[str, Any]],
    memo_text: str,
    structured_insights: dict[str, list[str]] | None = None,
) -> None:
    """Write a single self-contained HTML file."""
    doc = render_html_report(title, dataset_description, analyses, memo_text, structured_insights)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
