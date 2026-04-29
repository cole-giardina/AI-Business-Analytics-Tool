"""
ai_narrative.py
---------------
Claude API: executive memo, structured JSON insights, and natural-language Q&A
grounded in provided stats (no invented numbers).
"""

import json
import re

import anthropic

MODEL = "claude-sonnet-4-20250514"

TONE_INSTRUCTIONS = {
    "executive": (
        "Write for a busy executive: concise, decision-oriented, minimal jargon. "
        "Focus on commercial impact and tradeoffs."
    ),
    "board": (
        "Write for a board audience: strategic framing, governance and risk awareness, "
        "balanced tone, suitable for high-level oversight."
    ),
    "operations": (
        "Write for operations leaders: practical, process-focused, emphasize execution "
        "and measurable next steps."
    ),
}


def generate_memo(dataset_description: str, findings: list[dict], api_key: str, tone: str = "executive") -> str:
    findings_text = "\n".join(f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding"))
    tone_line = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["executive"])

    prompt = f"""You are a senior business analyst. {tone_line}

Based on the dataset overview and findings below, write a concise executive memo (3–4 paragraphs, no bullet points) that:
1. Opens with a one-sentence summary of what the data covers
2. Highlights the two or three most important insights a business leader should act on
3. Closes with one concrete recommendation

Dataset overview:
{dataset_description}

Key findings from automated analysis:
{findings_text}

Write the memo in a professional but direct tone. Do not use headers or bullet points — flowing paragraphs only. Do not repeat the raw numbers verbatim; interpret them for a business audience."""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_memo_streaming(
    dataset_description: str,
    findings: list[dict],
    api_key: str,
    tone: str = "executive",
):
    findings_text = "\n".join(f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding"))
    tone_line = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["executive"])

    prompt = f"""You are a senior business analyst. {tone_line}

Based on the dataset overview and findings below, write a concise executive memo (3–4 paragraphs, no bullet points) that:
1. Opens with a one-sentence summary of what the data covers
2. Highlights the two or three most important insights a business leader should act on
3. Closes with one concrete recommendation

Dataset overview:
{dataset_description}

Key findings from automated analysis:
{findings_text}

Write the memo in a professional but direct tone. Do not use headers or bullet points — flowing paragraphs only. Do not repeat the raw numbers verbatim; interpret them for a business audience."""

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(model=MODEL, max_tokens=1000, messages=[{"role": "user", "content": prompt}]) as stream:
        for text in stream.text_stream:
            yield text


def generate_structured_insights(
    dataset_description: str,
    findings: list[dict],
    api_key: str,
) -> dict:
    """
    Returns a dict with keys: risks, opportunities, kpis (each a list of short strings).
    """
    findings_text = "\n".join(f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding"))

    prompt = f"""You are a senior analyst. Given ONLY the dataset overview and findings below, respond with a single JSON object and nothing else (no markdown fences). Use this exact schema:
{{"risks": ["..."], "opportunities": ["..."], "kpis": ["label — brief target or watch metric"]}}
Rules:
- Each array must have 2–4 short strings (under 120 chars each).
- Base every item strictly on the provided text; do not invent statistics not implied by the findings.
- kpis should be actionable monitoring ideas tied to the data.

Dataset overview:
{dataset_description}

Findings:
{findings_text}"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    data = json.loads(raw)
    for key in ("risks", "opportunities", "kpis"):
        if key not in data:
            data[key] = []
        if not isinstance(data[key], list):
            data[key] = [str(data[key])]
    return data


def answer_question(
    dataset_description: str,
    findings: list[dict],
    sample_csv: str,
    question: str,
    api_key: str,
) -> str:
    """
    Answer using only the supplied context. Instructs the model not to fabricate numbers.
    """
    findings_text = "\n".join(f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding"))

    prompt = f"""You answer questions about a dataset using ONLY the information below.

Rules:
- If the answer is not supported by the overview, findings, or sample rows, say what is missing and suggest what analysis would be needed.
- Do not invent specific numbers, dates, or categories not present in the context.
- Be concise (2–6 short paragraphs or bullet points as appropriate).

Dataset overview:
{dataset_description}

Automated findings:
{findings_text}

Sample rows (CSV):
{sample_csv}

Question:
{question}"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def answer_question_streaming(
    dataset_description: str,
    findings: list[dict],
    sample_csv: str,
    question: str,
    api_key: str,
):
    findings_text = "\n".join(f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding"))

    prompt = f"""You answer questions about a dataset using ONLY the information below.

Rules:
- If the answer is not supported by the overview, findings, or sample rows, say what is missing and suggest what analysis would be needed.
- Do not invent specific numbers, dates, or categories not present in the context.
- Be concise (2–6 short paragraphs or bullet points as appropriate).

Dataset overview:
{dataset_description}

Automated findings:
{findings_text}

Sample rows (CSV):
{sample_csv}

Question:
{question}"""

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(model=MODEL, max_tokens=1200, messages=[{"role": "user", "content": prompt}]) as stream:
        for text in stream.text_stream:
            yield text
