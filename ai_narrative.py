"""
ai_narrative.py
---------------
Passes structured EDA findings to Claude and returns a polished
executive business memo as plain text.
"""

import anthropic


def generate_memo(dataset_description: str, findings: list[dict], api_key: str) -> str:
    """
    findings: list of { title, finding } dicts from eda.run_all()
    Returns the memo as a plain-text string.
    """
    findings_text = "\n".join(
        f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding")
    )

    prompt = f"""You are a senior business analyst. Based on the dataset overview and findings below, write a concise executive memo (3–4 paragraphs, no bullet points) that:
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
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def generate_memo_streaming(dataset_description: str, findings: list[dict], api_key: str):
    """
    Streaming version — yields text chunks for Streamlit st.write_stream().
    """
    findings_text = "\n".join(
        f"- {f['title']}: {f['finding']}" for f in findings if f.get("finding")
    )

    prompt = f"""You are a senior business analyst. Based on the dataset overview and findings below, write a concise executive memo (3–4 paragraphs, no bullet points) that:
1. Opens with a one-sentence summary of what the data covers
2. Highlights the two or three most important insights a business leader should act on
3. Closes with one concrete recommendation

Dataset overview:
{dataset_description}

Key findings from automated analysis:
{findings_text}

Write the memo in a professional but direct tone. Do not use headers or bullet points — flowing paragraphs only. Do not repeat the raw numbers verbatim; interpret them for a business audience."""

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            yield text
