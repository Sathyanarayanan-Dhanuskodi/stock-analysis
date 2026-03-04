import os
from dotenv import load_dotenv

load_dotenv()


def is_sentiment_available() -> bool:
    """Check if the Anthropic API key is configured."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def analyze_sentiment(news_items: list[dict]) -> dict | None:
    """
    Analyze sentiment of stock news using Claude API.

    Returns dict with:
        - overall: "Bullish" | "Bearish" | "Neutral"
        - score: float from -1 (very bearish) to 1 (very bullish)
        - summary: brief analysis text
        - details: per-headline sentiment
    """
    if not is_sentiment_available() or not news_items:
        return None

    import anthropic

    headlines = "\n".join(
        f"- {item['title']} (Source: {item['publisher']})"
        for item in news_items if item.get("title")
    )

    if not headlines.strip():
        return None

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze the sentiment of these stock news headlines for an Indian market stock.
Rate the overall sentiment and each headline.

Headlines:
{headlines}

Respond in this exact format (no markdown, no extra text):
OVERALL: [Bullish/Bearish/Neutral]
SCORE: [float from -1.0 to 1.0]
SUMMARY: [1-2 sentence analysis]
DETAILS:
- [headline]: [Bullish/Bearish/Neutral]
""",
            }
        ],
    )

    return _parse_sentiment_response(message.content[0].text, news_items)


def _parse_sentiment_response(response: str, news_items: list[dict]) -> dict:
    """Parse the Claude API response into structured data."""
    lines = response.strip().split("\n")

    overall = "Neutral"
    score = 0.0
    summary = ""
    details = []

    for line in lines:
        line = line.strip()
        if line.startswith("OVERALL:"):
            overall = line.split(":", 1)[1].strip()
        elif line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(-1.0, min(1.0, score))
            except ValueError:
                score = 0.0
        elif line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("- ") and ":" in line:
            parts = line[2:].rsplit(":", 1)
            if len(parts) == 2:
                details.append({
                    "headline": parts[0].strip(),
                    "sentiment": parts[1].strip(),
                })

    return {
        "overall": overall,
        "score": score,
        "summary": summary,
        "details": details,
    }
