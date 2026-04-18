import httpx
import re
from app.config.settings import settings

# Phrases that indicate a result is useless
JUNK_PHRASES = [
    "not available", "data not found", "we couldn't find",
    "page not found", "subscribe to", "sign up", "cookie policy",
    "privacy policy", "terms of service", "advertisement"
]

def clean_web_data(results: list) -> list:
    """
    Filters and cleans raw Tavily results.
    - Removes snippets containing junk phrases
    - Strips HTML entities and extra whitespace
    - Keeps only the first 400 characters per result
    - Limits to top 3 clean results
    """
    cleaned = []
    for r in results:
        text = r.get("content", "").strip()
        title = r.get("title", "").strip()

        # Skip if too short to be useful
        if len(text) < 30:
            continue

        # Skip if it contains junk phrases
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in JUNK_PHRASES):
            continue

        # Remove HTML entities and clean whitespace
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Keep only first 400 chars - enough to extract a fact
        text = text[:400]

        cleaned.append({
            "title": title,
            "url": r.get("url", ""),
            "content": text
        })

    return cleaned[:3]


async def perform_tavily_search(query: str) -> list:
    """
    Executes Tavily web search and returns cleaned, filtered results.
    """
    if not settings.TAVILY_API_KEY:
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.TAVILY_API_KEY.strip(),
        "query": query,
        "search_depth": "advanced",  # Use advanced for better quality snippets
        "max_results": 5             # Fetch 5, then clean down to 3
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=12.0)
            if response.status_code == 200:
                raw = response.json().get("results", [])
                return clean_web_data(raw)
            return []
        except Exception:
            return []
