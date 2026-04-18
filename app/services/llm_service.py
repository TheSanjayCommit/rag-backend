import httpx
import json
from app.config.settings import settings

async def rewrite_query(query: str, history: list) -> str:
    """
    Rewrites a query using conversation history to resolve follow-ups.
    Example: "what about placements?" + history about IIT Bombay
           → "What are the placement statistics of IIT Bombay?"
    """
    if not history or not settings.GROQ_API_KEY:
        return query

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}

    history_context = "\n".join([
        f"User: {h['user']}\nAssistant: {h['assistant'][:150]}"
        for h in history[-3:]
    ])

    prompt = f"""You are a query rewriting assistant for a college information system.

Conversation context:
{history_context}

New user query: "{query}"

Rewrite the query to be clear, specific, and self-contained using the context above.
Rules:
- Return ONLY the rewritten query text
- Keep it concise (under 15 words)
- If the query is already clear and specific, return it unchanged
- Do NOT add explanations or quotes"""

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 60
            }, timeout=8.0)
            return res.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
    except Exception:
        return query

async def safe_rewrite(query: str, history: list) -> str:
    """
    Safe wrapper around rewrite_query. Falls back to original query if:
    - Rewrite fails (network/API error)
    - Result is too short (< 5 chars) — likely garbage
    - Result is suspiciously long (> 3x original) — likely hallucination
    - Result is empty
    """
    original = query.strip()
    try:
        rewritten = await rewrite_query(original, history)
        rewritten = rewritten.strip()

        # Validate output quality
        if not rewritten:
            return original
        if len(rewritten) < 5:
            return original
        if len(rewritten) > len(original) * 4:
            # LLM went off-rails and returned something very long
            return original

        return rewritten
    except Exception:
        return original


async def expand_query(query: str, history: list) -> list:
    if not settings.GROQ_API_KEY: return [query]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}
    prompt = f"Query: '{query}'. Generate 3 search variations. Return JSON: {{\"queries\": [\"v1\", \"v2\", \"v3\"]}}"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.0}, timeout=5.0)
            return json.loads(res.json()["choices"][0]["message"]["content"]).get("queries", [query])
    except Exception: return [query]

async def rerank_results(results: list, query: str) -> list:
    if not results or not settings.GROQ_API_KEY: return results[:3]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}
    ctx = "\n".join([f"{i+1}. {r['text']}" for i, r in enumerate(results[:10])])
    prompt = f"Query: {query}\n\nColleges:\n{ctx}\n\nReturn JSON: {{\"indices\": [1, 2, 3]}}"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.0}, timeout=5.0)
            idx = json.loads(res.json()["choices"][0]["message"]["content"]).get("indices", [1, 2, 3])
            return [results[i-1] for i in idx if i-1 < len(results)]
    except Exception: return results[:3]

async def extract_facts_from_web(query: str, web_snippets: list) -> str:
    """
    PRE-EXTRACTION STEP: Uses LLM to distill raw web snippets into clean factual answers
    BEFORE the final synthesis. This prevents the final prompt from receiving noise.
    """
    if not web_snippets or not settings.GROQ_API_KEY:
        return "\n".join([s.get("content", "") for s in web_snippets])

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}

    combined_snippets = "\n\n".join([
        f"Source: {s.get('title', 'Web')}\n{s.get('content', '')}"
        for s in web_snippets
    ])

    prompt = f"""
You are a precise fact extractor. A user asked: "{query}"

Below are web search snippets. Extract ONLY the specific factual answer.

RULES:
- Return ONLY the relevant fact (e.g., a fee amount, admission rate, ranking).
- If you find a number or range, include it exactly.
- Do NOT include financial aid, loan info, or irrelevant stats.
- If the answer is truly not in the snippets, say "Amount not found in search results."
- Keep your answer to 2-3 sentences maximum.

WEB SNIPPETS:
{combined_snippets}

EXTRACTED ANSWER:"""

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200
            }, timeout=10.0)
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return combined_snippets

async def generate_streaming_response(query: str, context: str, citations: list = None, source: str = "Database"):
    """
    Structured output synthesis — enforces clean, bullet-point professional format.
    """
    if not settings.GROQ_API_KEY: yield "Configuration error."; return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY.strip()}", "Content-Type": "application/json"}

    # Build source footer
    source_urls = list(set(citations))[:3] if citations else []
    source_footer = f"**Source:** {source}"
    if source_urls:
        source_footer += "\n" + "\n".join([f"- {u}" for u in source_urls])

    system_prompt = """You are a professional college admissions advisor AI.
Your job is to give SHORT, CLEAN, STRUCTURED answers — like a premium AI assistant.

STRICT OUTPUT RULES:
1. NEVER dump raw data or long paragraphs.
2. ALWAYS use bullet points (•) for facts.
3. MAX 4 bullet points per answer.
4. Include ONLY: main value + 1 optional secondary value.
5. REMOVE: per-credit-hour data, duplicate values, financial aid stats, empty fields.
6. NEVER show a table unless ALL columns have real values.
7. For a single institution query: Name → bullet facts.
8. For comparisons: use a Markdown table with ONLY filled columns.
9. If a value is missing, skip that bullet entirely. Do NOT write "N/A" or "Data not available".
10. Keep total response under 120 words."""

    prompt = f"""DATA SOURCE: {source}

CONTEXT:
{context}

USER QUERY: {query}

FORMAT YOUR ANSWER LIKE THIS EXAMPLE:
---
**Harvard University** — Tuition & Costs:
• Tuition: ~$56,550/year
• Total estimated cost: ~$79,450/year (including room & board)

{source_footer}
---

Now answer the actual query above in this format. Only include bullet points that have real data from the context."""

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", url, headers=headers, json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 512,
                "stream": True
            }, timeout=60.0) as res:
                async for line in res.aiter_lines():
                    if line.startswith("data: "):
                        line = line[6:]
                        if "[DONE]" in line: break
                        try:
                            token = json.loads(line)["choices"][0].get("delta", {}).get("content", "")
                            if token: yield token
                        except Exception: continue
        except Exception:
            yield "\n[Stream error — please retry]"
