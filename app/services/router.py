from app.utils.helpers import normalize_query, is_compare_query, is_valid_query, extract_filters, get_context, is_followup_query
from app.services.rag_service import search_rag, is_valid_rag
from app.services.api_service import fetch_college_scorecard
from app.services.web_service import perform_tavily_search
from app.services.llm_service import generate_streaming_response, safe_rewrite, expand_query, rerank_results, extract_facts_from_web
from app.services.recommendation_service import is_recommendation_query, recommend_colleges

# Queries containing these terms skip RAG and go straight to API/Web
FOREIGN_INSTITUTION_KEYWORDS = [
    "stanford", "mit", "harvard", "yale", "princeton", "oxford", "cambridge",
    "caltech", "cornell", "columbia", "nyu", "ucla", "uc berkeley", "usa",
    "us university", "american university"
]

async def route_and_stream(query: str, history: list = None):
    """
    GUARANTEED Fallback Pipeline:
      STEP 1: RAG  → if valid → STOP
      STEP 2: API  → if valid → STOP
      STEP 3: WEB  → always tried if above fail
      FINAL:  Return failure only if ALL THREE sources return nothing
    """
    if not is_valid_query(query):
        yield "Please enter a valid query about colleges or admissions."
        return

    # ── FAST PATH: PERSONALIZED RECOMMENDATION ENGINE ────────────────────────
    # Handles: "suggest colleges", "I got 5000 rank", "low budget colleges", etc.
    # Uses ONLY internal dataset — no API or Web calls.
    if is_recommendation_query(query):
        result = recommend_colleges(query)
        yield result
        return

    # ── PRE-PROCESSING ───────────────────────────────────────────────────
    # Only rewrite if this looks like a follow-up question (short/vague)
    # This avoids unnecessary LLM calls for clear, specific queries
    if history and is_followup_query(query, history):
        processed_query = await safe_rewrite(query, history)
    else:
        processed_query = query

    norm_query = normalize_query(processed_query)
    filters    = extract_filters(norm_query)
    is_compare = is_compare_query(norm_query.lower())
    is_foreign = any(k in norm_query.lower() for k in FOREIGN_INSTITUTION_KEYWORDS)

    context_hits = []
    citations    = set()
    source_label = "No Source"

    # ── STEP 1: INTERNAL RAG ─────────────────────────────────────────────────
    if not is_foreign:
        search_queries = await expand_query(norm_query, history)
        for sq in search_queries:
            hits = search_rag(str(sq), "india", filters=filters)
            if is_valid_rag([h["text"] for h in hits], str(sq)):
                context_hits.extend(hits)

        if context_hits and not is_compare:
            # RAG succeeded → stop here
            source_label = "Verified Internal Database"
            # Fall through to synthesis below
        elif context_hits and is_compare:
            # RAG has partial data for a comparison query → continue to API for other side
            source_label = "Hybrid (Database + External)"

    # ── STEP 2: OFFICIAL US COLLEGE API ──────────────────────────────────────
    # Triggered when: foreign query, compare query, OR RAG returned nothing
    if is_foreign or is_compare or not context_hits:
        api_res = await fetch_college_scorecard(norm_query)
        if api_res and "No official records" not in api_res:
            context_hits.append({
                "text": f"OFFICIAL_API_DATA: {api_res}",
                "name": "College Scorecard API"
            })
            citations.add("Official US College Scorecard API")
            if source_label == "No Source":
                source_label = "Official API"

    # ── STEP 3: WEB SEARCH ────────────────────────────────────────────────────
    # Triggered when: STILL no data after RAG+API, OR user asked for news/latest
    is_news = any(k in norm_query.lower() for k in ["latest", "news", "2026", "update", "deadline", "cutoff"])
    if is_news or not context_hits:
        web_hits = await perform_tavily_search(norm_query)
        if web_hits:
            # PRE-EXTRACTION: Distill raw snippets into clean factual answer first
            extracted_facts = await extract_facts_from_web(norm_query, web_hits)
            context_hits.append({
                "text": f"WEB_EXTRACTED_FACTS: {extracted_facts}",
                "name": f"Web: {web_hits[0]['title'][:40]}"
            })
            citations.update([h["url"] for h in web_hits])
            if source_label == "No Source":
                source_label = "Web Search"

    # ── FINAL FALLBACK: All Three Sources Failed ──────────────────────────────
    if not context_hits:
        yield (
            "I could not find verified data for your query across my internal "
            "database, official APIs, and web search. Please try rephrasing or "
            "asking about a specific institution (e.g., 'IIT Bombay fees' or "
            "'Harvard tuition')."
        )
        return

    # ── RE-RANKING & SYNTHESIS ────────────────────────────────────────────────
    seen = set()
    unique_hits = []
    for h in context_hits:
        key = h.get("name", h["text"][:60])
        if key not in seen:
            unique_hits.append(h)
            seen.add(key)

    final_hits = await rerank_results(unique_hits, norm_query)
    context    = "\n\n---\n\n".join([h["text"] for h in final_hits])

    async for token in generate_streaming_response(
        norm_query, context,
        citations=list(citations),
        source=source_label
    ):
        yield token
