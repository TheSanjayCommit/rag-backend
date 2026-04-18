import re

def extract_filters(query: str) -> dict:
    """
    Extracts structured filters: Budget, Rating, State from natural language.
    Example: "colleges in Telangana under 2 lakh" -> {'state': 'Telangana', 'max_fee': 200000}
    """
    filters = {}
    q = query.lower()

    budget_match = re.search(r'(?:under|below|budget|fees?)\s*(\d+(?:\.\d+)?)\s*(?:l|lakh)?', q)
    if budget_match:
        val = float(budget_match.group(1))
        filters["max_fee"] = val * 100000 if val < 100 else val

    rating_match = re.search(r'rating\s*(?:>|above|of)?\s*(\d+(?:\.\d+)?)', q)
    if rating_match:
        filters["min_rating"] = float(rating_match.group(1))

    states = ["telangana", "karnataka", "maharashtra", "tamil nadu", "delhi",
              "punjab", "gujarat", "andhra pradesh", "kerala", "rajasthan",
              "uttar pradesh", "west bengal"]
    for state in states:
        if state in q:
            filters["state"] = state.title()
            break

    return filters

def normalize_query(query: str) -> str:
    q = query.lower()
    mappings = {
        r"\biit\b": "indian institute of technology",
        r"\bnit\b": "national institute of technology",
        r"\biiit\b": "international institute of information technology",
        r"\bcse\b": "computer science engineering",
        r"\bece\b": "electronics communication engineering",
        r"\bmba\b": "master of business administration",
    }
    for pattern, replacement in mappings.items():
        q = re.sub(pattern, replacement, q)
    return q.strip()

def is_compare_query(query: str) -> bool:
    return any(k in query.lower() for k in ["compare", "vs", "versus", "difference", "better"])

def is_valid_query(query: str) -> bool:
    return len(query.strip()) >= 3 and bool(re.search('[a-zA-Z]', query))

# ── CONVERSATION HISTORY UTILITIES ───────────────────────────────────────────

def get_context(history: list) -> str:
    """
    Extracts the last meaningful user query from conversation history.
    Used to resolve follow-up questions like "what about placements?"

    Example:
        history = [{"user": "IIT Bombay fees", "assistant": "..."}]
        get_context(history) → "IIT Bombay fees"
    """
    if not history:
        return ""
    for turn in reversed(history):
        user_msg = turn.get("user", "").strip()
        if len(user_msg) > 4:
            return user_msg
    return ""

def update_history(history: list, query: str, answer: str) -> list:
    """
    Appends a turn to history and keeps only the last 5 turns.
    Backend remains stateless — history is passed per request from the frontend.

    Args:
        history: Existing list of {"user": str, "assistant": str} dicts
        query:   The user's original query
        answer:  The full AI response
    Returns:
        Pruned history (max 5 turns)
    """
    history = list(history or [])
    history.append({"user": query, "assistant": answer})
    return history[-5:]

def is_followup_query(query: str, history: list) -> bool:
    """
    Detects if a query is a short follow-up that needs context injection.

    Examples: "what about placements?", "and fees?", "tell me more"
    """
    if not history:
        return False
    q = query.lower().strip()
    followup_starters = [
        "what about", "and ", "tell me", "how about", "also ",
        "more about", "what is", "what are", "its ", "their ",
        "same for", "how much", "what's"
    ]
    short = len(q.split()) <= 5
    starts_with_followup = any(q.startswith(s) for s in followup_starters)
    return short or starts_with_followup
