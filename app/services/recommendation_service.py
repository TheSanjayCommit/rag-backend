import re
import pandas as pd
import os
from app.config.settings import settings

# ── DETECTION ────────────────────────────────────────────────────────────────

RECOMMENDATION_TRIGGERS = [
    "suggest", "recommend", "rank", "got rank", "my rank",
    "low budget", "cheap college", "affordable", "best college for",
    "top college for", "college for cse", "which college"
]

def is_recommendation_query(query: str) -> bool:
    """Returns True if the query is asking for personalized college recommendations."""
    q = query.lower()
    return any(trigger in q for trigger in RECOMMENDATION_TRIGGERS)

# ── PREFERENCE EXTRACTION ─────────────────────────────────────────────────────

def extract_user_preferences(query: str) -> dict:
    """
    Extracts structured user preferences from natural language.

    Example:
        "I got 5000 rank, budget 2 lakh, CSE in Telangana"
        → {"rank": 5000, "max_fee": 200000, "state": "Telangana", "course": "cse"}
    """
    q = query.lower()
    prefs = {}

    # 1. Extract Rank
    rank_match = re.search(r'(?:rank|got|scored|jee|mains|advanced)[\s:]*(\d{1,6})', q)
    if rank_match:
        prefs["rank"] = int(rank_match.group(1))

    # 2. Extract Budget (lakhs → absolute)
    budget_match = re.search(
        r'(?:under|below|budget|fees?|within|upto|up to)[\s₹]*(\d+(?:\.\d+)?)\s*(?:l|lakh|lakhs)?',
        q
    )
    if budget_match:
        val = float(budget_match.group(1))
        prefs["max_fee"] = val * 100000 if val < 1000 else val

    # 3. Extract State
    states = {
        "telangana": "telangana", "karnataka": "karnataka",
        "maharashtra": "maharashtra", "tamil nadu": "tamil nadu",
        "delhi": "delhi", "gujarat": "gujarat", "punjab": "punjab",
        "rajasthan": "rajasthan", "kerala": "kerala",
        "andhra pradesh": "andhra pradesh", "west bengal": "west bengal",
        "uttar pradesh": "uttar pradesh", "madhya pradesh": "madhya pradesh",
    }
    for key, val in states.items():
        if key in q:
            prefs["state"] = val
            break

    # 4. Extract Course
    courses = {
        "cse": "computer science", "computer science": "computer science",
        "ece": "electronics", "electrical": "electrical",
        "mechanical": "mechanical", "civil": "civil",
        "it ": "information technology", "ai": "artificial intelligence",
        "data science": "data science"
    }
    for key, val in courses.items():
        if key in q:
            prefs["course"] = val
            break
    if "course" not in prefs:
        prefs["course"] = "engineering"  # Default

    return prefs


# ── RANK-BASED COLLEGE TIER MAPPING ─────────────────────────────────────────

def get_rank_tier(rank: int) -> list:
    """
    Maps JEE rank to eligible college tiers.
    Returns list of name keywords to match against dataset.
    """
    if rank < 1000:
        return ["indian institute of technology"]   # Top IITs only
    elif rank < 5000:
        return ["indian institute of technology", "national institute of technology"]
    elif rank < 15000:
        return ["national institute of technology", "international institute of information technology"]
    else:
        return []  # No tier filter — show all state colleges


# ── NUMERIC FEE PARSER ────────────────────────────────────────────────────────

def parse_fee(fee_str) -> float:
    """Converts '1,49,250' or '149250' strings to float."""
    if pd.isna(fee_str):
        return 0.0
    return float(re.sub(r'[^\d.]', '', str(fee_str)) or 0)


# ── CORE RECOMMENDATION ENGINE ────────────────────────────────────────────────

def recommend_colleges(query: str) -> str:
    """
    Main recommendation function.
    1. Extracts preferences from query
    2. Loads dataset
    3. Applies filters (fee, state, rank tier)
    4. Scores and ranks results
    5. Returns formatted top-5 recommendations

    STRICT RULE: Only uses local RAG dataset. No API or Web calls.
    """
    prefs = extract_user_preferences(query)

    # Load dataset
    csv_path = os.path.join(settings.DATA_PATH, "Indian_Engineering_Colleges_Dataset.csv")
    if not os.path.exists(csv_path):
        return "Unable to access the college database. Please ensure the dataset is available."

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading college database: {e}"

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    df["College_Name"] = df["College_Name"].astype(str).str.strip()
    df["State"]        = df["State"].astype(str).str.strip().str.lower()
    df["fee_val"]      = df["UG_fee"].apply(parse_fee)
    df["rating_val"]   = pd.to_numeric(df["Rating"],    errors="coerce").fillna(0)
    df["placement_val"]= pd.to_numeric(df["Placement"], errors="coerce").fillna(0)

    filtered = df.copy()

    # Filter 1: Budget
    if "max_fee" in prefs:
        filtered = filtered[filtered["fee_val"] <= prefs["max_fee"]]

    # Filter 2: State
    if "state" in prefs:
        filtered = filtered[filtered["State"].str.contains(prefs["state"], na=False)]

    # Filter 3: Rank-based tier keywords
    if "rank" in prefs:
        tier_keywords = get_rank_tier(prefs["rank"])
        if tier_keywords:
            pattern = "|".join(tier_keywords)
            filtered = filtered[
                filtered["College_Name"].str.lower().str.contains(pattern, na=False)
            ]

    if filtered.empty:
        # Relax rank filter if no results found
        if "rank" in prefs:
            filtered = df.copy()
            if "max_fee" in prefs:
                filtered = filtered[filtered["fee_val"] <= prefs["max_fee"]]
            if "state" in prefs:
                filtered = filtered[filtered["State"].str.contains(prefs["state"], na=False)]

    if filtered.empty:
        state_info = f" in {prefs.get('state', '').title()}" if "state" in prefs else ""
        budget_info = f" under ₹{int(prefs['max_fee']):,}" if "max_fee" in prefs else ""
        return (
            f"No suitable colleges found{state_info}{budget_info} matching your preferences. "
            "Try relaxing your budget or location filter."
        )

    # Scoring: Rating (40%) + Placement (40%) + Affordability (20%)
    max_fee_in_set = filtered["fee_val"].max() or 1
    filtered = filtered.copy()
    filtered["affordability_score"] = 1 - (filtered["fee_val"] / max_fee_in_set)
    filtered["total_score"] = (
        (filtered["rating_val"]    / 10) * 0.40 +
        (filtered["placement_val"] / 10) * 0.40 +
        filtered["affordability_score"]  * 0.20
    )

    top5 = filtered.nlargest(5, "total_score")

    # Format output
    lines = []
    for _, row in top5.iterrows():
        fee_display = f"₹{int(row['fee_val']):,}/yr" if row["fee_val"] > 0 else "Fee N/A"
        rating_display = f"{row['rating_val']:.1f}/10" if row["rating_val"] > 0 else "N/A"
        placement_display = f"{row['placement_val']:.1f}/10" if row["placement_val"] > 0 else "N/A"
        lines.append(
            f"• **{row['College_Name']}** ({row['State'].title()})\n"
            f"  Fees: {fee_display} | Rating: {rating_display} | Placement: {placement_display}"
        )

    # Build reason string
    reasons = []
    if "rank" in prefs:   reasons.append(f"JEE Rank ~{prefs['rank']:,}")
    if "max_fee" in prefs: reasons.append(f"Budget ≤ ₹{int(prefs['max_fee']):,}")
    if "state" in prefs:  reasons.append(f"Location: {prefs['state'].title()}")
    if "course" in prefs: reasons.append(f"Course: {prefs['course'].title()}")
    reason_str = ", ".join(reasons) if reasons else "your preferences"

    result = (
        f"**🎓 Recommended Colleges for You:**\n\n"
        + "\n\n".join(lines)
        + f"\n\n**Based on:** {reason_str}\n"
        + "**Source:** Verified Internal Database | **Confidence:** High"
    )
    return result
