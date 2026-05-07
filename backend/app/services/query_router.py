"""
LLM-powered semantic query router using Groq (free, fast).
Keeps NVIDIA NIM rate budget for the actual agent response.
Falls back to keyword matching if Groq fails.
"""
import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """Classify this student query into exactly one category.

Categories:
- places: finding study spots, dining, printers, libraries, buildings
- resources: BU policies, advising, CPT/OPT, financial aid, health, housing
- events: campus events, workshops, fairs, meetups, hackathons

Reply with ONLY the category name. No explanation.

Query: {query}"""

ROUTE_PROMPTS = {
    "places": (
        "You are a BU campus places expert. Given the student's query, recommend "
        "the best places from the provided data. Mention building, features, and "
        "approximate walking time. Be concise and practical."
    ),
    "resources": (
        "You are a BU policy and services expert. Answer the student's question "
        "using the provided BU resource data. Always cite the official BU source URL. "
        "Be specific about steps, deadlines, and locations."
    ),
    "events": (
        "You are a BU events expert. Recommend the most relevant events from the "
        "provided data based on the student's interests. Mention date, location, "
        "and why it's relevant to them. Be concise."
    ),
}

_router_llm = None


def _get_router_llm():
    """Groq-powered router LLM — fast, free, separate rate limit from NIM."""
    global _router_llm
    if _router_llm is None:
        import os
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            _router_llm = ChatOpenAI(
                model="llama-3.1-8b-instant",
                temperature=0,
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                max_retries=2,
                request_timeout=5,
            )
        else:
            logger.warning("GROQ_API_KEY not set — router will use keyword fallback")
    return _router_llm


async def route_query(message: str) -> str:
    """Route query using Groq LLM. Falls back to keyword matching on failure."""
    llm = _get_router_llm()
    if llm is None:
        return _keyword_fallback(message)

    try:
        prompt = ROUTER_PROMPT.format(query=message)
        response = await llm.ainvoke(prompt)
        route = response.content.strip().lower()

        if route in ROUTE_PROMPTS:
            logger.info("Groq routed '%s' → %s", message[:40], route)
            return route

        logger.warning("Groq returned unknown route '%s', falling back", route)
    except Exception as e:
        logger.warning("Groq router failed (%s), using keyword fallback", e)

    return _keyword_fallback(message)


def _keyword_fallback(message: str) -> str:
    """Fast keyword fallback when Groq is unavailable or fails."""
    msg = message.lower()

    event_kw = ["event", "hackathon", "fair", "workshop", "meetup", "happening", "upcoming"]
    resource_kw = ["how do i", "how to", "cpt", "opt", "advising", "financial aid", "housing", "policy", "apply"]

    if any(w in msg for w in event_kw):
        return "events"
    if any(w in msg for w in resource_kw):
        return "resources"
    return "places"
