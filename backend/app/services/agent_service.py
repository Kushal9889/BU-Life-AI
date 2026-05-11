import json
import hashlib
import time
import logging
import uuid
from contextvars import ContextVar

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from app.services.places_service import search_places
from app.services.rag_service import search_bu_resources
from app.services.events_service import search_events
from app.services.llm_provider import get_llm

load_dotenv()

logger = logging.getLogger(__name__)

llm = get_llm()
checkpointer = MemorySaver()
store = InMemoryStore()

# set before each request so tools can access the DB session
db_var: ContextVar = ContextVar("db_var")

CACHE_TTL = 300
_cache: dict[str, tuple[float, dict]] = {}

BLOCKED = ["ignore previous", "system prompt", "jailbreak", "forget instructions"]


def guardrail_hook(state):
    last = state["messages"][-1].content.lower()
    if any(p in last for p in BLOCKED):
        return {"messages": [AIMessage(content="I can only help with BU campus questions.")]}
    return {"messages": trim_messages(state["messages"], max_tokens=4000, token_counter=len)}


# tools — each grabs the db session from context at call time

@tool
async def get_nearby_places(
    location: str,
    place_type: str = "any",
    features: list[str] = None,
    max_walk_minutes: int = 10,
) -> str:
    """Find campus places (study spots, dining, printers, libraries) near a location.
    place_type: study, dining, printer, library, support, any.
    features: optional list like ['quiet', 'outlets', 'coffee']."""
    db = db_var.get()
    result = await search_places(
        db=db,
        location=location,
        place_type=place_type,
        features=features or [],
        max_walk_minutes=int(max_walk_minutes),
    )
    if not result.get("places"):
        return json.dumps({"message": "No results found. Try broader search."})
    return json.dumps(result)


@tool
async def search_bu_resource(query: str) -> str:
    """Answer questions about BU services, policies, advising, career center,
    international students (OPT/CPT), health services, financial aid, housing."""
    db = db_var.get()
    try:
        result = await search_bu_resources(db=db, query=query)
        if not result.get("sources"):
            return json.dumps({"message": "No relevant resources found. Try rephrasing."})
        return json.dumps(result)
    except Exception as e:
        logger.exception("resource search failed")
        return json.dumps({"error": str(e), "message": "Resource search failed."})


@tool
async def get_events(interests: list[str], days_ahead: int = 7) -> str:
    """Get event recommendations based on student interests.
    interests: list like ['AI', 'career', 'startup', 'wellness']."""
    db = db_var.get()
    result = await search_events(db=db, interests=interests, days_ahead=int(days_ahead))
    if not result.get("events"):
        return json.dumps({"message": "No events found. Try broader interests or longer timeframe."})
    return json.dumps(result)


# sub-agents — one per domain, each with a single specialized tool

places_agent = create_react_agent(
    llm, [get_nearby_places],
    prompt=(
        "You help BU students find campus places. Given a query, use your tool "
        "to search for study spots, dining, printers, or libraries. Mention the "
        "building name, key features, and approximate walking time. Keep it short."
    ),
)

resource_agent = create_react_agent(
    llm, [search_bu_resource],
    prompt=(
        "You help BU students with policies and services. Use your tool to look up "
        "official BU resources. Always cite the source URL. Be specific about steps, "
        "deadlines, and office locations."
    ),
)

events_agent = create_react_agent(
    llm, [get_events],
    prompt=(
        "You help BU students discover campus events. Use your tool to find events "
        "matching their interests. Mention the date, location, and why it fits them."
    ),
)

# supervisor — routes to the right sub-agent via .as_tool()

SUPERVISOR_PROMPT = (
    "You are BU Life AI, a campus assistant for Boston University students.\n\n"
    "You have three experts available:\n"
    "- places_expert: finds study spots, dining, printers, libraries, buildings\n"
    "- resource_expert: answers questions about BU policies, advising, CPT/OPT, "
    "financial aid, health services, housing\n"
    "- events_expert: finds campus events, workshops, fairs, meetups\n\n"
    "Pick the ONE expert that best matches the student's question, call them, "
    "then give a clear and concise answer based on what they found. "
    "Include source URLs when available."
)

supervisor = create_react_agent(
    model=llm,
    tools=[
        places_agent.as_tool(
            name="places_expert",
            description="Find campus places near the student (study spots, dining, printers, libraries)",
        ),
        resource_agent.as_tool(
            name="resource_expert",
            description="Answer questions about BU policies, services, advising, CPT/OPT, financial aid",
        ),
        events_agent.as_tool(
            name="events_expert",
            description="Find relevant campus events, workshops, fairs, meetups",
        ),
    ],
    prompt=SUPERVISOR_PROMPT,
    checkpointer=checkpointer,
    store=store,
    pre_model_hook=guardrail_hook,
)


# cache helpers

def _cache_key(msg: str, loc: str | None, t: int | None) -> str:
    raw = f"{msg.lower().strip()}|{loc}|{t}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> dict | None:
    if key in _cache:
        ts, resp = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return resp
        del _cache[key]
    return None


def _set_cache(key: str, resp: dict):
    now = time.time()
    _cache[key] = (now, resp)
    if len(_cache) > 200:
        expired = [k for k, (ts, _) in _cache.items() if now - ts > CACHE_TTL]
        for k in expired:
            del _cache[k]


# public api

def build_messages(msg: str, location: str | None, time_available: int | None, interests: list | None) -> list:
    parts = [f"Student query: {msg}"]
    if location:
        parts.append(f"Current location: {location}")
    if time_available:
        parts.append(f"Time available: {time_available} minutes")
    if interests:
        parts.append(f"Interests: {', '.join(interests)}")
    return [HumanMessage(content="\n".join(parts))]


async def handle_query(
    message: str,
    location: str | None,
    time_available: int | None,
    interests: list | None,
    db,
    session_id: str = "default",
) -> dict:
    key = _cache_key(message, location, time_available)
    cached = _get_cached(key)
    if cached:
        logger.info("cache hit: %s", message[:50])
        return cached

    db_var.set(db)
    messages = build_messages(message, location, time_available, interests)
    thread_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

    try:
        result = await supervisor.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )

        final = result["messages"][-1]
        response = {
            "response": final.content if isinstance(final.content, str) else str(final.content),
            "type": "assistant",
            "sources": [],
        }

        _set_cache(key, response)
        return response

    except Exception as e:
        err = str(e).lower()
        if "429" in err or "rate" in err or "too many" in err:
            logger.warning("rate limited: %s", e)
            return {
                "response": "I'm temporarily rate-limited. Please wait 30-60 seconds and try again.",
                "type": "error",
                "sources": [],
            }
        raise
