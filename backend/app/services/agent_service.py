import json
import hashlib
import time
import logging
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore  # cross-thread user prefs

from app.services.places_service import search_places
from app.services.rag_service import search_bu_resources
from app.services.events_service import search_events
from app.services.llm_provider import get_llm
from app.services.query_router import route_query, ROUTE_PROMPTS

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300
_response_cache: dict[str, tuple[float, dict]] = {}


class AgentResponse(BaseModel):
    answer: str
    sources: list[str] = []
    response_type: str


llm = get_llm()


SYSTEM_PROMPT = """You are BU Life AI, a smart campus assistant for Boston University students.

Your job is to help students with campus life: finding places, understanding policies, and discovering events.

IMPORTANT: Call exactly ONE tool, then respond based on its results. Be concise and practical."""


BLOCKED_PATTERNS = ["ignore previous", "system prompt", "jailbreak", "forget instructions"]


def guardrail_hook(state):
    last_msg = state["messages"][-1].content.lower()
    if any(p in last_msg for p in BLOCKED_PATTERNS):
        return {"messages": [AIMessage(content="I can only help with BU campus questions.")]}
    trimmed = trim_messages(state["messages"], max_tokens=4000, token_counter=len)
    return {"messages": trimmed}


checkpointer = MemorySaver()
store = InMemoryStore()


def _cache_key(message: str, location: str | None, time_available: int | None) -> str:
    raw = f"{message.lower().strip()}|{location}|{time_available}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> dict | None:
    if key in _response_cache:
        ts, response = _response_cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return response
        del _response_cache[key]
    return None


def _set_cache(key: str, response: dict):
    now = time.time()
    _response_cache[key] = (now, response)
    if len(_response_cache) > 200:
        expired = [k for k, (ts, _) in _response_cache.items() if now - ts > CACHE_TTL_SECONDS]
        for k in expired:
            del _response_cache[k]


def _build_tools(db):

    @tool
    async def get_nearby_places(
        location: str,
        place_type: str,
        features: list[str] = None,
        max_walk_minutes: int = 10,
    ) -> str:
        """Find campus places (study spots, dining, printers, libraries) near a location.
        place_type must be one of: study, dining, printer, library, support, any.
        features is an optional list like ['quiet', 'outlets', 'coffee']."""
        result = await search_places(
            db=db,
            location=location,
            place_type=place_type,
            features=features or [],
            max_walk_minutes=max_walk_minutes,
        )
        if not result.get("places"):
            return json.dumps({"message": "No results found. Try broader search parameters."})
        return json.dumps(result)

    @tool
    async def search_bu_resource(query: str) -> str:
        """Answer questions about BU services, policies, advising, career center,
        international students (OPT/CPT), health services, financial aid, housing, etc."""
        try:
            result = await search_bu_resources(db=db, query=query)
            if not result.get("sources"):
                return json.dumps({"message": "No relevant resources found. Try rephrasing the query."})
            return json.dumps(result)
        except Exception as e:
            logger.exception("search_bu_resource tool error")
            return json.dumps({"error": str(e), "message": "Resource search failed. Try a simpler query."})

    @tool
    async def get_events(interests: list[str], days_ahead: int = 7) -> str:
        """Get personalized event recommendations based on student interests.
        interests is a list of topics like ['AI', 'career', 'startup', 'wellness']."""
        result = await search_events(db=db, interests=interests, days_ahead=days_ahead)
        if not result.get("events"):
            return json.dumps({"message": "No events found. Try broader interests or longer timeframe."})
        return json.dumps(result)

    return {
        "places": get_nearby_places,
        "resources": search_bu_resource,
        "events": get_events,
    }


_agent_cache: dict[str, object] = {}


def _build_agent_for_route(db, route: str):
    cache_key = f"{id(db)}_{route}"
    if cache_key in _agent_cache:
        return _agent_cache[cache_key]

    tool_map = _build_tools(db)
    route_tool = tool_map[route]
    route_prompt = ROUTE_PROMPTS[route]

    agent = create_react_agent(
        model=llm,
        tools=[route_tool],
        prompt=route_prompt,
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=guardrail_hook,
    )

    _agent_cache[cache_key] = agent
    return agent


def _build_supervisor(db):
    """Build default agent with all tools (used for streaming endpoint)."""
    cache_key = f"{id(db)}_all"
    if cache_key in _agent_cache:
        return _agent_cache[cache_key]

    tool_map = _build_tools(db)
    all_tools = list(tool_map.values())

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=guardrail_hook,
    )

    _agent_cache[cache_key] = agent
    return agent


def _build_messages(message: str, location: str | None, time_available: int | None, interests: list | None, route_prompt: str) -> list:
    # NOTE: Do NOT add SystemMessage here — the agent already gets it via prompt= param.
    # Adding it here duplicates it in every invocation and inflates token usage.
    context_parts = [f"Student query: {message}"]
    if location:
        context_parts.append(f"Current location: {location}")
    if time_available:
        context_parts.append(f"Time available: {time_available} minutes")
    if interests:
        context_parts.append(f"Interests: {', '.join(interests)}")

    return [
        HumanMessage(content="\n".join(context_parts)),
    ]


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
        logger.info("Cache hit for query: %s", message[:50])
        return cached

    # Groq-powered semantic routing (free, fast, separate rate limit)
    route = await route_query(message)
    logger.info("Routed '%s' → %s", message[:50], route)

    agent = _build_agent_for_route(db, route)
    route_prompt = ROUTE_PROMPTS[route]
    messages = _build_messages(message, location, time_available, interests, route_prompt)

    try:
        # Unique thread_id per query prevents message accumulation across queries
        thread_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        result = await agent.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_message = result["messages"][-1]
        response = {
            "response": final_message.content if isinstance(final_message.content, str) else str(final_message.content),
            "type": route,
            "sources": [],
        }

        _set_cache(key, response)
        return response

    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate" in error_msg or "too many" in error_msg:
            logger.warning("Rate limited by NIM API: %s", e)
            for _, (ts, resp) in _response_cache.items():
                if time.time() - ts < CACHE_TTL_SECONDS * 2:
                    return {
                        "response": "I'm currently experiencing high demand. Please try again in a minute.",
                        "type": "error",
                        "sources": [],
                    }
            return {
                "response": "I'm temporarily rate-limited. Please wait 30-60 seconds and try again.",
                "type": "error",
                "sources": [],
            }
        raise
