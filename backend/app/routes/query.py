import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse

from langchain_core.messages import AIMessageChunk
from app.services.agent_service import handle_query, _build_agent_for_route, _build_messages
from app.services.query_router import route_query, ROUTE_PROMPTS
from app.db.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class QueryRequest(BaseModel):
    message: str
    location: str | None = None
    time_available: int | None = None
    interests: list[str] | None = None
    session_id: str = "default"


@router.post("/query")
@limiter.limit("100/hour")
async def query(request: Request, req: QueryRequest, db=Depends(get_db)):
    try:
        result = await handle_query(
            message=req.message,
            location=req.location,
            time_available=req.time_available,
            interests=req.interests,
            db=db,
            session_id=req.session_id,
        )
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate" in error_msg:
            logger.warning("Rate limit hit: %s", e)
            return {
                "response": "I'm temporarily rate-limited. Please wait 30-60 seconds and try again.",
                "type": "error",
                "sources": [],
            }
        logger.exception("Query failed: %s", e)
        return {
            "response": "Something went wrong processing your query. Please try rephrasing or try again.",
            "type": "error",
            "sources": [],
        }


@router.post("/query/stream")
@limiter.limit("100/hour")
async def query_stream(request: Request, req: QueryRequest, db=Depends(get_db)):
    route = await route_query(req.message)
    agent = _build_agent_for_route(db, route)
    route_prompt = ROUTE_PROMPTS[route]
    messages = _build_messages(req.message, req.location, req.time_available, req.interests, route_prompt)

    async def generate():
        try:
            async for chunk, metadata in agent.astream(
                {"messages": messages},
                config={"configurable": {"thread_id": f"{req.session_id}_{uuid.uuid4().hex[:8]}"}},
                stream_mode="messages",
            ):
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not chunk.tool_calls
                    and not chunk.tool_call_chunks
                ):
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg:
                logger.warning("NIM rate limit during stream: %s", e)
                yield f"data: {json.dumps({'token': 'Rate limited — please wait 30-60 seconds and try again.'})}\n\n"
            else:
                logger.exception("Stream failed")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
