from fastapi import APIRouter, Depends, Query
from app.services.events_service import search_events
from app.db.connection import get_db

router = APIRouter()


@router.get("/events")
async def get_events(
    interests: list[str] = Query(default=["general"]),
    days_ahead: int = Query(default=7),
    db=Depends(get_db),
):
    return await search_events(db=db, interests=interests, days_ahead=days_ahead)
