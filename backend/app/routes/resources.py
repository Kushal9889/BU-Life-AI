from fastapi import APIRouter, Depends, Query
from app.services.rag_service import search_bu_resources
from app.db.connection import get_db

router = APIRouter()


@router.get("/resources")
async def get_resources(
    q: str = Query(..., description="Question about BU resources"),
    db=Depends(get_db),
):
    return await search_bu_resources(db=db, query=q)
