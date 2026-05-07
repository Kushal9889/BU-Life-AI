"""
Build and store embeddings for places and BU resources using LangChain + PGVector.
Run after scrape_bu_resources.py and seed_data.py.
Usage: cd backend && python scripts/build_embeddings.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from app.db.connection import SessionLocal
from app.models.db_models import Place, Event
from app.services.llm_provider import get_embeddings

embeddings = get_embeddings()
CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql://localhost/bulife")

db = SessionLocal()

# ── BU Resources → PGVector collection "bu_resources" ─────────────────────────
resources_path = os.path.join(os.path.dirname(__file__), "../../data/bu_resources.json")
if os.path.exists(resources_path):
    with open(resources_path) as f:
        resources = json.load(f)

    docs = [
        Document(
            page_content=f"{r['title']}. {r['content']}",
            metadata={"title": r["title"], "url": r["url"], "category": r["category"]},
        )
        for r in resources
    ]

    print(f"Storing {len(docs)} BU resource embeddings via LangChain PGVector...")
    PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="bu_resources",
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,  # re-run safe: clears old embeddings first
    )
    print("BU resources done.")
else:
    print("No bu_resources.json found. Run scrape_bu_resources.py first.")

# ── Places → store embeddings back on Place rows ───────────────────────────────
places = db.query(Place).all()
places_without_embedding = [p for p in places if p.embedding is None]

if places_without_embedding:
    for place in places_without_embedding:
        print(f"Embedding place: {place.name}")
        text = f"{place.name}. {place.description}. Features: {', '.join(place.features or [])}"
        vector = embeddings.embed_query(text[:8000])
        place.embedding = vector

    db.commit()
    print(f"Embedded {len(places_without_embedding)} places.")
else:
    print("All places already have embeddings, skipping.")

# ── Events → store embeddings back on Event rows ─────────────────────────────
events = db.query(Event).all()
events_without_embedding = [e for e in events if e.embedding is None]

if events_without_embedding:
    for event in events_without_embedding:
        print(f"Embedding event: {event.title}")
        tags_str = ", ".join(event.tags or [])
        text = f"{event.title}. {event.description}. Tags: {tags_str}"
        vector = embeddings.embed_query(text[:8000])
        event.embedding = vector

    db.commit()
    print(f"Embedded {len(events_without_embedding)} events.")
else:
    print("All events already have embeddings, skipping.")

db.close()
print("Done.")
