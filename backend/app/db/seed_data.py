"""
Run this script once to load places.json and events.json into the database.
Usage: cd backend && python -m app.db.seed_data
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv

load_dotenv()

from app.db.connection import SessionLocal
from app.models.db_models import Base, Place, Event
from app.db.connection import engine

Base.metadata.create_all(bind=engine)

db = SessionLocal()

# ── Places ────────────────────────────────────────────────────────────────────
places_path = os.path.join(os.path.dirname(__file__), "../../../data/places.json")
with open(places_path) as f:
    places_data = json.load(f)

existing_place_names = {p.name for p in db.query(Place.name).all()}
added_places = 0
for p in places_data:
    if p["name"] not in existing_place_names:
        db.add(Place(**p))
        added_places += 1

print(f"Added {added_places} places.")

# ── Events ────────────────────────────────────────────────────────────────────
events_path = os.path.join(os.path.dirname(__file__), "../../../data/events.json")
if os.path.exists(events_path):
    with open(events_path) as f:
        events_data = json.load(f)

    existing_event_titles = {e.title for e in db.query(Event.title).all()}
    added_events = 0
    for e in events_data:
        if e["title"] not in existing_event_titles:
            db.add(Event(**e))
            added_events += 1
    print(f"Added {added_events} events.")
else:
    print("No events.json found, skipping events.")

db.commit()
db.close()
print("Seed complete.")
