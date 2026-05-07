"""
Seed places and events from JSON files into the Neon database.
Usage: cd backend && python scripts/seed_data.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text
from app.db.connection import SessionLocal

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

db = SessionLocal()

# ── Seed Places ──────────────────────────────────────────────────────────────
places_path = os.path.join(DATA_DIR, "places.json")
with open(places_path) as f:
    places = json.load(f)

db.execute(text("DELETE FROM places"))
for p in places:
    db.execute(
        text("""
            INSERT INTO places (name, category, building, campus_zone, description, hours, features)
            VALUES (:name, :category, :building, :campus_zone, :description, :hours, :features)
        """),
        {
            "name": p["name"],
            "category": p["category"],
            "building": p["building"],
            "campus_zone": p["campus_zone"],
            "description": p["description"],
            "hours": p["hours"],
            "features": p.get("features", []),
        },
    )
print(f"Seeded {len(places)} places.")

# ── Seed Events ──────────────────────────────────────────────────────────────
events_path = os.path.join(DATA_DIR, "events.json")
with open(events_path) as f:
    events = json.load(f)

db.execute(text("DELETE FROM events"))
for e in events:
    db.execute(
        text("""
            INSERT INTO events (title, description, location, event_date, category, tags, source_url)
            VALUES (:title, :description, :location, :event_date, :category, :tags, :source_url)
        """),
        {
            "title": e["title"],
            "description": e["description"],
            "location": e["location"],
            "event_date": e["event_date"],
            "category": e["category"],
            "tags": e.get("tags", []),
            "source_url": e.get("source_url", ""),
        },
    )
print(f"Seeded {len(events)} events.")

db.commit()
db.close()
print("Done.")
