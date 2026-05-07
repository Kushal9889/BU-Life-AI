from sqlalchemy.orm import Session
from app.models.db_models import Event
from datetime import date, timedelta


async def search_events(db: Session, interests: list, days_ahead: int = 7) -> dict:
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    events = db.query(Event).filter(
        Event.event_date >= today,
        Event.event_date <= end_date,
    ).all()

    matched = []
    for event in events:
        if event.tags:
            match_score = sum(
                1
                for interest in interests
                if any(interest.lower() in tag.lower() for tag in event.tags)
            )
            if match_score > 0:
                matched.append((match_score, event))

    # Include all upcoming events if no interest match found
    if not matched:
        matched = [(0, e) for e in events]

    matched.sort(key=lambda x: x[0], reverse=True)

    return {
        "events": [
            {
                "title": e.title,
                "description": e.description,
                "location": e.location,
                "date": str(e.event_date),
                "category": e.category,
                "tags": e.tags,
                "url": e.source_url,
            }
            for _, e in matched[:5]
        ]
    }
