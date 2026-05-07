from sqlalchemy import Column, Integer, String, Text, ARRAY, DECIMAL, Date, Time, TIMESTAMP
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Place(Base):
    __tablename__ = "places"

    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    category = Column(String(50))
    building = Column(String(100))
    floor = Column(String(20))
    campus_zone = Column(String(50))
    description = Column(Text)
    hours = Column(Text)
    features = Column(ARRAY(String))
    lat = Column(DECIMAL(10, 8))
    lng = Column(DECIMAL(11, 8))
    embedding = Column(Vector(1024))


class BUResource(Base):
    __tablename__ = "bu_resources"

    id = Column(Integer, primary_key=True)
    title = Column(String(300))
    url = Column(Text)
    category = Column(String(100))
    content = Column(Text)
    summary = Column(Text)
    embedding = Column(Vector(1024))


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    title = Column(String(300))
    description = Column(Text)
    location = Column(String(200))
    event_date = Column(Date)
    start_time = Column(Time)
    end_time = Column(Time)
    category = Column(String(100))
    tags = Column(ARRAY(String))
    source_url = Column(Text)
    embedding = Column(Vector(1024))


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True)
    interests = Column(ARRAY(String))
    role = Column(String(50))
