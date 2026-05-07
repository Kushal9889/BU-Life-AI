-- Enable pgvector extension first
CREATE EXTENSION IF NOT EXISTS vector;

-- Campus Places (study spots, dining, libraries, printers, etc.)
CREATE TABLE IF NOT EXISTS places (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),         -- 'study', 'dining', 'library', 'printer', 'support'
    building VARCHAR(100),
    floor VARCHAR(20),
    lat DECIMAL(10, 8),
    lng DECIMAL(11, 8),
    description TEXT,
    hours TEXT,                   -- e.g. "Mon-Thu 8am-midnight"
    features TEXT[],              -- e.g. {'outlets', 'quiet', 'wifi', 'whiteboards'}
    campus_zone VARCHAR(50),      -- 'CAS', 'CDS', 'GSU', 'Questrom', etc.
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- BU Resource Documents (for RAG)
CREATE TABLE IF NOT EXISTS bu_resources (
    id SERIAL PRIMARY KEY,
    title VARCHAR(300) NOT NULL,
    url TEXT NOT NULL,
    category VARCHAR(100),        -- 'advising', 'career', 'international', 'health', etc.
    content TEXT,
    summary TEXT,
    embedding vector(1024),
    scraped_at TIMESTAMP DEFAULT NOW()
);

-- Events
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    title VARCHAR(300) NOT NULL,
    description TEXT,
    location VARCHAR(200),
    event_date DATE,
    start_time TIME,
    end_time TIME,
    category VARCHAR(100),        -- 'career', 'academic', 'social', 'ai', 'startup'
    tags TEXT[],
    source_url TEXT,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Sessions (optional, for personalization)
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE,
    interests TEXT[],
    role VARCHAR(50),             -- 'undergrad', 'grad', 'international'
    created_at TIMESTAMP DEFAULT NOW()
);
