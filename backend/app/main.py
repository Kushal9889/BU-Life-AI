import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from app.routes import query, places, events, resources

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="BU Life AI API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ALLOWED_ORIGINS = comma-separated list in env, e.g.:
# "http://localhost:3000,https://bulife-ai.netlify.app"
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api")
app.include_router(places.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(resources.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
