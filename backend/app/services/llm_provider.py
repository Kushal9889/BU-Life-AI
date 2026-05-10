import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")


def get_llm():
    """Groq-powered LLM — fast, free, reliable. No more NIM rate limit issues."""
    return ChatOpenAI(
        model="llama-3.3-70b-versatile",
        temperature=0,
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_KEY,
        max_retries=2,
        request_timeout=15,
    )


def get_embeddings():
    """NVIDIA NV-EmbedQA — used only for vector retrieval (embeddings already built)."""
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
    )
