import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.3-70b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")


def get_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_KEY,
        max_retries=3,
        request_timeout=30,
    )


def get_embeddings():
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
    )
