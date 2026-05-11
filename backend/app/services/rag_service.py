import os
import logging

from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from app.services.llm_provider import get_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

embeddings = get_embeddings()
CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql://localhost/bulife")

_retriever_instance = None


def _get_vectorstore() -> PGVector:
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="bu_resources",
        use_jsonb=True,
    )


def _load_all_docs() -> list[Document]:
    import json
    data_path = os.path.join(os.path.dirname(__file__), "../../../data/bu_resources.json")
    if not os.path.exists(data_path):
        return []
    with open(data_path) as f:
        resources = json.load(f)
    return [
        Document(
            page_content=f"{r['title']}. {r['content']}",
            metadata={"title": r["title"], "url": r["url"], "category": r["category"]},
        )
        for r in resources
    ]


def _preprocess_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _build_retriever():
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    vectorstore = _get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    all_docs = _load_all_docs()
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            k=5,
            preprocess_func=_preprocess_for_bm25,
        )
        _retriever_instance = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
        )
    else:
        _retriever_instance = vector_retriever

    return _retriever_instance


def _format_docs(docs: list[Document]) -> str:
    seen: set[str] = set()
    unique = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(doc)
    return "\n\n".join(
        f"Source: {d.metadata.get('title', 'BU Resource')} ({d.metadata.get('url', '')})\n{d.page_content[:1000]}"
        for d in unique[:3]
    )


def _extract_sources(docs: list[Document]) -> list[dict]:
    seen: set[str] = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url not in seen:
            seen.add(url)
            sources.append({
                "title": doc.metadata.get("title", "BU Resource"),
                "url": url,
                "category": doc.metadata.get("category", ""),
            })
    return sources[:3]


async def search_bu_resources(db: Session, query: str) -> dict:
    retriever = _build_retriever()

    # single retrieval pass, extract both context and sources
    chain = retriever | RunnableLambda(
        lambda docs: {
            "context": _format_docs(docs),
            "sources": _extract_sources(docs),
        }
    )

    result = chain.invoke(query)

    if not result["context"].strip():
        return {"context": "No relevant BU resources found.", "sources": []}

    return result
