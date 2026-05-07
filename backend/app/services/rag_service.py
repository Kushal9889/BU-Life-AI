import os
import logging

from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import Session

from app.services.llm_provider import get_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

embeddings = get_embeddings()
CONNECTION_STRING = os.getenv("DATABASE_URL", "postgresql://localhost/bulife")

# Singleton retriever — built once, reused for all queries
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
    """Lowercase tokenization so acronyms like CPT/OPT match case-insensitively."""
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
    seen_urls: set[str] = set()
    unique = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(doc)
    return "\n\n".join(
        f"Source: {d.metadata.get('title', 'BU Resource')} ({d.metadata.get('url', '')})\n{d.page_content[:1000]}"
        for d in unique[:3]
    )


def _extract_sources(docs: list[Document]) -> list[dict]:
    seen_urls: set[str] = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "title": doc.metadata.get("title", "BU Resource"),
                "url": url,
                "category": doc.metadata.get("category", ""),
            })
    return sources[:3]


# --- LCEL RAG Chain ---
_rag_prompt = ChatPromptTemplate.from_template(
    "Use the following BU resources to answer the student's question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Provide a helpful, cited answer:"
)


async def search_bu_resources(db: Session, query: str) -> dict:
    retriever = _build_retriever()

    # LCEL: RunnableParallel for concurrent context retrieval + question passthrough
    retrieval_chain = RunnableParallel(
        context=retriever | RunnableLambda(_format_docs),
        question=RunnablePassthrough(),
    )

    result = retrieval_chain.invoke(query)

    if not result["context"] or result["context"].strip() == "":
        return {
            "context": "No relevant BU resources found.",
            "sources": [],
        }

    # Get raw docs for source extraction (uses cached retriever, no extra embedding call)
    raw_docs = retriever.invoke(query)

    return {
        "context": result["context"],
        "sources": _extract_sources(raw_docs),
    }
