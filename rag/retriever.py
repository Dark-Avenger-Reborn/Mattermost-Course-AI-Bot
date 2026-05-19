"""
rag/retriever.py — Query pipeline: embed → ChromaDB search → rerank → return chunks.
"""

import logging

import config
from llm.client import embed_one, rerank
from rag.ingestor import get_collection

logger = logging.getLogger(__name__)


async def retrieve(query: str, top_k: int = config.RAG_TOP_K, top_n: int = config.RAG_RERANK_TOP_N) -> list[dict]:
    """
    Full RAG retrieval pipeline:
      1. Embed the query with bge-m3
      2. ANN search ChromaDB for top_k candidates
      3. Rerank with bge-reranker-v2-m3, keep top_n
      4. Return list of {text, source, page, score}

    Returns empty list if no course material is loaded.
    """
    collection = get_collection()

    # Check if anything is loaded
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — no course material ingested yet")
        return []

    # Step 1: Embed query
    query_vector = await embed_one(query)

    # Step 2: Vector search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    if not docs:
        return []

    # Step 3: Rerank
    reranked = await rerank(query, docs, top_n=top_n)

    # Step 4: Build final result list with metadata
    final = []
    for item in reranked:
        original_idx = item["index"]
        meta = metas[original_idx] if original_idx < len(metas) else {}
        final.append({
            "text": item["document"],
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "score": item["score"],
        })

    logger.debug("retrieve(%r) → %d chunks after rerank", query[:60], len(final))
    return final


def format_rag_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for the LLM prompt.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']}, page {chunk['page']}\n{chunk['text']}"
        )

    return "\n\n".join(parts)
