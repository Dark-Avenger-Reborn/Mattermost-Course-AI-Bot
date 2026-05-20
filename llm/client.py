"""
llm/client.py — Async wrapper around the UB LiteLLM API (OpenAI-compatible).

Handles:
  - chat completions
  - embeddings
  - reranking (via /rerank endpoint)
"""

import asyncio
import logging
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

# Lazy-loaded local embedding model (SentenceTransformer)
_local_embed_model = None

# Shared async HTTP client (reused across all calls)
_http: httpx.AsyncClient | None = None


def _get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            base_url=config.UB_AI_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.UB_AI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    return _http


async def close():
    """Call on shutdown to cleanly close the HTTP client."""
    global _http
    if _http and not _http.is_closed:
        await _http.aclose()


# ── Chat completion ─────────────────────────────────────────────────────────

async def chat(
    messages: list[dict],
    model: str = config.CHAT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> str:
    """
    Send a chat completion request and return the assistant message string.
    """
    http = _get_http()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    logger.debug("chat() → model=%s, messages=%d", model, len(messages))

    resp = await http.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    # Some OpenAI-compatible providers can return structured content parts.
    if isinstance(content, list):
        text_parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(t for t in text_parts if t).strip()

    return str(content).strip()


# ── Embeddings ──────────────────────────────────────────────────────────────

async def embed(texts: list[str], model: str = config.EMBED_MODEL) -> list[list[float]]:
    """
    Embed a list of strings. Returns a list of float vectors.
    Batches automatically if the list is large.
    """
    # If configured to use local embeddings, use sentence-transformers
    if config.LOCAL_EMBEDDINGS:
        global _local_embed_model
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as _np
        except Exception as e:
            raise RuntimeError("Local embeddings requested but sentence-transformers is not installed") from e

        if _local_embed_model is None:
            _local_embed_model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

        BATCH = 64
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            # Run encoding in a thread to avoid blocking the event loop
            vecs = await asyncio.to_thread(_local_embed_model.encode, batch, show_progress_bar=False)

            # sentence-transformers may return numpy arrays
            for v in vecs:
                if isinstance(v, _np.ndarray):
                    all_vectors.append(v.tolist())
                else:
                    all_vectors.append(list(v))

        return all_vectors

    # Default: remote HTTP embeddings via UB/LiteLLM
    http = _get_http()
    BATCH = 64  # bge-m3 handles up to 8192 tokens per text; batch by count

    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        payload = {"model": model, "input": batch}
        resp = await http.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI format: data["data"] is a list of {index, embedding, object}
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        all_vectors.extend(item["embedding"] for item in sorted_items)

    return all_vectors


async def embed_one(text: str, model: str = config.EMBED_MODEL) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    vectors = await embed([text], model=model)
    return vectors[0]


# ── Reranking ───────────────────────────────────────────────────────────────

async def rerank(
    query: str,
    documents: list[str],
    top_n: int = config.RAG_RERANK_TOP_N,
    model: str = config.RERANK_MODEL,
) -> list[dict]:
    """
    Rerank documents against a query using bge-reranker-v2-m3.

    Returns a list of dicts sorted by relevance score descending:
        [{"index": int, "score": float, "document": str}, ...]
    """
    if not documents:
        return []

    http = _get_http()

    # LiteLLM exposes reranking at /rerank (Cohere-compatible format)
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
        "return_documents": True,
    }

    try:
        resp = await http.post("/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Cohere-style response: {"results": [{"index", "relevance_score", "document": {"text"}}]}
        results = data.get("results", [])
        return [
            {
                "index": r["index"],
                "score": r["relevance_score"],
                "document": r["document"]["text"] if isinstance(r.get("document"), dict) else documents[r["index"]],
            }
            for r in results
        ]

    except Exception as e:
        # Reranker is best-effort; fall back to original order if it fails
        logger.warning("Reranker failed (%s), using original order", e)
        return [
            {"index": i, "score": 1.0 - i * 0.01, "document": doc}
            for i, doc in enumerate(documents[:top_n])
        ]
