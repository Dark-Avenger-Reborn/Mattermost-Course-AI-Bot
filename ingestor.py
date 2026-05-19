"""
rag/ingestor.py — Ingest course material (PDF, PPTX) into ChromaDB.

Pipeline:
  file → extract text → chunk → embed (bge-m3) → upsert into ChromaDB

Run via:  python scripts/ingest.py
Or call:  asyncio.run(ingest_all())
"""

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import chromadb
import fitz  # pymupdf
from pptx import Presentation
from tqdm import tqdm

import config
from llm.client import embed

logger = logging.getLogger(__name__)

# ── ChromaDB setup ──────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(
        name="course_material",
        metadata={"hnsw:space": "cosine"},
    )


# ── Text extraction ─────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> list[dict]:
    """Extract text per page from a PDF. Returns list of {page, text}."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_pptx(path: Path) -> list[dict]:
    """Extract text per slide from a PPTX. Returns list of {page, text}."""
    prs = Presentation(str(path))
    slides = []
    for i, slide in enumerate(prs.slides):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        parts.append(line)
        text = "\n".join(parts).strip()
        if text:
            slides.append({"page": i + 1, "text": text})
    return slides


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping word-count chunks.
    chunk_size and overlap are in words (not tokens).
    bge-m3 handles up to ~8192 tokens; 512 words ≈ 600-700 tokens, safe.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _chunk_id(source: str, page: int, chunk_idx: int) -> str:
    raw = f"{source}::p{page}::c{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main ingest pipeline ────────────────────────────────────────────────────

async def ingest_file(path: Path, collection: chromadb.Collection) -> int:
    """Ingest a single file. Returns number of chunks added."""
    suffix = path.suffix.lower()
    source_name = path.name

    if suffix == ".pdf":
        pages = extract_pdf(path)
    elif suffix in (".pptx", ".ppt"):
        pages = extract_pptx(path)
    else:
        logger.warning("Skipping unsupported file type: %s", path)
        return 0

    if not pages:
        logger.warning("No text extracted from %s", path)
        return 0

    # Build all chunks with metadata
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadata: list[dict] = []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for idx, chunk in enumerate(chunks):
            cid = _chunk_id(source_name, page_data["page"], idx)
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metadata.append({
                "source": source_name,
                "page": page_data["page"],
                "chunk_index": idx,
                "type": suffix.lstrip("."),
            })

    if not all_chunks:
        return 0

    # Embed in batches and upsert
    EMBED_BATCH = 32
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch_texts = all_chunks[i : i + EMBED_BATCH]
        batch_ids = all_ids[i : i + EMBED_BATCH]
        batch_meta = all_metadata[i : i + EMBED_BATCH]

        vectors = await embed(batch_texts)

        collection.upsert(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    logger.info("Ingested %s: %d chunks from %d pages", source_name, len(all_chunks), len(pages))
    return len(all_chunks)


async def ingest_all(material_path: str = config.COURSE_MATERIAL_PATH) -> dict:
    """
    Ingest all PDFs and PPTXs from the course material directory.
    Returns a summary dict.
    """
    path = Path(material_path)
    if not path.exists():
        path.mkdir(parents=True)
        logger.info("Created course material directory: %s", path)

    files = list(path.glob("**/*.pdf")) + list(path.glob("**/*.pptx")) + list(path.glob("**/*.ppt"))

    if not files:
        logger.warning("No course material files found in %s", path)
        return {"files": 0, "chunks": 0}

    collection = get_collection()
    total_chunks = 0

    for f in tqdm(files, desc="Ingesting course material"):
        chunks = await ingest_file(f, collection)
        total_chunks += chunks

    summary = {"files": len(files), "chunks": total_chunks}
    logger.info("Ingestion complete: %s", summary)
    return summary


def list_sources() -> list[str]:
    """Return a deduplicated list of ingested source document names."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    if not results or not results.get("metadatas"):
        return []
    sources = sorted({m["source"] for m in results["metadatas"] if m})
    return sources
