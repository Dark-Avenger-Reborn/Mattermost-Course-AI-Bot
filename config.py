"""
config.py — Central configuration loaded from .env
All other modules import from here; nothing else touches os.environ directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def _csv(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


# ── Mattermost ─────────────────────────────────────────────────────────────
MM_URL: str = _require("MATTERMOST_URL").rstrip("/")
MM_TOKEN: str = _require("MATTERMOST_TOKEN")
MM_BOT_USERNAME: str = os.getenv("MATTERMOST_BOT_USERNAME", "course-bot")

# ── UB AI / LiteLLM ────────────────────────────────────────────────────────
UB_AI_BASE_URL: str = _require("UB_AI_BASE_URL").rstrip("/")
UB_AI_API_KEY: str = os.getenv("UB_AI_API_KEY", "placeholder")  # LiteLLM may not need a real key

# ── Models ─────────────────────────────────────────────────────────────────
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Use local embeddings instead of remote API if set (true/false)
LOCAL_EMBEDDINGS: bool = os.getenv("LOCAL_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
# Local embedding model name for sentence-transformers (only used when LOCAL_EMBEDDINGS is true)
LOCAL_EMBED_MODEL: str = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Course identity ─────────────────────────────────────────────────────────
COURSE_NAME: str = os.getenv("COURSE_NAME", "This Course")
COURSE_DESCRIPTION: str = os.getenv("COURSE_DESCRIPTION", "")
INSTRUCTOR_NAME: str = os.getenv("INSTRUCTOR_NAME", "the instructor")

# ── Channel context ─────────────────────────────────────────────────────────
CONTEXT_CHANNEL_IDS: list[str] = _csv("CONTEXT_CHANNEL_IDS")
CHANNEL_FETCH_LIMIT: int = int(os.getenv("CHANNEL_FETCH_LIMIT", "50"))

# ── Admin users ─────────────────────────────────────────────────────────────
ADMIN_USERNAMES: list[str] = _csv("ADMIN_USERNAMES")

# ── Multimodal (image attachments) ──────────────────────────────────────────
MM_ENABLE_IMAGE_INPUT: bool = os.getenv("MM_ENABLE_IMAGE_INPUT", "true").lower() in ("1", "true", "yes")
MM_MAX_IMAGES_PER_MESSAGE: int = int(os.getenv("MM_MAX_IMAGES_PER_MESSAGE", "3"))
MM_MAX_IMAGE_BYTES: int = int(os.getenv("MM_MAX_IMAGE_BYTES", "5000000"))

# ── RAG ─────────────────────────────────────────────────────────────────────
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
COURSE_MATERIAL_PATH: str = os.getenv("COURSE_MATERIAL_PATH", "./course_material")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
RAG_RERANK_TOP_N: int = int(os.getenv("RAG_RERANK_TOP_N", "4"))
