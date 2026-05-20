"""
bot/channel_reader.py — Fetch recent posts from Mattermost channels via REST API.

Uses the Mattermost REST API directly (not the WebSocket driver) since we need
one-off fetches, not a persistent connection.
"""

import json
import logging
import base64
from datetime import datetime, timezone

import httpx

import config

logger = logging.getLogger(__name__)

# Shared HTTP client for Mattermost REST calls
_mm_http: httpx.AsyncClient | None = None


def _get_mm_http() -> httpx.AsyncClient:
    global _mm_http
    if _mm_http is None or _mm_http.is_closed:
        _mm_http = httpx.AsyncClient(
            base_url=f"{config.MM_URL}/api/v4",
            headers={
                "Authorization": f"Bearer {config.MM_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    return _mm_http


async def close():
    global _mm_http
    if _mm_http and not _mm_http.is_closed:
        await _mm_http.aclose()


async def build_image_inputs_from_file_ids(
    file_ids: list[str],
    *,
    max_images: int,
    max_image_bytes: int,
) -> list[dict]:
    """
    Convert Mattermost file attachments into OpenAI-compatible image input parts.

    Returns content parts shaped like:
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    http = _get_mm_http()
    image_parts: list[dict] = []

    for file_id in file_ids:
        if len(image_parts) >= max_images:
            break

        try:
            info_resp = await http.get(f"/files/{file_id}/info")
            info_resp.raise_for_status()
            info = info_resp.json()

            mime_type = (info.get("mime_type") or "").lower()
            if not mime_type.startswith("image/"):
                continue

            size = int(info.get("size") or 0)
            if size > max_image_bytes:
                logger.info(
                    "Skipping image %s (%d bytes): exceeds MM_MAX_IMAGE_BYTES=%d",
                    file_id,
                    size,
                    max_image_bytes,
                )
                continue

            file_resp = await http.get(f"/files/{file_id}")
            file_resp.raise_for_status()
            encoded = base64.b64encode(file_resp.content).decode("ascii")
            data_url = f"data:{mime_type};base64,{encoded}"

            image_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        except Exception as e:
            logger.warning("Failed to process attachment %s as image: %s", file_id, e)

    return image_parts


# ── Channel metadata ────────────────────────────────────────────────────────

async def get_channel_info(channel_id: str) -> dict | None:
    """Fetch basic info (name, display_name) for a channel."""
    http = _get_mm_http()
    try:
        resp = await http.get(f"/channels/{channel_id}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Could not fetch channel info for %s: %s", channel_id, e)
        return None


async def get_all_channel_summaries() -> str:
    """
    Build a summary string of all configured context channels for the routing prompt.
    Format: "channel_id | #display-name | purpose"
    """
    if not config.CONTEXT_CHANNEL_IDS:
        return "(no channels configured)"

    lines = []
    for cid in config.CONTEXT_CHANNEL_IDS:
        info = await get_channel_info(cid)
        if info:
            name = info.get("display_name") or info.get("name", cid)
            purpose = info.get("purpose", "")
            lines.append(f"- ID: `{cid}` | #{name}" + (f" — {purpose}" if purpose else ""))
        else:
            lines.append(f"- ID: `{cid}` | (could not fetch info)")

    return "\n".join(lines)


# ── Post fetching ───────────────────────────────────────────────────────────

async def fetch_channel_posts(
    channel_id: str,
    limit: int = config.CHANNEL_FETCH_LIMIT,
) -> list[dict]:
    """
    Fetch the most recent `limit` posts from a channel.
    Returns list of {author, timestamp, message} sorted oldest→newest.
    """
    http = _get_mm_http()
    try:
        resp = await http.get(
            f"/channels/{channel_id}/posts",
            params={"per_page": limit, "page": 0},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch posts from channel %s: %s", channel_id, e)
        return []

    posts = data.get("posts", {})
    order = data.get("order", [])  # newest first

    # Resolve usernames for a sample of posts (batch lookup)
    user_ids = list({p.get("user_id") for p in posts.values() if p.get("user_id")})
    username_map = await _resolve_usernames(user_ids)

    result = []
    for post_id in reversed(order):  # oldest first
        post = posts.get(post_id, {})
        message = post.get("message", "").strip()
        if not message:
            continue

        ts = post.get("create_at", 0)
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        author = username_map.get(post.get("user_id", ""), "unknown")

        result.append({
            "author": author,
            "timestamp": dt,
            "message": message,
        })

    return result


async def _resolve_usernames(user_ids: list[str]) -> dict[str, str]:
    """Batch resolve user IDs to usernames."""
    if not user_ids:
        return {}

    http = _get_mm_http()
    try:
        resp = await http.post("/users/ids", json=user_ids)
        resp.raise_for_status()
        users = resp.json()
        return {u["id"]: u.get("username", u["id"]) for u in users}
    except Exception as e:
        logger.warning("Could not resolve usernames: %s", e)
        return {}


# ── Multi-channel fetch ─────────────────────────────────────────────────────

async def fetch_channels_context(channel_ids: list[str]) -> str:
    """
    Fetch posts from multiple channels and format them as a context block.
    """
    if not channel_ids:
        return ""

    all_sections = []
    for cid in channel_ids:
        info = await get_channel_info(cid)
        channel_name = info.get("display_name", cid) if info else cid

        posts = await fetch_channel_posts(cid)
        if not posts:
            continue

        lines = [f"### #{channel_name}"]
        for p in posts:
            lines.append(f"**{p['author']}** ({p['timestamp']}): {p['message']}")
        all_sections.append("\n".join(lines))

    return "\n\n".join(all_sections)
