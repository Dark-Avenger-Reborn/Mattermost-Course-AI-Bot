"""
bot/listener.py — Mattermost WebSocket event handler.

Connects to Mattermost via WebSocket, listens for posted messages,
filters for mentions and DMs, and dispatches to the responder.
"""

import asyncio
import logging
import re

from mattermostdriver import Driver

import config
from bot import responder
from llm.prompts import HELP_MESSAGE

logger = logging.getLogger(__name__)

# Global driver instance
_driver: Driver | None = None
_bot_user_id: str = ""
_bot_username: str = ""


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        parsed = _parse_url(config.MM_URL)
        _driver = Driver({
            "url": parsed["host"],
            "token": config.MM_TOKEN,
            "scheme": parsed["scheme"],
            "port": parsed["port"],
            "verify": True,
            "timeout": 30,
        })
    return _driver


def _parse_url(url: str) -> dict:
    """Parse scheme, host, port from a URL string."""
    scheme = "https"
    if url.startswith("http://"):
        scheme = "http"
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]

    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        port = int(port_str)
    else:
        host = url
        port = 443 if scheme == "https" else 80

    return {"scheme": scheme, "host": host, "port": port}


async def connect():
    """Log in and start the WebSocket listener."""
    global _bot_user_id, _bot_username

    driver = get_driver()
    driver.login()

    me = driver.users.get_user("me")
    _bot_user_id = me["id"]
    _bot_username = me["username"]
    logger.info("Logged in as @%s (id=%s)", _bot_username, _bot_user_id)

    logger.info("Starting WebSocket listener...")
    await driver.init_websocket(_handle_event)


# ── Event dispatch ───────────────────────────────────────────────────────────

async def _handle_event(event: dict):
    """Top-level WebSocket event handler."""
    event_type = event.get("event")

    if event_type == "posted":
        await _handle_post(event)


async def _handle_post(event: dict):
    """Handle a new post event."""
    import json as _json

    # The post data is nested as a JSON string inside the event data
    data = event.get("data", {})
    post_raw = data.get("post")
    if not post_raw:
        return

    try:
        post = _json.loads(post_raw) if isinstance(post_raw, str) else post_raw
    except Exception:
        return

    # Skip our own messages
    if post.get("user_id") == _bot_user_id:
        return

    message: str = post.get("message", "").strip()
    channel_id: str = post.get("channel_id", "")
    post_id: str = post.get("id", "")
    root_id: str = post.get("root_id") or post_id  # thread root
    user_id: str = post.get("user_id", "")
    channel_type: str = data.get("channel_type", "")  # "D" = DM, "O"/"P" = channel

    # Determine if we should respond:
    # 1. Direct message
    # 2. Channel message that mentions us
    is_dm = channel_type == "D"
    bot_mention_pattern = re.compile(
        rf"@{re.escape(_bot_username)}\b", re.IGNORECASE
    )
    is_mention = bool(bot_mention_pattern.search(message))

    if not (is_dm or is_mention):
        return

    # Strip the mention from the question text
    question = bot_mention_pattern.sub("", message).strip()

    if not question:
        return

    logger.info(
        "Received question from user %s in channel %s: %r",
        user_id, channel_id, question[:100]
    )

    # Handle special commands
    if await _handle_commands(question, post_id, root_id, channel_id, user_id):
        return

    # Post a "thinking" reaction
    _add_reaction(post_id, "hourglass_flowing_sand")

    try:
        # Fetch thread history for context (if this is a reply in a thread)
        conversation_history = await _get_thread_history(root_id, post_id)

        # Generate response
        answer = await responder.generate_response(
            question=question,
            conversation_history=conversation_history,
        )

        # Reply in thread
        _post_reply(channel_id, root_id, answer)

    except Exception as e:
        logger.exception("Error generating response: %s", e)
        _post_reply(
            channel_id,
            root_id,
            "⚠️ Sorry, I ran into an error while processing your question. Please try again.",
        )
    finally:
        _remove_reaction(post_id, "hourglass_flowing_sand")


# ── Command handling ─────────────────────────────────────────────────────────

async def _handle_commands(
    text: str,
    post_id: str,
    root_id: str,
    channel_id: str,
    user_id: str,
) -> bool:
    """
    Check for special bot commands.
    Returns True if a command was handled (skip normal QA pipeline).
    """
    cmd = text.strip().lower()

    if cmd == "help":
        _post_reply(channel_id, root_id, HELP_MESSAGE)
        return True

    if cmd == "sources":
        answer = await responder.handle_sources_command()
        _post_reply(channel_id, root_id, answer)
        return True

    if cmd == "reload":
        username = _get_username(user_id)
        if username not in config.ADMIN_USERNAMES:
            _post_reply(channel_id, root_id, "⛔ Only admins can reload course material.")
            return True

        _post_reply(channel_id, root_id, "⏳ Re-ingesting course material, please wait...")
        try:
            from rag.ingestor import ingest_all
            summary = await ingest_all()
            _post_reply(
                channel_id,
                root_id,
                f"✅ Ingestion complete! Loaded **{summary['files']} files** → **{summary['chunks']} chunks**.",
            )
        except Exception as e:
            logger.exception("Reload failed: %s", e)
            _post_reply(channel_id, root_id, f"❌ Ingestion failed: `{e}`")
        return True

    return False


# ── Thread history ───────────────────────────────────────────────────────────

async def _get_thread_history(root_id: str, current_post_id: str) -> list[dict]:
    """
    Fetch earlier messages in the current thread to provide conversation context.
    Returns OpenAI-format message list (role/content).
    """
    if not root_id or root_id == current_post_id:
        return []  # This is the root post itself; no prior history

    driver = get_driver()
    try:
        thread = driver.posts.get_thread(root_id)
        posts = thread.get("posts", {})
        order = thread.get("order", [])

        history = []
        for pid in order:
            if pid == current_post_id:
                break
            post = posts.get(pid, {})
            msg = post.get("message", "").strip()
            if not msg:
                continue
            is_bot = post.get("user_id") == _bot_user_id
            history.append({
                "role": "assistant" if is_bot else "user",
                "content": msg,
            })

        # Keep only the last 6 turns to stay within context limits
        return history[-6:]
    except Exception as e:
        logger.warning("Could not fetch thread history: %s", e)
        return []


# ── Mattermost API helpers ───────────────────────────────────────────────────

def _post_reply(channel_id: str, root_id: str, message: str):
    driver = get_driver()
    try:
        driver.posts.create_post({
            "channel_id": channel_id,
            "message": message,
            "root_id": root_id,
        })
    except Exception as e:
        logger.error("Failed to post reply: %s", e)


def _add_reaction(post_id: str, emoji: str):
    driver = get_driver()
    try:
        driver.reactions.create_reaction({
            "user_id": _bot_user_id,
            "post_id": post_id,
            "emoji_name": emoji,
        })
    except Exception as e:
        logger.warning("Could not add reaction: %s", e)


def _remove_reaction(post_id: str, emoji: str):
    driver = get_driver()
    try:
        driver.reactions.delete_reaction(_bot_user_id, post_id, emoji)
    except Exception as e:
        logger.warning("Could not remove reaction: %s", e)


def _get_username(user_id: str) -> str:
    driver = get_driver()
    try:
        user = driver.users.get_user(user_id)
        return user.get("username", "")
    except Exception:
        return ""
