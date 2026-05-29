"""
bot/listener.py — Mattermost WebSocket event handler.

Connects to Mattermost via WebSocket, listens for posted messages,
filters for mentions and DMs, and dispatches to the responder.

Architecture note:
  mattermostdriver calls loop.run_until_complete() internally for its WebSocket,
  which conflicts with an already-running asyncio event loop. We solve this by
  running the driver in a dedicated background thread with its own event loop.
  Async work (LLM calls, RAG, etc.) is bridged back to the main loop via
  asyncio.run_coroutine_threadsafe().
"""

import asyncio
import logging
import re
import ssl
import threading

from mattermostdriver import Driver
import mattermostdriver.websocket as mm_websocket

import config
from bot import responder
from bot.channel_reader import build_image_inputs_from_file_ids, build_text_inputs_from_file_ids
from llm.prompts import HELP_MESSAGE

logger = logging.getLogger(__name__)

# Global state
_driver: Driver | None = None
_bot_user_id: str = ""
_bot_username: str = ""
_main_loop: asyncio.AbstractEventLoop | None = None  # the main asyncio loop
_ssl_context_patched = False


def _patch_mattermostdriver_ssl_context() -> None:
    global _ssl_context_patched

    if _ssl_context_patched:
        return

    original_create_default_context = mm_websocket.ssl.create_default_context

    def _create_default_context(*args, **kwargs):
        purpose = kwargs.get("purpose")
        if purpose is ssl.Purpose.CLIENT_AUTH:
            kwargs = dict(kwargs)
            kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
        return original_create_default_context(*args, **kwargs)

    mm_websocket.ssl.create_default_context = _create_default_context
    _ssl_context_patched = True


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _patch_mattermostdriver_ssl_context()
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


def _run_driver_thread():
    """
    Entry point for the background thread.
    Creates its own event loop so mattermostdriver can call
    loop.run_until_complete() freely without conflicting with the main loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        driver = get_driver()
        # Wrap the handler to catch exceptions
        loop.run_until_complete(driver.init_websocket(_wrapped_event_handler))
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        loop.close()


async def _wrapped_event_handler(event):
    """Wrapper that safely calls the main event handler with error handling."""
    try:
        await _handle_event(event)
    except Exception as e:
        logger.error("Error in event handler: %s", e, exc_info=True)


async def connect():
    """
    Log in (sync, fine on main thread), then spin up the WebSocket
    listener in a background thread to avoid event loop conflicts.
    Blocks until the thread exits (i.e. runs forever).
    """
    global _bot_user_id, _bot_username, _main_loop

    _main_loop = asyncio.get_running_loop()

    driver = get_driver()
    driver.login()

    me = driver.users.get_user("me")
    _bot_user_id = me["id"]
    _bot_username = me["username"]
    logger.info("Logged in as @%s (id=%s)", _bot_username, _bot_user_id)

    logger.info("Starting WebSocket listener thread...")

    thread = threading.Thread(target=_run_driver_thread, daemon=True, name="mm-websocket")
    thread.start()

    # Keep the main coroutine alive while the thread runs
    while thread.is_alive():
        await asyncio.sleep(1)


# ── Event dispatch ───────────────────────────────────────────────────────────

async def _handle_event(event):
    """
    Top-level WebSocket event handler — runs inside the driver thread's loop.
    Dispatches real work to the main event loop so all async I/O
    (httpx clients, ChromaDB, etc.) stays on one loop.
    """
    import json as _json
    
    # Handle case where event might be a string
    if isinstance(event, str):
        try:
            event = _json.loads(event)
        except Exception:
            logger.debug("Failed to parse event as JSON: %s", event)
            return
    
    if not isinstance(event, dict):
        logger.debug("Event is not a dict: %s", type(event))
        return
    
    event_type = event.get("event")

    if event_type == "posted":
        if _main_loop is None:
            return
        # Schedule the coroutine on the main loop from this thread
        asyncio.run_coroutine_threadsafe(_handle_post(event), _main_loop)


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
    file_ids: list[str] = post.get("file_ids") or []
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

        # Extract image attachments (screenshots, photos, diagrams) for multimodal models.
        image_inputs: list[dict] = []
        text_inputs: list[dict] = []
        if file_ids:
            if config.MM_ENABLE_IMAGE_INPUT:
                image_inputs = await build_image_inputs_from_file_ids(
                    file_ids,
                    max_images=config.MM_MAX_IMAGES_PER_MESSAGE,
                    max_image_bytes=config.MM_MAX_IMAGE_BYTES,
                )

            # Also attempt to extract text from attachments (PDF/PPTX/DOCX/TXT)
            try:
                text_inputs = await build_text_inputs_from_file_ids(
                    file_ids,
                    max_files=3,
                    max_bytes=config.MM_MAX_IMAGE_BYTES * 2,
                )
            except Exception:
                logger.exception("Text extraction from attachments failed")

        # Generate response
        answer = await responder.generate_response(
            question=question,
            conversation_history=conversation_history,
            image_inputs=image_inputs,
            text_inputs=text_inputs,
        )

        # Reply in thread
        _post_reply(channel_id, root_id, answer)

        print("LLM returned final answer")

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
    
    if cmd == "ping":
        _post_reply(channel_id, root_id, "Pong! 🏓")
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