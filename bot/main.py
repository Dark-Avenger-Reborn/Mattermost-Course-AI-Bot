"""
bot/main.py — Entry point. Sets up logging and runs the bot.
"""

import asyncio
import logging
import signal
import sys

import config  # noqa: F401 — validates env vars on import
from bot import channel_reader, listener
from llm import client as llm_client


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mattermostdriver").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


async def _shutdown():
    logging.getLogger(__name__).info("Shutting down...")
    await llm_client.close()
    await channel_reader.close()


async def main():
    _setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("UB Course AI Bot starting")
    log.info("Course: %s", config.COURSE_NAME)
    log.info("Chat model: %s", config.CHAT_MODEL)
    log.info("Embed model: %s", config.EMBED_MODEL)
    log.info("Context channels: %d configured", len(config.CONTEXT_CHANNEL_IDS))
    log.info("=" * 60)

    try:
        loop = asyncio.get_running_loop()

        def _signal_handler():
            asyncio.ensure_future(_shutdown())
            loop.stop()

        # add_signal_handler is Unix-only; skip gracefully on Windows
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    except NotImplementedError:
        # Windows — Ctrl+C will still raise KeyboardInterrupt and be caught below
        pass

    try:
        await listener.connect()
    except KeyboardInterrupt:
        log.info("Interrupted, shutting down...")
        await _shutdown()


if __name__ == "__main__":
    asyncio.run(main())