"""
bot/responder.py — Core pipeline for generating a bot response.

Orchestrates:
  1. RAG retrieval
  2. Channel routing decision
  3. Channel context fetch (if needed)
  4. Final answer generation
"""

import json
import logging
import re

import config
from bot.channel_reader import fetch_channels_context, get_all_channel_summaries, get_channel_info
from llm.client import chat
from llm.prompts import answer_prompt, channel_routing_prompt
from rag.ingestor import list_sources
from rag.retriever import format_rag_context, retrieve

logger = logging.getLogger(__name__)


async def generate_response(
    question: str,
    conversation_history: list[dict] | None = None,
    image_inputs: list[dict] | None = None,
    text_inputs: list[dict] | None = None,
) -> str:
    """
    Full pipeline: question → answer string.

    Args:
        question: The student's question text (bot mention stripped)
        conversation_history: Prior messages in the thread (for context continuity)

    Returns:
        The bot's response string (Mattermost markdown)
    """

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────
    logger.info("Retrieving RAG context for: %r", question[:80])
    rag_chunks = await retrieve(question)
    rag_context = format_rag_context(rag_chunks)

    if rag_chunks:
        logger.info("RAG: %d chunks retrieved", len(rag_chunks))
    else:
        logger.info("RAG: no chunks retrieved (empty knowledge base or no match)")

    # ── Step 2: Channel routing decision ─────────────────────────────────
    channel_context = ""
    channel_names: list[str] = []
    if config.CONTEXT_CHANNEL_IDS:
        channel_context, channel_names = await _maybe_fetch_channel_context(question, rag_chunks)

    # ── Step 3: Generate final answer ─────────────────────────────────────
    messages = answer_prompt(
        question=question,
        rag_context=rag_context,
        channel_context=channel_context,
        conversation_history=conversation_history,
        image_inputs=image_inputs,
        text_inputs=text_inputs,
    )

    logger.info("Calling LLM for final answer...")
    answer = await chat(messages, temperature=0.3, max_tokens=1024)

    if not answer.strip():
        logger.warning("LLM returned empty response")
        return "I couldn't generate a complete answer just now. Please try rephrasing your question or sending it again."

    # Append source attribution if we used RAG
    if rag_chunks or channel_names:
        sources_used = sorted({c["source"] for c in rag_chunks}) if rag_chunks else []
        parts: list[str] = []
        if sources_used:
            parts.append(", ".join(f"`{s}`" for s in sources_used))
        if channel_names:
            parts.append(", ".join(f"#{n}" for n in channel_names))
        if parts:
            answer += "\n\n---\n📎 *Sources: " + ", ".join(parts) + "*"

    return answer


async def _maybe_fetch_channel_context(question: str, rag_chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Ask the LLM if channel context would help, then fetch it if so.
    Returns formatted channel context string (empty if not needed).
    """

    try:
        channel_summaries = await get_all_channel_summaries()
        routing_messages = channel_routing_prompt(question, channel_summaries)

        routing_response = await chat(
            routing_messages,
            model=config.CHAT_MODEL,
            temperature=0.0,
            max_tokens=20000,
        )

        print("Channel routing response:", routing_response)
        # Parse the JSON routing decision
        decision = _parse_json_safe(routing_response)

        if not decision or not decision.get("needs_channels"):
            logger.info("Channel routing: not needed (%s)", decision.get("reason", "?") if decision else "parse error")
            return "", []

        channel_ids = decision.get("channel_ids", [])

        # Safety: only allow channels that are in our configured list
        allowed = set(config.CONTEXT_CHANNEL_IDS)
        channel_ids = [cid for cid in channel_ids if cid in allowed]

        if not channel_ids:
            logger.info("Channel routing: needed but no valid IDs returned")
            return "", []

        logger.info("Channel routing: fetching %d channels: %s", len(channel_ids), channel_ids)
        # Resolve display names for the header
        channel_names: list[str] = []
        for cid in channel_ids:
            info = await get_channel_info(cid)
            if info:
                channel_names.append(info.get("display_name") or info.get("name", cid))
            else:
                channel_names.append(cid)

        context = await fetch_channels_context(channel_ids)
        return context, channel_names

    except Exception as e:
        logger.warning("Channel routing failed, skipping: %s", e)
        return "", []


def _question_needs_channel_context(question: str) -> bool:
    """Heuristic for questions that are likely answered by recent channel posts."""
    lowered = question.lower()
    cues = (
        "announcement",
        "announcements",
        "posted",
        "post",
        "mentioned",
        "said in class",
        "in class",
        "today",
        "this week",
        "last class",
        "recent",
        "zoom",
        "office hours",
        "deadline change",
    )
    return any(cue in lowered for cue in cues)


def _parse_json_safe(text: str) -> dict | None:
    """Try to extract and parse a JSON object from model output."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Command handlers ─────────────────────────────────────────────────────────

async def handle_sources_command() -> str:
    sources = list_sources()
    if not sources:
        return "📚 No course documents have been loaded yet. Ask an admin to run the ingestion script."
    lines = ["📚 **Loaded Course Documents:**\n"]
    for s in sources:
        lines.append(f"- `{s}`")
    return "\n".join(lines)
