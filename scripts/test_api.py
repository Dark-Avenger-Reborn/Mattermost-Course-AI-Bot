#!/usr/bin/env python3
"""
scripts/test_api.py — Smoke test for the UB LiteLLM API.

Tests chat, embeddings, and reranking endpoints.
Run this before starting the bot to verify connectivity.

Usage:
    python scripts/test_api.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from llm.client import chat, embed, embed_one, rerank


async def test_chat():
    print("🤖 Testing chat completion...")
    response = await chat(
        messages=[{"role": "user", "content": "Reply with exactly: 'Chat OK'"}],
        max_tokens=20,
        temperature=0.0,
    )
    print(f"   ✅ Response: {response!r}")
    return True


async def test_embeddings():
    print("🔢 Testing embeddings...")
    texts = ["Hello world", "Test embedding"]
    vectors = await embed(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    print(f"   ✅ Got {len(vectors)} vectors, dim={len(vectors[0])}")
    return True


async def test_reranker():
    print("📊 Testing reranker...")
    results = await rerank(
        query="What is Python?",
        documents=[
            "Python is a programming language.",
            "The weather is nice today.",
            "Python uses indentation for blocks.",
        ],
        top_n=2,
    )
    print(f"   ✅ Got {len(results)} reranked results")
    for r in results:
        print(f"      [{r['score']:.3f}] {r['document'][:60]}")
    return True


async def test_mattermost():
    print("💬 Testing Mattermost connection...")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{config.MM_URL}/api/v4/users/me",
            headers={"Authorization": f"Bearer {config.MM_TOKEN}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        me = resp.json()
        print(f"   ✅ Connected as @{me['username']} (id={me['id']})")
    return True


async def main():
    print(f"\n{'='*50}")
    print(f"UB AI Bot — API Connectivity Test")
    print(f"  LiteLLM URL: {config.UB_AI_BASE_URL}")
    print(f"  Chat model:  {config.CHAT_MODEL}")
    print(f"  Embed model: {config.EMBED_MODEL}")
    print(f"  MM URL:      {config.MM_URL}")
    print(f"{'='*50}\n")

    tests = [
        ("Mattermost", test_mattermost),
        ("Chat", test_chat),
        ("Embeddings", test_embeddings),
        ("Reranker", test_reranker),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
        print()

    print(f"{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed:
        print("\n⚠️  Fix the failures above before running the bot.")
        sys.exit(1)
    else:
        print("\n🎉 All systems go! Run the bot with: python -m bot.main")


if __name__ == "__main__":
    asyncio.run(main())
