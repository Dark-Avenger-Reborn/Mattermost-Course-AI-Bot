#!/usr/bin/env python3
"""
scripts/ingest.py — Standalone script to ingest course material into ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --path /path/to/material
    python scripts/ingest.py --clear   # wipe and re-ingest
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import config
from rag.ingestor import get_collection, ingest_all, list_sources


async def main():
    parser = argparse.ArgumentParser(description="Ingest course material into ChromaDB")
    parser.add_argument("--path", default=config.COURSE_MATERIAL_PATH, help="Path to course material directory")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingesting")
    args = parser.parse_args()

    if args.clear:
        print("⚠️  Clearing existing course material from ChromaDB...")
        collection = get_collection()
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            print(f"   Deleted {len(all_ids)} existing chunks.")
        else:
            print("   Collection was already empty.")

    print(f"\n📂 Ingesting from: {args.path}")
    summary = await ingest_all(material_path=args.path)

    print(f"\n✅ Done! Ingested {summary['files']} files → {summary['chunks']} chunks")

    if summary["files"] > 0:
        print("\n📚 Loaded sources:")
        for s in list_sources():
            print(f"   - {s}")


if __name__ == "__main__":
    asyncio.run(main())
