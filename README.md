# UB Mattermost Course AI Bot

An AI-powered teaching assistant bot for Mattermost that uses RAG (Retrieval-Augmented Generation) over course materials and can reference class channels to answer student questions.

## Architecture

```
Mattermost WebSocket
        │
        ▼
   bot/main.py          ← Entry point, event loop
        │
        ├── bot/listener.py     ← WebSocket event handler
        ├── bot/responder.py    ← Reply logic, threading
        ├── bot/channel_reader.py ← Fetches channel history via REST
        │
        ├── rag/ingestor.py     ← PDF/PPTX → chunks → embed → ChromaDB
        ├── rag/retriever.py    ← Query → embed → search → rerank
        │
        ├── llm/client.py       ← LiteLLM/OpenAI-compat API wrapper
        ├── llm/prompts.py      ← All system/user prompt templates
        │
        └── config.py           ← Loads .env, central config
```

## Models Used

| Role | Model |
|---|---|
| Chat | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |
| Embeddings | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Ingest course material
Drop PDFs and PPTX files into `course_material/` then run:
```bash
python scripts/ingest.py
```

### 4. Run the bot
```bash
python -m bot.main
```

## How It Works

1. **Bot mentions / DMs** trigger the listener
2. The bot posts a ⏳ reaction while thinking
3. The question is embedded and searched against the ChromaDB vector store (course material)
4. Results are reranked for precision
5. The LLM decides if channel context is needed (via a routing call)
6. If yes, relevant channels are fetched and injected into context
7. Final answer is generated and posted as a thread reply

## Channel Access

The bot only reads channels it is a member of. Add it to channels in Mattermost, then list those channel IDs in `.env` under `CONTEXT_CHANNEL_IDS`.

## Commands

These work in DMs or any channel the bot is in:

| Command | Description |
|---|---|
| `@bot help` | Show help message |
| `@bot reload` | Re-ingest course material (admin only) |
| `@bot sources` | Show what course documents are loaded |
