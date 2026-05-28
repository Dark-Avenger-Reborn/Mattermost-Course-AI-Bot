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
Drop course files into `course_material/` then run:
```bash
python scripts/ingest.py
```

Supported and Unsupported File Formats

| **Supported Formats**                                      | **Not Supported (Skipped or Ignored)**                                     |
|------------------------------------------------------------|----------------------------------------------------------------------------|
| PDF: `.pdf`                                                | Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.ico`, `.webp`   |
| PowerPoint: `.pptx`, `.ppt`                                | Audio/Video: `.mp3`, `.mp4`, `.wav`, `.avi`, `.mov`, `.mkv`                |
| Word: `.docx`                                              | Archives: `.zip`, `.tar`, `.gz`, `.rar`, `.7z`                             |
| Excel: `.xlsx`, `.xlsm`, `.xlsb`                           | Executables/Libraries: `.exe`, `.dll`, `.so`, `.dylib`                     |
| HTML: `.html`, `.htm`                                      | Python cache: `.pyc`, `.pyo`, `__pycache__`                                |
| Markdown: `.md`                                            | Misc: `.DS_Store`, `.gitignore`, `.git`, `.db`, `.sqlite`, `.sqlite3`      |
| Plain text: `.txt`, `.csv`, `.json`, `.yaml`, `.yml`,      | **Files starting with a dot (`.`)** (e.g., `.env`, `.config`)              |
| `.xml`, `.rst`, `.tex`                                     | **Binary files or files unreadable as UTF-8**                              |
| Source/code/scripts: `.py`, `.js`, `.ts`, `.java`, `.c`,   |                                                                            |
| `.cpp`, `.h`, `.sh`, `.bat`, `.ps1`, `.r`, `.sql`          |                                                                            |
| **Any unknown text-readable file**                         |                                                                            |

> **Note:**  
> - Files not explicitly listed as supported are attempted as UTF-8 text—if not readable, they are ignored/skipped.  
> - Binary and media files are skipped to avoid ingesting non-text data.  
> - Files and folders starting with a dot (.) are skipped by default.


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

## Image Attachments (Screenshots)

The bot can pass attached images (screenshots, photos, diagrams) to a multimodal chat model.

- Supported input: Mattermost post attachments with image MIME types (`image/*`)
- The bot converts images to data URLs and includes them in the chat request
- Non-image files are ignored

## Other Accepted Attachment Types

- **Images:** any image MIME type (`image/*`) — e.g. `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.ico`, `.webp`. Images are converted to data URLs and included in the chat request for multimodal models. Controlled by `MM_ENABLE_IMAGE_INPUT`, `MM_MAX_IMAGES_PER_MESSAGE`, and `MM_MAX_IMAGE_BYTES`.
- **Documents (text extraction):** `.pdf`, `.pptx`, `.ppt`, `.docx`, `.doc`, `.txt`, `.md` — the bot will attempt to extract text from these attachments and include the extracted text in the prompt. The bot also attempts other text-readable files (CSV/JSON/HTML/source code) when possible. Text extraction is limited (default: up to 3 files per message, with size limits).
- **Ignored / Not used for ingestion:** audio/video, archives, executables, and other binary media (e.g. `.mp3`, `.mp4`, `.zip`, `.exe`) are skipped for text ingestion. Non-image attachments that are not text-extractable will be ignored.

Environment variables:

- `MM_ENABLE_IMAGE_INPUT` (default: `true`)
- `MM_MAX_IMAGES_PER_MESSAGE` (default: `3`)
- `MM_MAX_IMAGE_BYTES` (default: `5000000`)

If your selected `CHAT_MODEL` is text-only, image attachments will still be sent but may be ignored by the model/provider.

## Commands

These work in DMs or any channel the bot is in:

| Command | Description |
|---|---|
| `@bot help` | Show help message |
| `@bot reload` | Re-ingest course material (admin only) |
| `@bot sources` | Show what course documents are loaded |
| `@bot ping` | Responds with `Pong! 🏓` if Online |
