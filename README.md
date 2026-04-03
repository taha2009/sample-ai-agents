# Google GenAI test

Small training-style examples for **Gemini** using the official [`google-genai`](https://github.com/googleapis/python-genai) Python SDK, plus optional **LangChain** (`langchain-google-genai`) variants of the same ideas.

## Prerequisites

- Python 3.10+ (3.13 works with these scripts; RAG uses **FastEmbed/ONNX**, not PyTorch)
- A [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)

## Setup

1. Create and activate a virtual environment (recommended):

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked by execution policy, run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

   **macOS / Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Environment variables:

   - Copy `.env.example` to `.env` (Windows: `copy .env.example .env`; macOS/Linux: `cp .env.example .env`).
   - Set **`GEMINI_API_KEY`** in `.env`.
   - Optionally set **`GEMINI_MODEL`** (default in code: `gemini-2.5-flash-lite` if unset).
   - For **RAG** scripts, optionally set **`EMBEDDING_MODEL`** (default: `BAAI/bge-small-en-v1.5` via FastEmbed).

   The apps load `.env` via `python-dotenv`. You can also export variables in the shell instead of using a file.

4. **LangChain tracing (optional):** If you use the `*_lc.py` scripts and want [LangSmith](https://docs.smith.langchain.com/) traces, fill in the `LANGSMITH_*` entries in `.env.example` / `.env`. They are not required for local runs.

## Run

Each script is a simple REPL: type messages, then **`exit`** to quit.

### Direct `google-genai` SDK

| Command | What it does |
|--------|----------------|
| `python agent.py` | Chat with conversation memory (`generate_content`). |
| `python tool_agent.py` | Chat with **tools**: `get_weather` (demo cities) and `get_latest_news` (demo topics), via SDK automatic function calling. |
| `python rag_agent.py` | **RAG**: in-memory **Chroma** + **FastEmbed** embeddings, then answer from retrieved context. |

### LangChain (`langchain-google-genai`)

| Command | What it does |
|--------|----------------|
| `python agent_lc.py` | Same idea as `agent.py`, using `ChatGoogleGenerativeAI` + message history. |
| `python tool_agent_lc.py` | Same tool demos as `tool_agent.py`, bound as LangChain tools. |
| `python rag_agent_lc.py` | Same RAG pipeline as `rag_agent.py`, generation via LangChain. |

`lc_transformers_shim.py` is imported by the `*_lc.py` scripts **before** LangChain so optional tokenizer paths that pull in `torch` are skipped. Do not use that shim for code that needs the real Hugging Face stack.

## Project layout

| Path | Purpose |
|------|---------|
| `agent.py` | Minimal Gemini CLI chat |
| `tool_agent.py` | Gemini + tools (weather / news demos) |
| `rag_agent.py` | Vector RAG (Chroma + FastEmbed + Gemini) |
| `agent_lc.py` / `tool_agent_lc.py` / `rag_agent_lc.py` | LangChain equivalents |
| `lc_transformers_shim.py` | Optional import shim for LangChain on fragile `torch` setups |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for `GEMINI_*`, optional `EMBEDDING_MODEL`, optional LangSmith |
| `.env` | Your secrets (create locally; ignored by git) |

## Notes

- **RAG:** First run may download a small ONNX embedding model from Hugging Face; the vector store is **ephemeral** (in RAM only).
- **Training:** Tool responses and news text are **hardcoded demos**, not live APIs.
