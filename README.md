# Google GenAI test (Gemini CLI)

A small interactive CLI that chats with the Gemini API using the official `google-genai` Python SDK.

## Prerequisites

- Python 3.10+ (recommended)
- A Gemini API key

## Setup

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:

   - Copy the sample env file to `.env` (Windows: `copy .env.example .env`; macOS/Linux: `cp .env.example .env`).

   - Open `.env` and set `GEMINI_API_KEY` to your key.

   Optionally set `GEMINI_MODEL` to a model id (default if omitted: `gemini-2.5-flash-lite`).

   The app loads variables from `.env` via `python-dotenv`. You can also set `GEMINI_API_KEY` and `GEMINI_MODEL` in the shell instead of using a file.

## Run

```bash
python main.py
```

Type messages at the prompt; enter `exit` to quit.

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI loop and Gemini calls |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables (safe to commit) |
| `.env` | Your real key (ignored by git; create locally) |
