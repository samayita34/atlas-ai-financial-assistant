# Atlas — AI Financial Assistant

A Telegram-native financial research assistant built for a hackathon. Atlas combines a LangGraph agent workflow, Gemini LLM, real-time Finnhub market data, pgvector-backed document retrieval, voice transcription, a background APScheduler, and a conversational onboarding flow into a single FastAPI service.

---

## Problem

Getting useful financial context requires juggling multiple tools: a brokerage app for prices, a news aggregator for headlines, a PDF reader for earnings reports, and a spreadsheet for comparing companies. Atlas collapses that workflow into a single Telegram conversation. Ask a question, upload a filing, set a price alert, or request a briefing — all in one place, with no context switching.

---

## Features

| Feature | Status |
|---|---|
| Real-time company research via Finnhub | Implemented |
| Multi-company comparison (concurrent data fetch) | Implemented |
| Financial document Q&A with RAG + pgvector | Implemented |
| Voice message transcription and response | Implemented |
| Natural-language price alerts | Implemented |
| Daily AI-generated market briefing via scheduler | Implemented |
| Conversational onboarding and user personalization | Implemented |
| Watchlist management (add / remove / view) | Implemented |
| Persistent conversation history | Implemented |
| LangGraph intent routing with fallback extraction | Implemented |

---

## Architecture

### Request flow

```
Telegram user message
        |
        v
 telegram_bot.py          (aiogram handler layer)
        |
        v
  run_agent()             (LangGraph entrypoint)
        |
        v
  load_context_node       -> MemoryService: load user profile + last 10 turns
        |
        v (new user?)
  onboarding_node         -> 4-question flow, persists to User row, marks complete
        | (returning user)
        v
  classify_intent_node    -> Gemini JSON classification + fallback regex extraction
        |
        +---> company_research_node  -> FinancialDataService.get_company_overview()
        |                                 (quote + profile + metrics + news)
        +---> document_qa_node       -> DocumentService.get_context_for_query()
        |                                 (pgvector similarity search)
        +---> watchlist_node         -> WatchlistService add/remove/view
        +---> price_alert_node       -> AlertService.create_alert(), validates via Finnhub
        +---> conversation_node      -> pass-through (no tool call)
        +---> clarify_node           -> single clarifying question, END
        |
        v
  compose_response_node   -> Gemini text generation with personalized system prompt
        |
        v
 Telegram reply
```

### Background scheduler (APScheduler AsyncIOScheduler)

Three jobs run independently of the bot's polling loop:

| Job ID | Trigger | Behaviour |
|---|---|---|
| `daily_market_briefing` | Cron — time from `settings.default_brief_time` | Fetches watchlist data per user, generates a Gemini summary (<120 words), sends via Telegram |
| `watchlist_alert_check` | Interval — configurable, default 30 min | Checks all watchlist tickers; fires if intraday move >= configurable threshold (default 3%) |
| `price_alert_check` | Interval — configurable, default 1 min | Evaluates all active `PriceAlert` rows; deactivates before notifying to prevent double-fire |

All three jobs share the same `Bot`, `MemoryService`, `FinancialDataService`, and `WatchlistService` instances constructed at startup. The daily briefing also accepts a manual HTTP trigger (see [Testing](#testing)).

### LangGraph graph shape

```
load_context
    |
    +--[onboarding_completed=False]--> onboarding --> END
    |                                      |
    |                              [real request override]
    |                                      |
    +--------------------------------------+
    |
    v
classify_intent
    |
    +--[company_research]--> company_research_node --> compose_response --> END
    +--[document_qa]------> document_qa_node      --> compose_response --> END
    +--[watchlist]--------> watchlist_node         --> compose_response --> END
    +--[price_alert]------> price_alert_node       --> compose_response --> END
    +--[conversation]-----> conversation_node      --> compose_response --> END
    +--[ambiguous]--------> clarify_node                               --> END
```

The graph is compiled once at module import and reused across all requests.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Interface | Telegram Bot API via aiogram 3 |
| API server | FastAPI with async lifespan handler |
| Agent workflow | LangGraph (StateGraph) |
| LLM | Google Gemini (model name from `settings.gemini_model`) |
| Embeddings | Gemini `text-embedding-004` (768-dimensional) |
| Market data | Finnhub (quotes, company profiles, financial metrics, news) |
| Vector search | pgvector extension on PostgreSQL |
| Database ORM | SQLAlchemy 2.0 async with `AsyncSession` |
| Scheduler | APScheduler `AsyncIOScheduler` with `AsyncIOExecutor` |
| Voice transcription | `SpeechService` (wraps transcription API) |
| Config | Pydantic `BaseSettings` |

---

## Database Schema

Seven tables, all managed via SQLAlchemy declarative models:

| Table | Purpose |
|---|---|
| `users` | Identity, onboarding state, personalization columns (JSONB), briefing time |
| `conversation_messages` | Full per-user dialogue history, role-tagged |
| `documents` | Metadata for user-uploaded PDFs |
| `document_chunks` | Chunked text + 768-dim pgvector embedding per chunk |
| `daily_brief_log` | Audit log of briefing generation and delivery per user per day |
| `watchlist_items` | Per-user ticker subscriptions (unique on `(telegram_id, symbol)`) |
| `price_alerts` | User-created threshold alerts; deactivated on first trigger |

Personalization fields on `users`: `notes` (free text), `followed_companies` (JSONB), `followed_sectors` (JSONB), `insight_preferences` (JSONB), `brief_time` (Time), `timezone` (string), `onboarding_completed` (bool).

---

## Project Structure

```
app/
├── bot/
│   └── telegram_bot.py        # aiogram handlers, setup_bot(), run_agent() bridge
├── database/
│   ├── database.py            # Async engine, AsyncSessionLocal, init_db/close_db
│   └── models.py              # SQLAlchemy ORM models (7 tables)
├── langgraph/
│   └── graph.py               # LangGraph StateGraph, all nodes, run_agent()
├── prompts/
│   └── system_prompt.py       # ATLAS_PERSONA, per-intent prompts, compose_system_prompt()
├── services/
│   ├── alert_service.py       # PriceAlert CRUD
│   ├── document_service.py    # PDF ingestion, chunking, pgvector retrieval
│   ├── financial_data_service.py  # Finnhub wrapper (quote, overview, resolve_symbol)
│   ├── memory_service.py      # User CRUD, conversation history, onboarding/personalization
│   ├── speech_service.py      # Voice transcription
│   └── watchlist_service.py   # WatchlistItem CRUD with Finnhub ticker validation
├── config.py                  # Pydantic Settings
├── main.py                    # FastAPI app, lifespan handler, debug endpoint
└── scheduler.py               # APScheduler jobs (briefing, watchlist alerts, price alerts)
```

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension enabled:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- A Telegram bot token (from BotFather)
- A Gemini API key
- A Finnhub API key

### Environment variables

Create a `.env` file. Do not commit it.

```
TELEGRAM_BOT_TOKEN=...
GEMINI_API_KEY=...
FINNHUB_API_KEY=...
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/atlas
ENVIRONMENT=development          # or local, staging, production
GEMINI_MODEL=gemini-2.0-flash   # or whichever model you have access to
DEFAULT_BRIEF_TIME=08:00         # HH:MM
DEFAULT_BRIEF_TIMEZONE=Asia/Kolkata
LOG_LEVEL=INFO
```

### Install and run

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run database migrations / table creation
python -c "import asyncio; from app.database.database import init_db; asyncio.run(init_db())"

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The Telegram bot starts polling automatically as part of the FastAPI lifespan. The scheduler starts alongside it.

---

## Testing

### Health check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "ok", "bot_polling": true}
```

### Manual daily briefing trigger

Fires the exact same code path as the scheduled cron job. Pass `telegram_id` to target only your account (recommended during development):

```bash
curl -X POST "http://localhost:8000/debug/trigger-daily-briefing?telegram_id=YOUR_TELEGRAM_ID"
```

Expected response:
```json
{
  "telegram_id_filter": 123456789,
  "targets_found": 1,
  "briefings_sent": 1,
  "skipped_no_watchlist": 0,
  "failed": 0
}
```

This endpoint returns 403 when `ENVIRONMENT=production`.

---

## Example Interactions

**Company research**
```
User:   What is Nvidia's current stock situation?
Atlas:  NVDA is trading at $[price] ([change]% today). Revenue grew [X]%
        year-over-year. Recent headline: [top news item from Finnhub].
```

**Company comparison**
```
User:   Compare Apple and Microsoft
Atlas:  [Side-by-side summary of AAPL vs MSFT: price, P/E, revenue, key metrics]
```

**Document Q&A**
```
User:   [uploads earnings report PDF]
Atlas:  Document ingested. You can now ask questions about it.

User:   What was the gross margin in Q3?
Atlas:  According to the document, gross margin in Q3 was [X]%, up from [Y]%...
```

**Price alert**
```
User:   Alert me when AAPL goes above $230
Atlas:  Price alert set. I will notify you when AAPL trades above $230.00.
```

**Watchlist**
```
User:   Add TSLA and MSFT to my watchlist
Atlas:  Added TSLA and MSFT to your watchlist.

User:   Show my watchlist
Atlas:  Your watchlist: TSLA, MSFT
```

**Voice message**
```
User:   [sends voice note: "What is Tesla's current price?"]
Atlas:  Transcribed: "What is Tesla's current price?"
        [then proceeds with company research response]
```

**Daily briefing (received automatically at configured time)**
```
Atlas:  Good morning, [Name]. Here is your market update:
        - AAPL: $[price] ([change]%) — [headline]
        - NVDA: $[price] ([change]%) — [headline]
```

**Onboarding (first message from a new user)**
```
Atlas:  Hey! I'm Atlas, your financial assistant. I'll keep this quick —
        What best describes you — investor, analyst, founder, student, or
        finance professional?

User:   Beginner investor, focused on long-term growth

Atlas:  Which companies, sectors, or markets should I keep an eye on for you?

User:   Mainly technology stocks

Atlas:  What matters most to you day to day — market news, earnings, SEC
        filings, analyst ratings, or macro events?

User:   Earnings and market news

Atlas:  Last one — what time should I send your daily briefing?

User:   8:00 AM

Atlas:  That's everything I need for now — thanks! Ask me about a stock,
        add something to your watchlist, or upload a report any time.
```

Answers are persisted to the database. On subsequent requests, Atlas's system prompt includes the user's persona, followed companies, sectors, and insight preferences.

---

## Implementation Notes

**Intent classification** uses a two-pass approach: a Gemini JSON classification call with confidence scoring, followed by a deterministic regex fallback (`_fallback_entity_extraction`) that catches tickers, known company names, and phrasing patterns. Low-confidence results and the `AMBIGUOUS` class are routed to a clarifying question rather than guessing.

**RAG pipeline**: uploaded PDFs are chunked, each chunk is embedded with Gemini `text-embedding-004` (768 dimensions), and stored in the `document_chunks` table via pgvector. At query time, the user's question is embedded and a cosine similarity search retrieves the most relevant chunks, which are passed to the Gemini response prompt as grounding context.

**Price alerts** are deactivated in the database before their Telegram notification is sent. This ordering guarantees an alert cannot fire twice even if the notification delivery fails.

**Personalized system prompts** are assembled in `system_prompt.py` via `compose_system_prompt()`, which appends a `Personalization notes:` block containing the user's persona, followed companies, sectors, and insight preferences to whichever task-specific prompt is active. Users with no stored preferences receive an unmodified prompt.

**Concurrency** in the scheduler is bounded by a `asyncio.Semaphore(10)` shared across all Telegram send tasks within a single job run.

---

## Current Limitations

- Onboarding runs once per user. Preferences can be updated programmatically via `MemoryService.update_personalization()` but there is no Telegram command to edit them post-onboarding.
- The watchlist alert job uses percent change from the previous close as a proxy for intraday movement and does not deduplicate alerts already sent earlier in the same session.
- The service runs in long-polling mode. Webhook mode is architecturally straightforward to add but not currently implemented.
- No authentication layer on the FastAPI endpoints beyond the `ENVIRONMENT=production` guard on the debug trigger.
- Voice transcription quality depends on the configured `SpeechService` backend.

---

## Potential Extensions

- Per-user alert deduplication using a last-alerted-price cache in `MemoryService`
- Telegram commands for post-onboarding preference updates (`/preferences`, `/settime`)
- Webhook mode behind a reverse proxy for production deployments
- Support for additional document types beyond PDF
- Sector-level watchlist alerts in addition to individual ticker alerts
- Expanded financial data sources beyond Finnhub
