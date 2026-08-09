# Atlas AI Financial Assistant

An AI-powered Telegram financial assistant that provides personalized financial insights, real-time market data, financial document intelligence, voice-based interaction, proactive price alerts, and daily financial briefings.

## Features

### Core Features

- ✅ **AI Financial Assistant** — Conversational financial analysis through Telegram
- ✅ **Real-Time Market Data** — Live stock information using Finnhub
- ✅ **Company Research** — Get current prices, financial metrics, market information, and company insights
- ✅ **Company Comparison** — Compare two companies across key financial metrics
- ✅ **Financial Document Q&A** — Upload financial documents and ask questions using RAG
- ✅ **User Memory & Personalization** — Remembers user preferences, interests, and investment profile
- ✅ **Onboarding** — Personalized onboarding for understanding the user's investor profile and interests
- ✅ **Voice Messages** — Send financial questions through Telegram voice messages
- ✅ **Price Alerts** — Create alerts for stock price targets and receive Telegram notifications when conditions are met
- ✅ **Daily AI Briefing** — Receive personalized daily market and watchlist summaries

---

## Tech Stack

- **Python**
- **FastAPI** — Backend API
- **aiogram** — Telegram Bot
- **LangGraph** — Agent workflow and orchestration
- **Google Gemini** — LLM-powered financial analysis and responses
- **Finnhub API** — Real-time financial market data
- **SQLAlchemy** — Database ORM
- **PostgreSQL + pgvector** — Persistent data and vector search
- **Sentence Transformers** — Local document/query embeddings
- **Docker** — Containerization

---

## Architecture

```text
                    ┌─────────────────────┐
                    │   Telegram User     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Telegram Bot      │
                    │      aiogram        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LangGraph      │
                    │   Agent Workflow    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       Market Research     Document RAG     Personalization
              │                │                │
              ▼                ▼                ▼
         Finnhub API       pgvector       PostgreSQL
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Gemini LLM       │
                    │ Response Generation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Telegram Response  │
                    └─────────────────────┘
