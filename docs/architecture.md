1. Scoring math — where the points actually live
Criterion
Weight
What it really rewards
Usefulness, proactivity, user value
30%
Daily brief that's actually worth reading, watchlist alerts, "why this matters" reasoning
Product thinking / judgment
25%
Saying no to features. Conversational onboarding. Silence when there's nothing to say.
AI experience / conversational quality
20%
Memory across turns, clarifying questions, no command-bot feel
Depth of finance vertical
15%
Real market data, real filings, real numbers — not vibes
Engineering quality
10%
Deliberately last and smallest. Don't over-invest here.

Notice: 65% of the score (proactivity + product judgment + AI experience) comes from things that have almost nothing to do with how many integrations you wire up. Engineering polish is only 10%. This should completely reshape where you spend hours.
2. The 20% of features that drive 80% of the score
Build exactly these six things, nothing more:
Conversational onboarding (role, watchlist, brief time) — no forms, LLM-driven, skippable
Persistent memory + personalization — every answer should feel like it remembers who you are
One proactive daily brief that's genuinely well-reasoned (this is your single highest-leverage feature — it's literally the "proactivity" criterion made concrete)
Natural Q&A with live financial data + citations (stock price, news, earnings, SEC filings)
Document upload → conversational Q&A (PDF/earnings deck → ask questions)
Voice message support (cheap to add, disproportionately impresses judges — "wow it does voice too")
Deliberately cut: Gmail/Calendar/Sheets integrations, multi-vertical support (Legal/Healthcare/etc.), portfolio management, custom alert engines beyond one simple watchlist trigger. These are all "optional" in the brief and eat days for single-digit score impact. If you have time left at the end, add Gmail meeting-prep as a bonus — not before the core six are polished.
3. ArchitectureThat's the shape of it. A single LangGraph agent sits in the middle and owns all reasoning; everything else is a tool it calls or a trigger that wakes it up.
Why this shape, specifically:
One orchestrator, not five microservices. Judges score engineering at only 10% — a monolith that works flawlessly beats a "proper" distributed system that's flaky on demo day.
The scheduler and Telegram both call into the same LangGraph graph. A proactive brief and a user question run through identical reasoning — this is what makes the proactive brief feel intelligent instead of templated.
Postgres does double duty: relational tables for user profile/watchlist/preferences, plus pgvector for embedding conversation history and document chunks. One database, one connection pool, no Mongo needed for an MVP — cut it even though it's in your stack.
4. Recommended stack (mapped to what you already know)
Bot layer — aiogram (async, better voice/file handling than python-telegram-bot, plays well with FastAPI's async world)
Backend — FastAPI, exactly as planned. One route that receives Telegram updates via webhook, hands off to the agent.
Agent — LangGraph, single graph with:
a router node (decide: chat / research / document Q&A / onboarding step)
tool nodes (market data, SEC EDGAR, web search, RAG retrieval)
a memory-write node that runs after every turn to extract/update preferences
LLM — Claude (Sonnet-class) as the reasoning model. Use it for synthesis, "why this matters" reasoning, and document summarization — its long context is genuinely useful for earnings decks and 10-Ks. Model choice isn't scored, but Claude's tool-calling reliability inside LangGraph will save you debugging time.
Financial data — pick one primary source to avoid integration sprawl:
Finnhub (free tier, generous limits: quotes, news, earnings calendar, company profile) as primary
SEC EDGAR full-text search API (free, no key) for filings — this alone gives you real "depth of finance vertical" points cheaply
Skip FMP/Alpha Vantage/Polygon — redundant with Finnhub for an MVP
Document intelligence — unstructured or pdfplumber to parse PDFs → chunk → embed with text-embedding-3-small or Voyage → store in pgvector → retrieve top-k on question → Claude synthesizes
Voice — Telegram voice notes are OGG/Opus; transcribe with Whisper API (whisper-1 or gpt-4o-transcribe) — a few lines of code, disproportionate demo impact
Scheduling — APScheduler inside the FastAPI process. Skip Celery/Redis — one extra moving part you don't need at this scale, and it's one more thing that can fail during a live demo.
Deployment — Docker Compose: app (FastAPI+bot) + postgres. Two containers. Ship it on Railway/Render/Fly for a stable public webhook URL judges can actually message.
5. Telegram UX flow
The brief explicitly bans buttons and commands — lean into that, it's free "product judgment" points.
First message ever:
"Hey — I'm your financial assistant. I'll help you track markets, research companies, and dig into reports. Quick one to start: what's your role — investor, analyst, founder, something else?"
Then, conversationally, one question at a time, weaving in: what they follow, what kind of insight they want, when they want their brief. Every question ends with an implicit "or just skip ahead and ask me anything." If they say "just show me something," drop onboarding immediately and go useful — that's the single biggest "feels alive" signal you can give a judge in the first 30 seconds.
Ongoing chat, e.g.:
User: "Tell me about Apple" Bot: "Sure — are you after the latest news, how the stock's doing, or their last earnings?"
Ambiguity → one clarifying question, never a menu.
Daily brief (pushed proactively, not requested):
"Morning. Two things worth your attention: Nvidia beat on revenue but guided softer on margins — the stock's down 3% pre-market, likely margin-compression fears from tariffs. Also, the Fed's speaking at 2pm ET today — markets are pricing in a hawkish tone."
Note: two items, each with a reason, no headline dump. If nothing matters, send nothing that day — mention this explicitly to judges in your demo, it's a direct callout of a stated design principle.
Document upload: user just sends a PDF with no caption → "Got it — this looks like Q3 earnings for [Company]. Want a summary, or do you have specific questions?"
6. Feature → score justification
Feature
Scores against
Conversational onboarding, skippable
Product thinking (25%) — explicitly matches "not a form" requirement
Memory across sessions, personalized answers
AI experience (20%) + Product thinking (25%)
Daily brief with reasoning, silence when nothing matters
Proactivity (30%) — this is the criterion, made literal
Live Finnhub + SEC EDGAR data with clarifying questions
Finance depth (15%) + AI experience (20%)
Document upload → conversational Q&A
Finance depth (15%) — "financial document intelligence" is a named requirement
Voice message handling
AI experience (20%) — cheap, visible, judges will try it first
No buttons/commands anywhere
Product thinking (25%) — directly matches "avoid this" list in the brief

