Atlas AI Financial Assistant Hackathon — Requirement Analysis
1. Problem Summary
Finance professionals lose time context-switching across tools (market data, news, filings, calendars, email, docs) to stay informed and prepare for decisions. The task is to build a Telegram-native AI financial assistant that acts like a proactive analyst/executive assistant — not a chatbot wrapper — using only natural language (text/voice/image), that remembers context, personalizes over time, and surfaces only high-signal information (silence > noise). Finance is the mandatory primary vertical; everything else is optional.
2. Mandatory Features
Telegram bot as the sole interface, fully conversational — no slash commands, buttons, menus, or quick replies.
Input modalities: text, voice messages, images (must handle all three).
Conversational onboarding (not a form) that gradually gathers role, followed companies/sectors, insight preferences, briefing timing — while being fully skippable at every step.
Memory/context across sessions — conversational history, preferences, watchlists.
Proactive daily intelligence (briefings, alerts) that explains why something matters and stays silent when nothing matters.
Ambiguity handling — asks clarifying questions instead of guessing (e.g., "Tell me about Apple" → what dimension?).
Company & market research capability with sourced, structured answers (public and private companies).
Financial document intelligence — upload & conversationally query reports/filings/decks.
Live financial data retrieval — real prices/news/filings synthesized, not linked-out; explicit uncertainty when unverifiable.
At least one real financial data source integration (SEC EDGAR, Finnhub, FMP, Polygon, Alpha Vantage, etc.).
Personalization engine that improves over time from usage, not just onboarding answers.
Backend + database supporting profiles, conversation history, preferences, integrations, documents, memory.
Background jobs/scheduler for briefings and alerts.
Live, judge-testable deployed bot + demo video for submission (no source code required).
3. Optional Features
Additional verticals (startup ecosystem, business, tech, healthcare, education, legal, productivity) — only after finance is strong, and finance must remain primary.
Productivity integrations: Gmail, Google Calendar (summarization, meeting prep, action items).
Financial workspace integrations: Google Drive, Google Sheets (anomaly detection, model review).
Extra data providers: Bloomberg, Reuters, PitchBook, Crunchbase, portfolio tools.
"Additional creativity" features — anything genuinely valuable (portfolio intelligence, collaboration, accessibility, etc.), as a bonus layer only after core is solid.
Choice of AI model/framework is explicitly not judged.
4. Judging Criteria (with weights)
Criterion
Weight
Usefulness, proactivity, overall user value
30%
Product thinking, judgment, feature selection
25%
AI experience & conversational quality
20%
Depth of finance vertical
15%
Engineering quality & implementation
10%

Note the ordering: 75% of the score is product/UX/proactivity/conversation, only 10% is raw engineering. This is a product-judgment test disguised as a build task.
5. Hidden / Unstated Expectations
Restraint is a feature. The brief repeatedly punishes "doing more" — silence when nothing matters, skippable onboarding, no menus. Judges are likely testing whether you resist the urge to cram in features/integrations.
"Feels alive" in the closing note implies judges want to sense proactivity in the demo itself — e.g., an unprompted briefing arriving, not just reactive Q&A. A bot that only responds when spoken to will underperform even if technically complete.
No slash commands is a hard constraint but the LLM must still support commanding behavior (e.g., "track Tesla," "remind me before earnings," "alert me if it moves 5%") — meaning intent detection/entity extraction from free text is doing the job commands normally would. This is a nontrivial NLU/orchestration requirement disguised as a UX rule.
Clarification-seeking behavior is explicitly graded via example ("Tell me about Apple") — judges will likely test this directly in their live session.
Source reliability / epistemic honesty is being tested, not just retrieval — "avoid presenting unverified information as factual" suggests judges may probe with tricky/ambiguous financial questions to see if the bot hallucinates or hedges appropriately.
Judges will personally interact with the live bot, not just watch the video — meaning the bot needs to survive real, unscripted conversations, not a rehearsed demo path. Robustness to off-script input matters more than the video's polish.
Memory quality will likely be tested across a real session, not assumed from an early "remembers things" claim — e.g., "we're operating under Finance persona" continuity, referencing something said earlier in the conversation naturally.
Personalization has to visibly change the output, not just be stored — a memory system that's never applied won't read as personalized to a judge probing it live.
"Founding Engineer" framing signals they're evaluating ownership and taste, not scope. A narrow, deeply polished vertical experience is explicitly favored over broad shallow coverage (finance depth is a separate 15% line item from product thinking).
6. Technical Challenges
NLU/intent routing without commands: reliably parsing "track Tesla and notify me on SEC filings" into a persistent structured task (ticker, trigger type, notification channel) purely from free text.
Voice message handling: STT integration, plus deciding whether/how to reply in voice or text, and handling financial jargon/tickers in transcription accurately.
Image handling: parsing charts, screenshots of statements, or photographed documents — OCR/vision quality varies a lot with financial tables.
Long document ingestion inside chat: 10-K/annual report length vs. Telegram's terse, mobile-first response expectations — chunking, retrieval, and conversational (not dump-everything) summarization.
Reliable live financial data: rate limits, data freshness, and reconciling multiple sources (price feed vs. news feed vs. filings feed) into one coherent answer.
Uncertainty communication: getting an LLM to reliably say "I'm not confident" rather than confidently hallucinate stock moves/numbers.
Scheduling infrastructure: per-user timezone-aware daily briefings + reactive event-triggered alerts (e.g., 5% move) running as background jobs against live data.
Memory architecture: what to persist (raw history vs. summarized facts vs. structured preferences), retrieval at the right relevance/recency balance, and avoiding context bloat.
OAuth flows for Gmail/Calendar/Drive/Sheets inside a chat-only, buttonless UX — auth normally needs a web redirect; doing this conversationally in Telegram is awkward to make feel "natural."
"Stay silent" logic: building a materiality/significance filter for what's worth proactively pushing, which is a genuinely hard signal-vs-noise judgment call, not just a threshold.
7. Product Challenges
Defining "finance professional" narrowly enough to be deep, not generic. Depth of vertical is a distinct 15% criterion — an analyst's needs (comps, filings, ratios) differ from a founder's (fundraising, competitors) or a student's (education-first explanations). Trying to serve all personas equally risks shallowness.
Choosing 3–5 killer workflows over 20 mediocre ones — the brief explicitly warns against feature-count chasing; judges will penalize breadth-without-depth.
Designing onboarding that's short but still gets useful signal — too short = generic assistant, too long = violates "no forms" principle.
Making the notification cadence trustworthy — over-notify once and users (judges) will distrust "silence means nothing happened" for the rest of the demo.
Balancing conciseness with "explain why it matters" — every response needs analyst-level insight and brevity, which is a real tension in output design.
Integration justification — the brief explicitly asks participants to justify each integration; bolting on Gmail/Sheets without a clear workflow payoff will read as feature-padding, not value-add.
Demo-ability: because judges interact live, the product needs graceful behavior for unset-up integrations, cold-start (no history) users, and ambiguous queries — not just a golden-path scripted flow.
8. What Judges Will Actually Care About
Given the weights and tone of the brief, in priority order:
Does this feel like it actually saves me time and knows what matters, unprompted? (30% usefulness/proactivity — the single biggest bucket, and the thing explicitly called out in the closing note as "feels alive.")
Did they make smart, disciplined product choices — a tight, coherent set of workflows done well, clear reasoning for what was left out, not a feature checklist. (25%)
Does talking to it feel like talking to a competent analyst — natural language handling, good clarifying questions, memory that's actually used, appropriate hedging on uncertain data. (20%)
Is the finance vertical genuinely deep — real filings/data/analysis quality, not generic GPT financial chit-chat. (15%)
Engineering is a hygiene factor, not a differentiator — clean, working, modular code matters but is explicitly the smallest weight (10%). A dazzling architecture will not save a mediocre product experience.
Overall: judges are hiring-testing for product taste and restraint under an AI-hype framing — they want to see someone who could have built 30 things and chose the right 5.
