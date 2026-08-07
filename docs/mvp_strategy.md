1. Smallest MVP that could realistically score 90%+
A Telegram bot that:
Onboards you in ~4-5 conversational questions (role, companies/sectors followed, what kind of insights matter, briefing time) — skippable at any point
Holds natural conversation (text + voice) with real memory of what you've told it
Can research a company/stock on demand using one live financial data source + news, and explains why something matters, not just what happened
Can accept an uploaded document (PDF/earnings report) and answer questions about it conversationally
Sends one genuinely well-crafted proactive daily briefing (not a dump of headlines — 3-5 curated, personalized items with "why this matters")
Asks clarifying questions when a request is ambiguous ("Tell me about Apple" → "News, earnings, or valuation?")
That's it. One data source done well, one integration (not four), one proactive job done brilliantly. Judges are explicitly told to reward judgment, not breadth.
2. Absolutely mandatory features
Telegram bot, text + voice input (images at least gracefully handled)
No slash commands/buttons/menus — pure natural language
Conversational memory across turns (and ideally across sessions)
Conversational onboarding, skippable
Company/market research via live data (not hallucinated) with source-grounding
Ability to say "I'm not sure" instead of fabricating numbers
At least one proactive/scheduled push (daily brief or watchlist alert)
Ambiguity handling via clarifying questions
Document upload + conversational Q&A over it
3. Optional / safely skippable
Gmail, Calendar, Drive, Sheets integrations — nice signal but not core; one well-executed integration beats three shallow ones
Multi-vertical support (startup ecosystem, healthcare, etc.) — brief explicitly says finance-first, others "only after finance is well-developed." Skip entirely unless you have massive spare time.
Bloomberg/Reuters/PitchBook/Crunchbase — explicitly labeled optional
Portfolio management, collaboration features, "additional creativity" extras
Complex scheduling/meeting creation — cute but not core to the 30%/25% weighted criteria
4. With only 2 days, exact feature list
Day 1: Telegram bot skeleton + LLM conversation loop with persistent memory (DB-backed, not just session context) → conversational onboarding → one financial data API integrated (pick Financial Modeling Prep or Finnhub — good free tiers, clean data) → basic company research flow with source grounding.
Day 2: Document upload + Q&A (upload a 10-K or earnings deck, ask questions) → one scheduled job (personalized morning brief using stored preferences) → voice message handling (transcription in, natural reply out) → ambiguity-detection polish → then spend remaining hours purely on conversational tone/quality and demo rehearsal, not new features.
Do not touch Gmail/Calendar/Sheets in a 2-day sprint — the ROI is low relative to time cost, and a shallow integration can actually hurt "product judgment" scoring if it feels bolted-on.
5. The single feature that impresses judges most live
The proactive, personalized daily brief that explains "why it matters," delivered unprompted during the demo (or triggered live) — combined with the assistant referencing something from an earlier point in the conversation without being asked. That combination directly hits "feels alive, watches, learns, acts" from the closing note, and is the hardest thing for lazy teams to fake. A chatbot that answers well is table stakes; an assistant that clearly remembers you and pushes value unprompted is what separates a winner.
6. Demo order
Open with proactive intelligence, not a question you type. Show a brief already sitting in the chat, or trigger it live — this is your hook, it screams "not a chatbot."
Natural conversation + memory — ask something that implicitly relies on what it learned in onboarding ("what's moving in my watchlist today") to prove personalization is real, not scripted.
Ambiguity handling — ask a deliberately vague question ("tell me about Tesla") and let it ask a clarifying question. This is a cheap, high-signal moment judges will remember.
Document intelligence — upload a real earnings report, ask a pointed question, get a sharp synthesized answer with source grounding.
Close on voice input — send a voice message, get a fast, natural spoken-style reply. Ending on multimodal + speed leaves the best final impression.
Never demo raw feature lists or "here's integration #4" — every beat should look like a real workflow moment, per the brief's design principles.
7. Common mistakes other teams will make
Building a wide but shallow integration buffet (Gmail + Calendar + Sheets + 3 data APIs) to look impressive — this actively contradicts the brief and the 25% product-judgment criterion.
Using Telegram inline buttons/menus "just to make it easier" — explicitly banned, and judges will notice instantly.
Long, scroll-heavy responses — the brief explicitly warns against this; verbosity reads as "AI wrapper," not "analyst."
No real memory — session-only context that resets, which breaks the personalization illusion the moment the demo runs long.
Sending noisy, generic daily updates instead of a small number of high-conviction, explained ones — brief says silence is better than noise.
Overclaiming/fabricating financial data instead of citing sources or admitting uncertainty — a direct violation of the "accuracy is extremely important" requirement, and an easy way to lose credibility with judges who will test it.
Treating this as a wrapper around GPT with a system prompt — no visible reasoning over multiple sources, no document handling, no proactive behavior.
8. Making it feel intelligent, not like a chatbot
Reference the past unprompted — "Since you're tracking semis, this is relevant to Nvidia too" without being asked to cross-reference.
Editorialize, don't transcribe — always answer "why does this matter to you," using stored preferences, not just "here's what happened."
Silence as a feature — explicitly design a rule: if nothing meets a relevance bar, send nothing that day. State this in your demo narration — it's a strong product-judgment signal.
Ask before assuming — clarifying questions are one of the cheapest ways to look smart; a plain chatbot never does this.
Terse, structured, analyst-voice responses — short paragraphs or tight bullet synthesis, not walls of text. Sound like a sharp colleague, not a search engine.
Confidence calibration — explicitly flag uncertainty ("I couldn't confirm this from a reliable source") rather than a flat, uniformly confident tone — this alone differentiates you from 90% of hackathon bots.
9. Prioritized roadmap
Must Have
Telegram bot, text + voice input, no commands/buttons
Persistent memory (DB-backed user profile + conversation history)
Conversational, skippable onboarding
Company/market research grounded in one live financial data source
Document upload + conversational Q&A
Ambiguity → clarifying question behavior
One well-designed proactive job (daily brief, personalized, with "why it matters," silence when nothing qualifies)
Graceful uncertainty handling (no fabricated numbers)
Should Have
One productivity integration done well (Gmail or Calendar — pick one) for meeting prep or email-context enrichment
Watchlist-based custom alerts (e.g., "notify me if X moves 5%")
Multi-turn document comparison (compare two reports/companies)
Growing personalization from ongoing chats (not just onboarding-captured prefs)
Nice to Have
Google Sheets/Drive analysis
Second financial data source for redundancy/breadth
Additional verticals (only if finance is airtight and time remains)
Portfolio-level intelligence, collaboration features, advanced anomaly detection
Bloomberg/Reuters/PitchBook-style premium integrations
