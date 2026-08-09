from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
from datetime import time
from enum import Enum
from typing import Any, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph

from app.database.database import AsyncSessionLocal
from app.database.models import AlertCondition, ConversationMessage, User
from app.services.alert_service import AlertError, AlertService
from app.services.document_service import DocumentService
from app.services.financial_data_service import (
    FinancialDataError,
    FinancialDataService,
    SymbolNotFoundError,
)
from app.services.memory_service import MemoryService
from app.services.watchlist_service import (
    DuplicateTickerError,
    InvalidTickerError,
    TickerNotOnWatchlistError,
    WatchlistError,
    WatchlistFullError,
)

# TODO: No dedicated "LLM client" service was present in "Already
# Implemented". Assuming a thin Gemini wrapper is reasonable to construct
# directly here via google-genai, configured from app.config.settings.
# Adjust this import if a shared `app/services/llm_service.py` (or similar)
# already exists in the real codebase.
from app.config import Environment, get_settings

settings = get_settings()


def _is_dev_environment() -> bool:
    """Return True when running locally so developer-friendly error details
    are surfaced in Telegram instead of being swallowed by a generic message."""
    return settings.environment == Environment.LOCAL

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]

    _api_key = settings.gemini_api_key.get_secret_value()
    _genai_client = genai.Client(api_key=_api_key)
except Exception:  # pragma: no cover - allows module import without the SDK
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _genai_client = None


logger = logging.getLogger(__name__)

# TODO: Confirm actual model name/version to use via settings; assuming a
# reasonable default consistent with "Gemini API" in the stack.
GEMINI_MODEL_NAME: str = settings.gemini_model

MAX_RESPONSE_CHARS: int = 1200  # keep replies concise for Telegram
CONVERSATION_HISTORY_LIMIT: int = 10  # number of prior turns to load for context

# Sentinel prefix that telegram_bot.py detects to send the message as plain
# text (no ParseMode), preventing Markdown parse failures on raw tracebacks.
_DEV_ERROR_PREFIX = "[DEV-ERROR]"


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """Coarse-grained classification of what the user is asking for."""

    COMPANY_RESEARCH = "company_research"
    DOCUMENT_QA = "document_qa"
    WATCHLIST = "watchlist"
    PRICE_ALERT = "price_alert"
    CONVERSATION = "conversation"
    AMBIGUOUS = "ambiguous"


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class _AgentStateRequired(TypedDict, total=True):
    """Required keys always present in AgentState from the initial invocation."""

    user_id: int
    text: str
    services: "ServiceBundle"


class AgentState(_AgentStateRequired, total=False):
    """
    Shared state threaded through the LangGraph workflow.

    Required keys (always present): user_id, text, services.
    Optional keys (populated by nodes as the graph progresses): all others.
    """

    history: list[dict[str, str]]
    user_profile: Optional[Any]
    intent: Intent
    intent_confidence: float
    entities: dict[str, Any]
    clarification_needed: bool
    tool_result: Optional[str]
    response: str
    error: Optional[str]


class ServiceBundle(TypedDict):
    """Lightweight container so services can travel inside AgentState."""

    memory_service: MemoryService
    financial_data_service: FinancialDataService
    document_service: DocumentService
    watchlist_service: Optional[Any]



# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


async def _call_gemini(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    response_mime_type: Optional[str] = None,
) -> str:
    """
    Thin async wrapper around the Gemini API for text generation.

    TODO: Confirm the exact google-genai async call pattern used elsewhere
    in the codebase (if a shared llm client already exists, replace this
    helper with a call into it instead of instantiating genai here).
    """
    if _genai_client is None or genai_types is None:  # pragma: no cover - defensive fallback
        logger.error("Gemini client is not configured; returning empty response")
        return ""

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = await _genai_client.aio.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=config,
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Node: load context (memory + profile)
# ---------------------------------------------------------------------------


async def load_context_node(state: AgentState) -> AgentState:
    """
    Load recent conversation history and the user's profile/preferences
    from MemoryService so downstream nodes (classification, response
    generation) have context.
    """
    services = state["services"]
    telegram_id = state["user_id"]
    history: list[dict[str, str]] = []
    user_profile = None

    try:
        async with AsyncSessionLocal() as session:
            memory_service = MemoryService(session)

            # ConversationMessage.user_id is the internal UUID primary key
            # (User.id), NOT the raw Telegram ID. Resolve the user row
            # first so get_recent_messages is queried with the correct
            # identifier type -- passing the Telegram ID directly causes
            # "operator does not exist: uuid = bigint" at the DB level.
            user_profile = await memory_service.get_user_by_telegram_id(telegram_id)

            if user_profile is not None:
                raw_messages = await memory_service.get_recent_messages(
                    user_id=user_profile.id, limit=CONVERSATION_HISTORY_LIMIT
                )
                history = [
                    {"role": msg.role.value, "content": msg.content}
                    for msg in raw_messages
                ]
            # else: new/unknown user -- no history to load yet, keep history=[]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load context for user_id=%s", telegram_id)
        history = []
        user_profile = None
        state["error"] = f"context_load_failed: {exc}"

    state["history"] = history or []
    state["user_profile"] = user_profile
    return state


# ---------------------------------------------------------------------------
# Node: classify intent
# ---------------------------------------------------------------------------


_CLASSIFIER_SYSTEM_PROMPT = """You are the intent router for Atlas, a financial \
analyst assistant. Classify the user's latest message into exactly one \
category and extract relevant entities.

Categories:
- company_research: asking about a company, stock, ticker, market data, \
financial metrics, news, or comparisons between companies.
- document_qa: asking a question about a previously uploaded document \
(earnings report, SEC filing, annual report), or referring to "the report", \
"the filing", "this document", etc.
- watchlist: asking to add/remove/view tickers on their watchlist.
- price_alert: asking to be notified/alerted when a ticker's price goes \
above or below a specific dollar value.
- conversation: greetings, small talk, thanks, general assistant questions \
not requiring live data or documents.
- ambiguous: the request could plausibly map to more than one category \
above, or lacks enough information to act on (e.g. missing a company name \
or ticker where one is clearly required).

Respond ONLY with strict JSON of the form:
{"intent": "<category>", "confidence": <float 0-1>, "entities": {"tickers": \
[...], "company_names": [...]}, "reason": "<short reason>"}
"""


_KNOWN_COMPANY_MAP: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "adobe": "ADBE",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "paypal": "PYPL",
    "boeing": "BA",
    "walmart": "WMT",
    "disney": "DIS",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
}

_EXCLUDED_UPPERCASE_WORDS: set[str] = {
    "A", "I", "AN", "AM", "AT", "BY", "DO", "IN", "IS", "IT", "MY", "NO", "ON",
    "OR", "SO", "TO", "UP", "US", "WE", "FOR", "THE", "AND", "ARE", "BUT", "NOT",
    "YOU", "ALL", "ANY", "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY",
    "GET", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO",
    "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE",
    "CEO", "SEC", "PDF", "USD", "USA", "API", "AI", "Q1", "Q2", "Q3", "Q4",
    "YOY", "TTM", "PE", "EPS", "WHAT", "HOW", "TELL", "SHOW", "GIVE", "NEWS",
}


def _fallback_entity_extraction(
    text: str,
    intent: Intent,
    confidence: float,
    entities: dict[str, Any],
) -> tuple[Intent, float, dict[str, Any]]:
    """
    Fallback and enrichment helper for intent classification and entity extraction.
    Ensures company names and ticker symbols are reliably extracted even if Gemini
    fails, returns empty entities, or returns low confidence.
    """
    tickers: list[str] = list(entities.get("tickers") or [])
    company_names: list[str] = list(entities.get("company_names") or [])

    text_lower = text.lower()

    # 1. Check known company mapping
    for name, ticker in _KNOWN_COMPANY_MAP.items():
        if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
            if name.capitalize() not in company_names and name.upper() not in [c.upper() for c in company_names]:
                company_names.append(name.capitalize())
            if ticker not in tickers:
                tickers.append(ticker)

    # 2. Extract potential uppercase tickers (e.g. AAPL, MSFT, TSLA, NVDA)
    raw_words = re.findall(r"\b[A-Z]{2,5}\b", text)
    for word in raw_words:
        if word not in _EXCLUDED_UPPERCASE_WORDS and word not in tickers:
            tickers.append(word)

    # 3. Dynamic pattern matching for company names if still none found
    if not company_names and not tickers:
        patterns = [
            r"(?:about|tell me about|research|analyze|summary for|summarize|overview of|check|what is|how is)\s+([A-Z][a-zA-Z0-9\.\s]+?)(?:\s+(?:stock|financials|price|market|shares|earnings|revenue|report|data)|'s|\s*$|\?|\!)",
            r"([A-Z][a-zA-Z0-9\.]+)\s+(?:stock|financials|price|market|shares|earnings|revenue)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                clean_m = m.strip()
                if clean_m and clean_m.lower() not in ["the", "a", "an", "this", "that"]:
                    company_names.append(clean_m)

    updated_entities = {
        "tickers": tickers,
        "company_names": company_names,
    }

    # 4. Infer intent if financial/company research cues are present
    has_entities = bool(tickers or company_names)
    financial_keywords = [
        "stock", "financial", "financials", "price", "market", "earnings",
        "revenue", "share", "shares", "ticker", "company", "overview",
        "tell me about", "what is", "summarize", "analyze", "report"
    ]
    has_financial_cue = any(kw in text_lower for kw in financial_keywords)

    # Watchlist phrasing must win over the generic company-research
    # fallback below -- otherwise "add AAPL to my watchlist" or "remove
    # AAPL" get silently re-routed to company_research whenever a ticker
    # is present and Gemini's own confidence was low. Two signals:
    #   1. An explicit mention of "watchlist" (always a strong signal,
    #      even with no ticker -- e.g. "show my watchlist").
    #   2. An unambiguous removal verb ("remove"/"drop"/etc.) combined
    #      with a ticker -- narrower than "add"/"watch"/"track"/"follow"
    #      on purpose, since those verbs are common in ordinary
    #      company-research phrasing too (e.g. "keep track of Tesla's
    #      earnings") and would cause false positives if included here.
    _WATCHLIST_MENTION_KEYWORDS = ("watchlist", "watch list")
    _WATCHLIST_REMOVE_KEYWORDS = ("remove", "drop", "delete", "untrack", "unfollow")

    has_watchlist_mention = any(kw in text_lower for kw in _WATCHLIST_MENTION_KEYWORDS)
    has_watchlist_removal = has_entities and any(
        kw in text_lower for kw in _WATCHLIST_REMOVE_KEYWORDS
    )
    has_watchlist_cue = has_watchlist_mention or has_watchlist_removal

    # Price-alert phrasing must win over BOTH the watchlist and
    # company-research fallbacks below -- "alert me when AAPL goes above
    # $320" contains a ticker (like company research) and could contain
    # "watch"-adjacent wording, so it needs to be checked first. Requires
    # an explicit alert/notify phrase AND an above/below/over/under
    # direction word, so it doesn't fire on ordinary research phrasing.
    _PRICE_ALERT_KEYWORDS = (
        "alert me when", "notify me when", "alert when", "notify when", "price alert",
    )
    has_price_alert_cue = (
        has_entities
        and any(kw in text_lower for kw in _PRICE_ALERT_KEYWORDS)
        and any(
            word in text_lower for word in ("above", "below", "over", "under")
        )
    )

    if has_price_alert_cue:
        if intent == Intent.AMBIGUOUS or confidence < 0.45:
            intent = Intent.PRICE_ALERT
            confidence = max(confidence, 0.9)
    elif has_watchlist_cue:
        if intent == Intent.AMBIGUOUS or confidence < 0.45:
            intent = Intent.WATCHLIST
            confidence = max(confidence, 0.9)
    elif has_entities or has_financial_cue:
        if intent == Intent.AMBIGUOUS or confidence < 0.45:
            if has_entities:
                intent = Intent.COMPANY_RESEARCH
                confidence = max(confidence, 0.9)

    return intent, confidence, updated_entities


_ALERT_CONDITION_KEYWORDS: dict[str, AlertCondition] = {
    "greater than": AlertCondition.ABOVE,
    "above": AlertCondition.ABOVE,
    "over": AlertCondition.ABOVE,
    "exceeds": AlertCondition.ABOVE,
    "less than": AlertCondition.BELOW,
    "falls below": AlertCondition.BELOW,
    "below": AlertCondition.BELOW,
    "under": AlertCondition.BELOW,
}


def _parse_price_alert_request(
    text: str, tickers: list[str]
) -> Optional[tuple[str, AlertCondition, float]]:
    """
    Parse a price-alert request like "alert me when AAPL goes above $320"
    into (symbol, condition, target_price).

    Returns None if no ticker was extracted, or the message doesn't
    contain a recognizable condition keyword followed by a numeric price
    -- callers should treat None as "could not create the alert" and
    surface that as an error rather than guessing.
    """
    if not tickers:
        return None

    text_lower = text.lower()
    matched_condition: Optional[AlertCondition] = None
    keyword_end = -1

    for keyword, cond in _ALERT_CONDITION_KEYWORDS.items():
        pos = text_lower.find(keyword)
        if pos != -1:
            matched_condition = cond
            keyword_end = pos + len(keyword)
            break

    if matched_condition is None:
        return None

    # Look for a numeric price at or after the condition keyword (handles
    # "$320", "320", "320.50").
    remainder = text[keyword_end:]
    price_match = re.search(r"\$?\s*(\d+(?:\.\d+)?)", remainder)
    if price_match is None:
        return None

    try:
        target_price = float(price_match.group(1))
    except ValueError:
        return None

    return tickers[0], matched_condition, target_price


async def classify_intent_node(state: AgentState) -> AgentState:
    """
    Use the LLM to classify the user's message into an Intent, extract
    lightweight entities (tickers/company names), and flag ambiguity.
    """
    history_snippet = _format_history_for_prompt(state.get("history", []))
    prompt = (
        f"Conversation so far:\n{history_snippet}\n\n"
        f"Latest user message: {state['text']!r}\n\n"
        "Classify this latest message."
    )

    try:
        raw = await _call_gemini(
            prompt,
            system_instruction=_CLASSIFIER_SYSTEM_PROMPT,
            response_mime_type="application/json",
        )
        parsed = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        logger.exception("Intent classification failed; defaulting to ambiguous")
        parsed = {}

    intent_str = parsed.get("intent", Intent.AMBIGUOUS.value)
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.AMBIGUOUS

    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    entities = parsed.get("entities", {}) or {}

    # Run fallback entity extraction and intent enrichment
    intent, confidence, entities = _fallback_entity_extraction(
        state["text"], intent, confidence, entities
    )

    # Guardrail: company_research with no ticker/company name is ambiguous.
    if intent == Intent.COMPANY_RESEARCH and not (
        entities.get("tickers") or entities.get("company_names")
    ):
        intent = Intent.AMBIGUOUS

    # Low-confidence classifications are treated as ambiguous regardless
    # of the label the model picked.
    if confidence < 0.45:
        intent = Intent.AMBIGUOUS

    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["entities"] = entities
    state["clarification_needed"] = intent == Intent.AMBIGUOUS
    return state


def _format_history_for_prompt(history: list[dict[str, str]]) -> str:
    """Render recent turns as a compact transcript for prompting."""
    if not history:
        return "(no prior context)"
    lines = [f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: clarify (ambiguous requests)
# ---------------------------------------------------------------------------


async def clarify_node(state: AgentState) -> AgentState:
    """
    Generate a short, targeted clarifying question when the router could
    not confidently determine what the user wants (e.g. missing a ticker,
    missing which document they mean, or a genuinely ambiguous request).
    """
    prompt = (
        "The user sent this message to a financial analyst assistant, but "
        "it's ambiguous or missing key information (e.g. which company, "
        "ticker, or document they mean):\n\n"
        f"User message: {state['text']!r}\n\n"
        "Write ONE short, friendly clarifying question (max 25 words) to "
        "resolve the ambiguity. Do not answer the question yourself."
    )
    try:
        question = await _call_gemini(prompt)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to generate clarifying question")
        question = ""

    state["response"] = question or (
        "Could you clarify which company, ticker, or document you mean?"
    )
    return state


# ---------------------------------------------------------------------------
# Node: company research
# ---------------------------------------------------------------------------


async def company_research_node(state: AgentState) -> AgentState:
    """
    Fetch live financial data for the requested company/companies via
    FinancialDataService and summarize it concisely.

    Single-company requests behave exactly as before. When two entities
    are extracted (a comparison request, e.g. "compare Apple and
    Microsoft"), both companies are fetched concurrently via
    ``asyncio.gather`` and both datasets are passed downstream together.
    If one lookup fails, the other's data is still returned -- the
    failure is attributed to its specific ticker/name rather than
    silently dropped or fabricated.
    """
    services = state["services"]
    financial_data_service = services["financial_data_service"]
    entities = state.get("entities", {})

    tickers: list[str] = entities.get("tickers") or []
    company_names: list[str] = entities.get("company_names") or []

    # Comparison request: exactly two distinct entities were extracted.
    # Prefer tickers (more precise); fall back to company names only if
    # fewer than two tickers were extracted. Only ever compares the first
    # two -- this does not attempt an N-way comparison.
    if len(tickers) >= 2:
        compare_queries = tickers[:2]
    elif len(company_names) >= 2:
        compare_queries = company_names[:2]
    else:
        compare_queries = None

    if compare_queries is not None:
        results = await asyncio.gather(
            *(
                financial_data_service.get_company_overview(entity)
                for entity in compare_queries
            ),
            return_exceptions=True,
        )

        companies: dict[str, object] = {}
        unavailable: dict[str, str] = {}
        for entity, result in zip(compare_queries, results, strict=True):
            if isinstance(result, Exception):
                logger.exception(
                    "FinancialDataService lookup failed for %r", entity
                )
                unavailable[entity] = str(result)
            else:
                companies[entity] = result

        if not companies:
            # Both lookups failed -- same failure shape as the
            # single-company branch below.
            state["error"] = f"financial_lookup_failed: {unavailable}"
            state["tool_result"] = None
            return state

        payload: dict[str, object] = {"comparison": companies}
        if unavailable:
            payload["unavailable"] = unavailable
        state["tool_result"] = json.dumps(payload, default=str)
        return state

    query = tickers[0] if tickers else (company_names[0] if company_names else state["text"])

    try:
        # get_company_overview bundles quote + profile + metrics + news
        # and is the designated entrypoint in FinancialDataService.
        data = await financial_data_service.get_company_overview(query)
        state["tool_result"] = json.dumps(data, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("FinancialDataService lookup failed for %r", query)
        state["error"] = f"financial_lookup_failed: {exc}"
        state["tool_result"] = None

    return state


# ---------------------------------------------------------------------------
# Node: document Q&A
# ---------------------------------------------------------------------------


async def document_qa_node(state: AgentState) -> AgentState:
    """
    Answer a question against the user's most recently uploaded document(s)
    using DocumentService (pgvector-backed retrieval).
    """
    telegram_id = state["user_id"]

    # Document.user_id is the internal UUID primary key (User.id), NOT the
    # raw Telegram id -- same resolution load_context_node already performs
    # for conversation history. Reuse its result instead of re-querying.
    user_profile = state.get("user_profile")
    if user_profile is None:
        logger.error(
            "document_qa_node: no resolved user_profile for telegram_id=%s; "
            "cannot resolve internal User.id for document lookup",
            telegram_id,
        )
        state["error"] = "document_qa_failed: user_profile not resolved"
        state["tool_result"] = None
        return state

    try:
        async with AsyncSessionLocal() as session:
            # DocumentService requires a session at construction time.
            doc_service = DocumentService(session)
            context = await doc_service.get_context_for_query(
                user_id=user_profile.id,
                query=state["text"],
            )
        state["tool_result"] = context if context else "No relevant documents found."
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "DocumentService Q&A failed for user_id=%s", user_profile.id
        )
        state["error"] = f"document_qa_failed: {exc}"
        state["tool_result"] = None

    return state


# ---------------------------------------------------------------------------
# Node: onboarding
# ---------------------------------------------------------------------------

# Each entry: (User column to fill, question text, Gemini extraction hint).
# Order matters -- this *is* the onboarding sequence, inferred purely from
# which of these columns is still empty. No separate progress counter is
# needed, which means onboarding survives a mid-flow restart for free.
#
# "role" is intentionally NOT in this list: `User.role` is a typed
# Enum(UserRole) with only ADMIN/USER values (access control), not a
# persona field. The persona answer goes to `notes` (free Text) instead.
_ONBOARDING_FIELDS: list[tuple[str, str]] = [
    ("notes", "What best describes you — investor, analyst, founder, student, or finance professional?"),
    ("followed_companies", "Which companies, sectors, or markets should I keep an eye on for you?"),
    ("insight_preferences", "What matters most to you day to day — market news, earnings, SEC filings, analyst ratings, or macro events?"),
    ("brief_time", "Last one — what time should I send your daily briefing? (e.g. \"8:00 AM\"), or say \"skip\" if you'd rather not get one."),
]

_SKIP_PHRASES = ("skip", "not now", "no thanks", "maybe later", "later")


def _first_unset_onboarding_field(user: User) -> Optional[tuple[str, str]]:
    """Return the (column, question) for the first still-empty onboarding
    field, in order, or None if onboarding is complete."""
    for field_name, question in _ONBOARDING_FIELDS:
        if not getattr(user, field_name, None):
            return field_name, question
    return None


async def _extract_companies_or_sectors(text: str) -> dict[str, Any]:
    """One Gemini call: split a free-text answer into companies/sectors
    lists. Falls back to an empty structure on any failure so onboarding
    never gets stuck on a parsing error."""
    prompt = f'User said: {text!r}\n\nExtract company names/tickers and sector names mentioned.'
    try:
        raw = await _call_gemini(
            prompt,
            system_instruction=(
                'Respond ONLY with strict JSON: {"companies": [...], "sectors": [...]}. '
                "Use company names or tickers as given; empty arrays if none mentioned."
            ),
            response_mime_type="application/json",
        )
        parsed = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        logger.exception("Onboarding: failed to extract companies/sectors")
        parsed = {}
    return {
        "companies": parsed.get("companies") or [],
        "sectors": parsed.get("sectors") or [],
    }


async def _extract_insight_preferences(text: str) -> list[str]:
    """One Gemini call: normalize a free-text answer into a short list of
    insight-type tags. Falls back to storing the raw text as a single
    item so the answer is never silently dropped."""
    prompt = f'User said: {text!r}\n\nWhat type(s) of financial insight do they care about?'
    try:
        raw = await _call_gemini(
            prompt,
            system_instruction=(
                'Respond ONLY with strict JSON: {"preferences": [...]}, '
                "a short list of tags like \"market_news\", \"earnings\", \"sec_filings\", "
                '"analyst_ratings", "macro_events". Infer from free text.'
            ),
            response_mime_type="application/json",
        )
        parsed = json.loads(raw) if raw else {}
        prefs = parsed.get("preferences") or []
        return prefs if prefs else [text.strip()]
    except Exception:  # noqa: BLE001
        logger.exception("Onboarding: failed to extract insight preferences")
        return [text.strip()]


def _parse_brief_time(text: str) -> Optional[time]:
    """Best-effort parse of a free-text time answer into a `datetime.time`.
    Deliberately simple (a couple of common formats) rather than another
    Gemini call -- this field is the last question, so a parse miss should
    never block onboarding from completing."""
    cleaned = text.strip().lower().replace(".", "")
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", cleaned)
    if not match:
        return None
    hour_str, minute_str, meridiem = match.groups()
    hour = int(hour_str)
    minute = int(minute_str) if minute_str else 0
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour=hour, minute=minute)


async def onboarding_node(state: AgentState) -> AgentState:
    """
    Drive a short, skippable, conversational onboarding flow for
    first-time users instead of requiring the /start command.

    The next question is inferred from whichever User column is still
    unset -- naturally resumable across restarts, no progress counter
    needed. If the message looks like a real request (has a ticker/company
    per the same heuristics classify_intent uses) or says "skip", onboarding
    is marked complete immediately; a real request falls through to
    classify_intent for that same message instead of being blocked.
    """
    user_profile = state.get("user_profile")
    text = state["text"]
    text_lower = text.lower().strip()

    if user_profile is None:
        # Shouldn't happen -- telegram_bot.py always creates the user row
        # before calling run_agent -- but fail safe into normal handling
        # rather than blocking the user on a data inconsistency.
        return state

    wants_skip = any(p in text_lower for p in _SKIP_PHRASES)
    _, _, enriched_entities = _fallback_entity_extraction(text, Intent.AMBIGUOUS, 0.0, {})
    looks_like_real_request = bool(
        enriched_entities.get("tickers") or enriched_entities.get("company_names")
    )

    is_very_first_message = not state.get("history")

    async def _mark_complete_and_persist(**field_updates: Any) -> None:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, user_profile.id)
            if user is None:
                return
            for key, value in field_updates.items():
                setattr(user, key, value)
            user.onboarding_completed = True
            await session.commit()

    if wants_skip or (looks_like_real_request and not is_very_first_message):
        await _mark_complete_and_persist()
        if looks_like_real_request:
            # Let this exact message flow through to classify_intent below.
            return state
        state["response"] = (
            "No problem, we can skip that. Ask me anything, anytime — "
            "stock research, watchlists, or upload a report to dig into."
        )
        return state

    if is_very_first_message:
        _, first_question = _ONBOARDING_FIELDS[0]
        state["response"] = (
            "Hey! I'm Atlas, your financial assistant. I'll keep this quick — "
            f"{first_question}\n\n"
            '(Say "skip" any time to jump straight to asking me things.)'
        )
        return state

    # Not the first message and not a skip/real-request override: this
    # message is the user's answer to whichever field is still unset.
    pending = _first_unset_onboarding_field(user_profile)
    if pending is None:
        await _mark_complete_and_persist()
        return state

    field_name, _ = pending
    field_updates: dict[str, Any] = {}

    if field_name == "notes":
        field_updates["notes"] = f"Persona: {text.strip()}"
    elif field_name == "followed_companies":
        extracted = await _extract_companies_or_sectors(text)
        field_updates["followed_companies"] = extracted["companies"]
        field_updates["followed_sectors"] = extracted["sectors"]
    elif field_name == "insight_preferences":
        field_updates["insight_preferences"] = await _extract_insight_preferences(text)
    elif field_name == "brief_time":
        parsed_time = _parse_brief_time(text)
        if parsed_time is not None:
            field_updates["brief_time"] = parsed_time
        # If parsing fails, we intentionally still advance/complete rather
        # than looping -- this is the last question, and getting stuck here
        # would block a real user indefinitely over a formatting quirk.

    # Reflect the update on the in-memory object so
    # _first_unset_onboarding_field sees it as filled on this same pass.
    for key, value in field_updates.items():
        setattr(user_profile, key, value)

    next_pending = _first_unset_onboarding_field(user_profile)

    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_profile.id)
        if user is not None:
            for key, value in field_updates.items():
                setattr(user, key, value)
            if next_pending is None:
                user.onboarding_completed = True
            await session.commit()

    if next_pending is None:
        state["response"] = (
            "That's everything I need for now — thanks! Ask me about a "
            "stock, add something to your watchlist, or upload a report "
            "any time."
        )
    else:
        _, next_question = next_pending
        state["response"] = next_question

    return state


# ---------------------------------------------------------------------------
# Node: watchlist
# ---------------------------------------------------------------------------


async def watchlist_node(state: AgentState) -> AgentState:
    """
    Handle watchlist add/remove/view requests by calling WatchlistService
    directly. The specific action (add/remove/view) is inferred from
    keywords in the message text, combined with any tickers/company names
    already extracted during intent classification.
    """
    services = state["services"]
    watchlist_service = services.get("watchlist_service")
    telegram_id = state["user_id"]
    text_lower = state["text"].lower()
    entities = state.get("entities", {})
    tickers: list[str] = entities.get("tickers") or []

    if watchlist_service is None:
        state["tool_result"] = None
        state["error"] = "watchlist_service_unavailable"
        return state

    remove_keywords = ("remove", "drop", "untrack", "delete", "unfollow")
    add_keywords = ("add", "track", "watch", "follow", "monitor")

    if any(kw in text_lower for kw in remove_keywords):
        action = "remove"
    elif tickers:
        # Ticker(s) present with no explicit verb, or an explicit add verb
        # -- default to add, since that's the far more common phrasing.
        action = "add"
    else:
        action = "view"

    try:
        async with AsyncSessionLocal() as session:
            if action == "add" and tickers:
                added: list[str] = []
                failed: list[dict[str, str]] = []
                for ticker in tickers:
                    try:
                        entry = await watchlist_service.add_ticker(
                            session, telegram_id=telegram_id, symbol=ticker
                        )
                        added.append(entry.symbol)
                    except (InvalidTickerError, DuplicateTickerError, WatchlistFullError) as exc:
                        failed.append({"ticker": ticker, "reason": str(exc)})
                result_payload: dict[str, Any] = {"action": "add", "added": added, "failed": failed}

            elif action == "remove" and tickers:
                removed: list[str] = []
                failed = []
                for ticker in tickers:
                    try:
                        await watchlist_service.remove_ticker(
                            session, telegram_id=telegram_id, symbol=ticker
                        )
                        removed.append(ticker.upper())
                    except TickerNotOnWatchlistError as exc:
                        failed.append({"ticker": ticker, "reason": str(exc)})
                result_payload = {"action": "remove", "removed": removed, "failed": failed}

            else:
                current = await watchlist_service.get_symbols(session, telegram_id=telegram_id)
                result_payload = {"action": "view", "watchlist": current}

        state["tool_result"] = json.dumps(result_payload, default=str)
    except WatchlistError as exc:
        logger.exception("Watchlist operation failed for telegram_id=%s", telegram_id)
        state["error"] = f"watchlist_op_failed: {exc}"
        state["tool_result"] = None

    return state


async def price_alert_node(state: AgentState) -> AgentState:
    """
    Parse and persist a price alert request (e.g. "alert me when AAPL
    goes above $320"). The ticker is validated/normalized via
    FinancialDataService before being stored, the same validation
    WatchlistService performs before persisting a watchlist add.
    """
    services = state["services"]
    financial_data_service = services["financial_data_service"]
    telegram_id = state["user_id"]
    entities = state.get("entities", {})
    tickers: list[str] = entities.get("tickers") or []

    parsed = _parse_price_alert_request(state["text"], tickers)
    if parsed is None:
        state["tool_result"] = None
        state["error"] = (
            "price_alert_parse_failed: could not determine the ticker, "
            "condition (above/below), and target price from the message"
        )
        return state

    raw_symbol, condition, target_price = parsed

    try:
        resolved_symbol = await financial_data_service.resolve_symbol(raw_symbol)
        # Confirm the symbol has live data behind it -- same guard
        # WatchlistService applies before persisting a watchlist add.
        await financial_data_service.get_quote(resolved_symbol)
    except SymbolNotFoundError:
        state["tool_result"] = None
        state["error"] = f"price_alert_invalid_symbol: {raw_symbol!r} is not a recognized ticker"
        return state
    except FinancialDataError as exc:
        state["tool_result"] = None
        state["error"] = f"price_alert_symbol_validation_failed: {exc}"
        return state

    alert_service = AlertService()
    try:
        async with AsyncSessionLocal() as session:
            entry = await alert_service.create_alert(
                session,
                telegram_id=telegram_id,
                symbol=resolved_symbol,
                condition=condition,
                target_price=target_price,
            )
        state["tool_result"] = json.dumps(
            {
                "action": "price_alert_created",
                "symbol": entry.symbol,
                "condition": entry.condition.value,
                "target_price": entry.target_price,
            },
            default=str,
        )
    except AlertError as exc:
        logger.exception("Failed to create price alert for telegram_id=%s", telegram_id)
        state["error"] = f"price_alert_creation_failed: {exc}"
        state["tool_result"] = None

    return state


# ---------------------------------------------------------------------------
# Node: plain conversation
# ---------------------------------------------------------------------------


async def conversation_node(state: AgentState) -> AgentState:
    """
    Handle general conversational turns (greetings, thanks, meta questions
    about Atlas) with no external data lookup required.
    """
    state["tool_result"] = None
    return state


# ---------------------------------------------------------------------------
# Node: compose final response
# ---------------------------------------------------------------------------


_RESPONSE_SYSTEM_PROMPT = """You are Atlas, a junior financial analyst \
assistant speaking to a client over Telegram. Write concise, professional, \
plain-language responses (no more than ~120 words unless the data \
genuinely requires more). Use the provided tool data as ground truth and \
do not fabricate numbers. If tool data is missing or an error occurred, \
say so briefly and offer a next step. Avoid financial advice disclaimers \
unless directly relevant; focus on being useful and precise. Markdown \
formatting (bold, bullet points) is supported."""


async def compose_response_node(state: AgentState) -> AgentState:
    """
    Turn the classified intent + any tool_result into a final,
    user-facing, concise natural-language response.
    """
    if state.get("response"):
        # Clarification node already produced the final response.
        return state

    history_snippet = _format_history_for_prompt(state.get("history", []))
    tool_result = state.get("tool_result")
    error = state.get("error")

    prompt_parts = [
        f"Conversation so far:\n{history_snippet}\n",
        f"User's latest message: {state['text']!r}",
        f"Detected intent: {state.get('intent', Intent.CONVERSATION).value}",
    ]
    if tool_result:
        prompt_parts.append(f"Tool data (JSON or text):\n{tool_result}")
    if error:
        prompt_parts.append(
            f"Note: a backend error occurred while gathering data: {error}. "
            "Acknowledge briefly and suggest retrying, without technical detail."
        )

    prompt = "\n\n".join(prompt_parts)

    try:
        reply = await _call_gemini(prompt, system_instruction=_RESPONSE_SYSTEM_PROMPT)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("Failed to compose final response")
        if _is_dev_environment():
            # Plain text — NO Markdown. Telegram rejects messages whose
            # ParseMode=Markdown clashes with raw traceback characters
            # (* _ ` etc.), causing a silent send failure (no reply at all).
            reply = (
                f"{_DEV_ERROR_PREFIX}\n"
                f"[DEV] compose_response_node raised {type(exc).__name__}\n"
                f"{exc}\n\n"
                f"--- traceback (last 1500 chars) ---\n{tb[-1500:]}"
            )
        else:
            reply = ""

    if not reply:
        if _is_dev_environment():
            # Gemini returned empty text with no exception — surface clearly.
            reply = (
                f"{_DEV_ERROR_PREFIX}\n"
                "[DEV] Gemini returned an empty response from compose_response_node.\n"
                f"Model: {GEMINI_MODEL_NAME} | Check API key, quota, and prompt."
            )
        else:
            reply = (
                "I ran into an issue putting that together. Could you try "
                "rephrasing or asking again in a moment?"
            )

    # Skip the character cap for dev error payloads so the full traceback
    # is always visible during debugging.
    if not reply.startswith(_DEV_ERROR_PREFIX) and len(reply) > MAX_RESPONSE_CHARS:
        reply = reply[: MAX_RESPONSE_CHARS - 3].rstrip() + "..."

    state["response"] = reply
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _route_after_context(state: AgentState) -> str:
    """Conditional edge: send unonboarded users to onboarding_node instead
    of classify_intent. This is what makes onboarding trigger on a brand
    new user's first plain-text message -- no /start required."""
    user_profile = state.get("user_profile")
    if user_profile is not None and not getattr(user_profile, "onboarding_completed", True):
        return "onboarding"
    return "classify_intent"


def _route_after_onboarding(state: AgentState) -> str:
    """Conditional edge: if onboarding_node already produced a response
    (asked a question / confirmed skip / confirmed completion), stop there.
    If it left state["response"] empty, the message was a real request
    that overrode onboarding -- fall through to classify_intent for it."""
    return "end" if state.get("response") else "classify_intent"


def _route_after_classification(state: AgentState) -> str:
    """Conditional edge: dispatch based on classified intent."""
    if state.get("clarification_needed"):
        return "clarify"

    intent = state.get("intent", Intent.CONVERSATION)
    return {
        Intent.COMPANY_RESEARCH: "company_research",
        Intent.DOCUMENT_QA: "document_qa",
        Intent.WATCHLIST: "watchlist",
        Intent.PRICE_ALERT: "price_alert",
        Intent.CONVERSATION: "conversation",
        Intent.AMBIGUOUS: "clarify",
    }[intent]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct (but do not compile) the Atlas LangGraph workflow.

    Graph shape:

        load_context -> classify_intent -> [route] -> {
            company_research -> compose_response -> END
            document_qa      -> compose_response -> END
            watchlist        -> compose_response -> END
            conversation      -> compose_response -> END
            clarify           -> END
        }
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("company_research", company_research_node)
    graph.add_node("document_qa", document_qa_node)
    graph.add_node("watchlist", watchlist_node)
    graph.add_node("price_alert", price_alert_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("compose_response", compose_response_node)

    graph.set_entry_point("load_context")

    graph.add_conditional_edges(
        "load_context",
        _route_after_context,
        {"onboarding": "onboarding", "classify_intent": "classify_intent"},
    )
    graph.add_conditional_edges(
        "onboarding",
        _route_after_onboarding,
        {"end": END, "classify_intent": "classify_intent"},
    )

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classification,
        {
            "clarify": "clarify",
            "company_research": "company_research",
            "document_qa": "document_qa",
            "watchlist": "watchlist",
            "price_alert": "price_alert",
            "conversation": "conversation",
        },
    )

    graph.add_edge("company_research", "compose_response")
    graph.add_edge("document_qa", "compose_response")
    graph.add_edge("watchlist", "compose_response")
    graph.add_edge("price_alert", "compose_response")
    graph.add_edge("conversation", "compose_response")

    graph.add_edge("clarify", END)
    graph.add_edge("compose_response", END)

    return graph


# Compiled once at import time and reused across requests.
_compiled_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# Public entrypoint (matches telegram_bot.py's expected interface)
# ---------------------------------------------------------------------------


async def run_agent(
    *,
    user_id: int,
    text: str,
    memory_service: MemoryService,
    financial_data_service: FinancialDataService,
    document_service: DocumentService,
    watchlist_service: Optional[Any] = None,
) -> str:
    """
    Run the Atlas LangGraph workflow for a single user turn and return the
    final natural-language response to send back via Telegram.

    Args:
        user_id: Telegram user id of the sender.
        text: User's message text (already transcribed if originally
            a voice note).
        memory_service: Shared MemoryService instance.
        financial_data_service: Shared FinancialDataService instance.
        document_service: Shared DocumentService instance.
        watchlist_service: Optional WatchlistService instance.

    Returns:
        A concise, user-facing reply string.
    """
    effective_watchlist_service = watchlist_service

    initial_state: AgentState = {
        "user_id": user_id,
        "text": text,
        "services": {
            "memory_service": memory_service,
            "financial_data_service": financial_data_service,
            "document_service": document_service,
            "watchlist_service": effective_watchlist_service,
        },
        "history": [],
        "user_profile": None,
        "entities": {},
        "clarification_needed": False,
        "tool_result": None,
        "response": "",
        "error": None,
    }

    try:
        final_state: AgentState = cast(AgentState, await _compiled_graph.ainvoke(initial_state))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("LangGraph execution failed for user_id=%s", user_id)
        if _is_dev_environment():
            return (
                f"{_DEV_ERROR_PREFIX}\n"
                f"[DEV] LangGraph ainvoke raised {type(exc).__name__}\n"
                f"{exc}\n\n"
                f"--- traceback (last 1500 chars) ---\n{tb[-1500:]}"
            )
        return (
            "I hit an unexpected error processing that request. Please "
            "try again shortly."
        )

    response = final_state.get("response")
    if response:
        return response

    # No response was produced — surface the internal error state if available.
    internal_error = final_state.get("error")
    logger.error(
        "run_agent produced no response for user_id=%s; internal error state: %s",
        user_id,
        internal_error,
    )
    if _is_dev_environment() and internal_error:
        return (
            f"{_DEV_ERROR_PREFIX}\n"
            f"[DEV] Agent returned no response.\n"
            f"Internal error state: {internal_error}"
        )
    return "I couldn't quite process that — could you rephrase your request?"