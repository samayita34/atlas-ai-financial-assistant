"""
app/langgraph/graph.py

LangGraph workflow definition for Atlas AI Financial Assistant.

This module builds and exposes the core conversational graph that powers
Atlas. It classifies each incoming user message, routes it to the
appropriate capability (company research, document Q&A, watchlist/memory
lookups, or plain conversation), asks a clarifying question when the
request is ambiguous, and produces a concise final response.

The public entrypoint, `run_agent`, matches the interface already assumed
by `app/bot/telegram_bot.py`:

    async def run_agent(
        *,
        user_id: int,
        text: str,
        memory_service: MemoryService,
        financial_data_service: FinancialDataService,
        document_service: DocumentService,
    ) -> str: ...

NOTE ON ASSUMPTIONS:
The exact method signatures of MemoryService, FinancialDataService, and
DocumentService are not fully specified in the provided context. Where an
interface is missing or ambiguous, the smallest reasonable assumption is
made and marked with a `# TODO:` comment.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph

from app.database.database import AsyncSessionLocal
from app.database.models import ConversationMessage
from app.services.document_service import DocumentService
from app.services.financial_data_service import FinancialDataService
from app.services.memory_service import MemoryService

# TODO: No dedicated "LLM client" service was present in "Already
# Implemented". Assuming a thin Gemini wrapper is reasonable to construct
# directly here via google-genai, configured from app.config.settings.
# Adjust this import if a shared `app/services/llm_service.py` (or similar)
# already exists in the real codebase.
from app.config import get_settings

settings = get_settings()

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


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """Coarse-grained classification of what the user is asking for."""

    COMPANY_RESEARCH = "company_research"
    DOCUMENT_QA = "document_qa"
    WATCHLIST = "watchlist"
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
    if _genai_client is None:  # pragma: no cover - defensive fallback
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
    user_id = state["user_id"]

    try:
        async with AsyncSessionLocal() as session:
            memory_service = MemoryService(session)
            # get_recent_messages returns list[ConversationMessage]; convert
            # to the {"role": str, "content": str} dicts the rest of the
            # graph expects.
            raw_messages = await memory_service.get_recent_messages(
                user_id=user_id, limit=CONVERSATION_HISTORY_LIMIT
            )
            history: list[dict[str, str]] = [
                {"role": msg.role.value, "content": msg.content}
                for msg in raw_messages
            ]
            user_profile = await memory_service.get_user_by_telegram_id(user_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load context for user_id=%s", user_id)
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

    if has_entities or has_financial_cue:
        if intent == Intent.AMBIGUOUS or confidence < 0.45:
            if has_entities:
                intent = Intent.COMPANY_RESEARCH
                confidence = max(confidence, 0.9)

    return intent, confidence, updated_entities


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
    Fetch live financial data for the requested company/ticker via
    FinancialDataService and summarize it concisely.
    """
    services = state["services"]
    financial_data_service = services["financial_data_service"]
    entities = state.get("entities", {})

    tickers: list[str] = entities.get("tickers") or []
    company_names: list[str] = entities.get("company_names") or []
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
    services = state["services"]
    document_service = services["document_service"]
    user_id = state["user_id"]

    try:
        async with AsyncSessionLocal() as session:
            # DocumentService requires a session at construction time.
            doc_service = DocumentService(session)
            context = await doc_service.get_context_for_query(
                user_id=user_id,
                query=state["text"],
            )
        state["tool_result"] = context if context else "No relevant documents found."
    except Exception as exc:  # noqa: BLE001
        logger.exception("DocumentService Q&A failed for user_id=%s", user_id)
        state["error"] = f"document_qa_failed: {exc}"
        state["tool_result"] = None

    return state


# ---------------------------------------------------------------------------
# Node: watchlist
# ---------------------------------------------------------------------------


async def watchlist_node(state: AgentState) -> AgentState:
    """
    Handle watchlist add/remove/view requests.
    Temporarily stubbed while WatchlistService dependency is detached.
    """
    state["tool_result"] = json.dumps({"watchlist": []}, default=str)
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
    except Exception:  # noqa: BLE001
        logger.exception("Failed to compose final response")
        reply = ""

    if not reply:
        reply = (
            "I ran into an issue putting that together. Could you try "
            "rephrasing or asking again in a moment?"
        )

    if len(reply) > MAX_RESPONSE_CHARS:
        reply = reply[: MAX_RESPONSE_CHARS - 3].rstrip() + "..."

    state["response"] = reply
    return state


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _route_after_classification(state: AgentState) -> str:
    """Conditional edge: dispatch based on classified intent."""
    if state.get("clarification_needed"):
        return "clarify"

    intent = state.get("intent", Intent.CONVERSATION)
    return {
        Intent.COMPANY_RESEARCH: "company_research",
        Intent.DOCUMENT_QA: "document_qa",
        Intent.WATCHLIST: "watchlist",
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
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("company_research", company_research_node)
    graph.add_node("document_qa", document_qa_node)
    graph.add_node("watchlist", watchlist_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("compose_response", compose_response_node)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classification,
        {
            "clarify": "clarify",
            "company_research": "company_research",
            "document_qa": "document_qa",
            "watchlist": "watchlist",
            "conversation": "conversation",
        },
    )

    graph.add_edge("company_research", "compose_response")
    graph.add_edge("document_qa", "compose_response")
    graph.add_edge("watchlist", "compose_response")
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
    except Exception:  # noqa: BLE001
        logger.exception("LangGraph execution failed for user_id=%s", user_id)
        return (
            "I hit an unexpected error processing that request. Please "
            "try again shortly."
        )

    return final_state.get("response") or (
        "I couldn't quite process that — could you rephrase your request?"
    )