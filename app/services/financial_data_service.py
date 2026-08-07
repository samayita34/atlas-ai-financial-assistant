"""
app/services/financial_data_service.py

Live company research service backed by the Finnhub API.

Provides:
- Company profile lookup
- Real-time quote lookup
- Basic financial metrics (key stats)
- Recent company news
- A combined "overview" aggregator used by the LangGraph company-research
  node (`app.langgraph.graph.company_research_node`)

All network I/O is async (httpx.AsyncClient). Errors are normalized into
`FinancialDataError` subclasses so callers (LangGraph nodes, bot handlers)
can handle failures uniformly without needing to know Finnhub-specific
error shapes.

NOTE ON ASSUMPTIONS:
- `app.config.settings` is assumed to expose `finnhub_api_key: str` and
  optionally `finnhub_base_url: str`. TODOs mark these assumptions.
- Ticker resolution from a free-text company name (e.g. "Apple" -> "AAPL")
  is assumed to go through Finnhub's `/search` endpoint since no dedicated
  symbol-resolution service exists in "Already implemented".
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Final, Optional

import httpx

from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TODO: Confirm attribute names on `settings`. Assuming `finnhub_api_key`
# is defined in app/config.py per the project's Finnhub integration.
FINNHUB_API_KEY: Final[str] = getattr(settings, "finnhub_api_key", "")
FINNHUB_BASE_URL: Final[str] = getattr(
    settings, "finnhub_base_url", "https://finnhub.io/api/v1"
)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_NEWS_LOOKBACK_DAYS: Final[int] = 7
DEFAULT_NEWS_LIMIT: Final[int] = 5
MAX_RETRIES: Final[int] = 2
RETRY_BACKOFF_SECONDS: Final[float] = 0.75

# Simple in-memory TTL cache to avoid hammering Finnhub for repeated
# lookups within a short window (e.g. multiple users asking about the
# same ticker). Not shared across processes.
# TODO: Replace with a Redis-backed cache if Atlas scales to multiple
# worker processes/instances.
_CACHE_TTL_SECONDS: Final[float] = 30.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FinancialDataError(Exception):
    """Base exception for all financial data service failures."""


class SymbolNotFoundError(FinancialDataError):
    """Raised when a ticker/company name cannot be resolved to a symbol."""


class FinnhubRequestError(FinancialDataError):
    """Raised when a Finnhub API call fails after retries."""


class FinnhubConfigError(FinancialDataError):
    """Raised when the service is used without a configured API key."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompanyProfile:
    """Normalized company profile data from Finnhub's `/stock/profile2`."""

    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    ipo_date: Optional[str] = None
    website: Optional[str] = None
    logo: Optional[str] = None
    phone: Optional[str] = None
    shares_outstanding: Optional[float] = None

    @classmethod
    def from_raw(cls, symbol: str, raw: dict[str, Any]) -> "CompanyProfile":
        return cls(
            symbol=symbol,
            name=raw.get("name"),
            exchange=raw.get("exchange"),
            industry=raw.get("finnhubIndustry"),
            country=raw.get("country"),
            currency=raw.get("currency"),
            market_cap=raw.get("marketCapitalization"),
            ipo_date=raw.get("ipo"),
            website=raw.get("weburl"),
            logo=raw.get("logo"),
            phone=raw.get("phone"),
            shares_outstanding=raw.get("shareOutstanding"),
        )


@dataclass(slots=True)
class Quote:
    """Normalized real-time quote data from Finnhub's `/quote`."""

    symbol: str
    current_price: Optional[float] = None
    change: Optional[float] = None
    percent_change: Optional[float] = None
    high_price_today: Optional[float] = None
    low_price_today: Optional[float] = None
    open_price_today: Optional[float] = None
    previous_close: Optional[float] = None
    timestamp: Optional[int] = None

    @classmethod
    def from_raw(cls, symbol: str, raw: dict[str, Any]) -> "Quote":
        return cls(
            symbol=symbol,
            current_price=raw.get("c"),
            change=raw.get("d"),
            percent_change=raw.get("dp"),
            high_price_today=raw.get("h"),
            low_price_today=raw.get("l"),
            open_price_today=raw.get("o"),
            previous_close=raw.get("pc"),
            timestamp=raw.get("t"),
        )


@dataclass(slots=True)
class FinancialMetrics:
    """
    Basic financial metrics from Finnhub's `/stock/metric` endpoint
    (metric=all). Only a curated subset of commonly useful fields is
    surfaced; the full raw payload is retained for advanced callers.
    """

    symbol: str
    pe_ratio_ttm: Optional[float] = None
    eps_ttm: Optional[float] = None
    revenue_growth_ttm_yoy: Optional[float] = None
    gross_margin_ttm: Optional[float] = None
    net_margin_ttm: Optional[float] = None
    roe_ttm: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield_ttm: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    beta: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_raw(cls, symbol: str, raw: dict[str, Any]) -> "FinancialMetrics":
        metric = raw.get("metric", {}) or {}
        return cls(
            symbol=symbol,
            pe_ratio_ttm=metric.get("peTTM"),
            eps_ttm=metric.get("epsTTM"),
            revenue_growth_ttm_yoy=metric.get("revenueGrowthTTMYoy"),
            gross_margin_ttm=metric.get("grossMarginTTM"),
            net_margin_ttm=metric.get("netProfitMarginTTM"),
            roe_ttm=metric.get("roeTTM"),
            debt_to_equity=metric.get("totalDebt/totalEquityAnnual"),
            dividend_yield_ttm=metric.get("dividendYieldIndicatedAnnual"),
            fifty_two_week_high=metric.get("52WeekHigh"),
            fifty_two_week_low=metric.get("52WeekLow"),
            beta=metric.get("beta"),
            raw=metric,
        )


@dataclass(slots=True)
class NewsItem:
    """A single company news item from Finnhub's `/company-news`."""

    headline: str
    summary: Optional[str]
    source: Optional[str]
    url: Optional[str]
    datetime_unix: Optional[int]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "NewsItem":
        return cls(
            headline=raw.get("headline", ""),
            summary=raw.get("summary"),
            source=raw.get("source"),
            url=raw.get("url"),
            datetime_unix=raw.get("datetime"),
        )


@dataclass(slots=True)
class CompanyOverview:
    """
    Aggregated research bundle combining profile, quote, metrics, and
    recent news — the shape consumed by the LangGraph company-research
    node.
    """

    symbol: str
    profile: Optional[CompanyProfile]
    quote: Optional[Quote]
    metrics: Optional[FinancialMetrics]
    news: list[NewsItem]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, suitable for JSON/LLM prompting."""
        return {
            "symbol": self.symbol,
            "profile": self.profile.__dict__ if self.profile else None,
            "quote": self.quote.__dict__ if self.quote else None,
            "metrics": (
                {k: v for k, v in self.metrics.__dict__.items() if k != "raw"}
                if self.metrics
                else None
            ),
            "news": [
                {
                    "headline": item.headline,
                    "summary": item.summary,
                    "source": item.source,
                    "url": item.url,
                    "datetime": item.datetime_unix,
                }
                for item in self.news
            ],
        }


# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------


class _TTLCache:
    """Minimal in-memory TTL cache keyed by arbitrary hashable keys."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: Any) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                self._store.pop(key, None)
                return None
            return value

    async def set(self, key: Any, value: Any) -> None:
        async with self._lock:
            self._store[key] = (time.monotonic() + self._ttl, value)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class FinancialDataService:
    """
    Async service for live company research via the Finnhub API.

    Designed to be instantiated once (e.g. in `app/main.py`'s lifespan)
    and shared across requests; it owns a single `httpx.AsyncClient` for
    connection pooling and must be closed via `aclose()` on shutdown.
    """

    def __init__(
        self,
        *,
        api_key: str = FINNHUB_API_KEY,
        base_url: str = FINNHUB_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Args:
            api_key: Finnhub API key. TODO: sourced from
                `settings.finnhub_api_key`; raises `FinnhubConfigError` at
                call time if empty.
            base_url: Finnhub REST API base URL.
            timeout_seconds: Per-request timeout.
            client: Optional pre-configured `httpx.AsyncClient` (useful
                for testing/dependency injection). If omitted, one is
                created lazily on first use.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = client
        self._owns_client = client is None

        self._symbol_cache = _TTLCache(ttl_seconds=300.0)  # symbols change rarely
        self._profile_cache = _TTLCache(ttl_seconds=300.0)
        self._quote_cache = _TTLCache(ttl_seconds=_CACHE_TTL_SECONDS)
        self._metrics_cache = _TTLCache(ttl_seconds=120.0)
        self._news_cache = _TTLCache(ttl_seconds=120.0)

    # -- lifecycle -----------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """
        Close the underlying HTTP client. Should be called during
        application shutdown (e.g. from `app/main.py`'s lifespan).

        TODO: `app/main.py` does not currently call this explicitly;
        wire `await financial_data_service.aclose()` into its shutdown
        sequence if/when this service holds a persistent client there.
        """
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- low-level request helper --------------------------------------

    async def _request(self, path: str, params: dict[str, Any]) -> Any:
        """
        Perform a GET request against the Finnhub API with retries and
        normalized error handling.

        Raises:
            FinnhubConfigError: if no API key is configured.
            FinnhubRequestError: on network failure, timeout, non-2xx
                response, or invalid JSON after exhausting retries.
        """
        if not self._api_key:
            raise FinnhubConfigError(
                "FINNHUB_API_KEY is not configured. Set it in app.config.settings."
            )

        client = await self._get_client()
        query = {**params, "token": self._api_key}

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 2):  # initial attempt + retries
            try:
                response = await client.get(path, params=query)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Don't retry on client errors other than rate limiting.
                if status == 429 and attempt <= MAX_RETRIES:
                    logger.warning(
                        "Finnhub rate limited on %s (attempt %d); backing off.",
                        path,
                        attempt,
                    )
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    last_exc = exc
                    continue
                logger.error("Finnhub HTTP error on %s: %s", path, exc)
                raise FinnhubRequestError(
                    f"Finnhub request to {path} failed with status {status}."
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "Finnhub network error on %s (attempt %d): %s; retrying.",
                        path,
                        attempt,
                        exc,
                    )
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                logger.error("Finnhub request to %s failed after retries: %s", path, exc)
                raise FinnhubRequestError(
                    f"Finnhub request to {path} failed after {MAX_RETRIES} retries."
                ) from exc
            except ValueError as exc:  # JSON decode error
                logger.error("Finnhub returned invalid JSON for %s: %s", path, exc)
                raise FinnhubRequestError(
                    f"Finnhub returned invalid JSON for {path}."
                ) from exc

        # Should be unreachable, but keeps type checkers happy.
        raise FinnhubRequestError(
            f"Finnhub request to {path} failed unexpectedly."
        ) from last_exc

    # -- symbol resolution ------------------------------------------------

    async def resolve_symbol(self, query: str) -> str:
        """
        Resolve a free-text query (ticker or company name) to a Finnhub
        ticker symbol using the `/search` endpoint.

        If `query` already looks like a bare ticker (short, uppercase,
        alphanumeric), it is returned as-is without a network call.

        Raises:
            SymbolNotFoundError: if no matching symbol is found.
        """
        normalized = query.strip()
        if not normalized:
            raise SymbolNotFoundError("Empty company/ticker query.")

        # Fast path: looks like an already-valid ticker (e.g. "AAPL", "MSFT").
        if normalized.isupper() and normalized.replace(".", "").isalnum() and len(normalized) <= 6:
            return normalized

        cache_key = normalized.lower()
        cached = await self._symbol_cache.get(cache_key)
        if cached is not None:
            return cached

        data = await self._request("/search", {"q": normalized})
        results: list[dict[str, Any]] = data.get("result", []) if isinstance(data, dict) else []

        # Prefer common stock listings over ETFs/other instrument types.
        best_match: Optional[dict[str, Any]] = None
        for item in results:
            if item.get("type") == "Common Stock":
                best_match = item
                break
        if best_match is None and results:
            best_match = results[0]

        if best_match is None or not best_match.get("symbol"):
            raise SymbolNotFoundError(f"No ticker found for {query!r}.")

        symbol = str(best_match["symbol"])
        await self._symbol_cache.set(cache_key, symbol)
        return symbol

    # -- individual lookups ------------------------------------------------

    async def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Fetch company profile data for a resolved ticker symbol."""
        resolved = await self.resolve_symbol(symbol)

        cached = await self._profile_cache.get(resolved)
        if cached is not None:
            return cached

        raw = await self._request("/stock/profile2", {"symbol": resolved})
        if not raw:
            raise SymbolNotFoundError(f"No company profile found for {resolved!r}.")

        profile = CompanyProfile.from_raw(resolved, raw)
        await self._profile_cache.set(resolved, profile)
        return profile

    async def get_quote(self, symbol: str) -> Quote:
        """Fetch a real-time quote for a resolved ticker symbol."""
        resolved = await self.resolve_symbol(symbol)

        cached = await self._quote_cache.get(resolved)
        if cached is not None:
            return cached

        raw = await self._request("/quote", {"symbol": resolved})
        if not raw or raw.get("c") in (None, 0):
            # Finnhub returns all-zero payloads for unknown symbols rather
            # than a 404, so treat that as "not found".
            raise SymbolNotFoundError(f"No quote data found for {resolved!r}.")

        quote = Quote.from_raw(resolved, raw)
        await self._quote_cache.set(resolved, quote)
        return quote

    async def get_financial_metrics(self, symbol: str) -> FinancialMetrics:
        """Fetch basic financial metrics for a resolved ticker symbol."""
        resolved = await self.resolve_symbol(symbol)

        cached = await self._metrics_cache.get(resolved)
        if cached is not None:
            return cached

        raw = await self._request(
            "/stock/metric", {"symbol": resolved, "metric": "all"}
        )
        metrics = FinancialMetrics.from_raw(resolved, raw or {})
        await self._metrics_cache.set(resolved, metrics)
        return metrics

    async def get_recent_news(
        self,
        symbol: str,
        *,
        lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
        limit: int = DEFAULT_NEWS_LIMIT,
    ) -> list[NewsItem]:
        """
        Fetch recent company news for a resolved ticker symbol within the
        last `lookback_days`, capped at `limit` items (most recent first).
        """
        resolved = await self.resolve_symbol(symbol)
        cache_key = (resolved, lookback_days, limit)

        cached = await self._news_cache.get(cache_key)
        if cached is not None:
            return cached

        from datetime import date, timedelta  # local import: narrow usage

        today = date.today()
        from_date = today - timedelta(days=lookback_days)

        raw = await self._request(
            "/company-news",
            {
                "symbol": resolved,
                "from": from_date.isoformat(),
                "to": today.isoformat(),
            },
        )
        items_raw = raw if isinstance(raw, list) else []
        items_raw.sort(key=lambda item: item.get("datetime", 0), reverse=True)

        news = [NewsItem.from_raw(item) for item in items_raw[:limit]]
        await self._news_cache.set(cache_key, news)
        return news

    # -- aggregate lookup (used by LangGraph company research node) ----

    async def get_company_overview(self, query: str) -> dict[str, Any]:
        """
        Fetch a combined research bundle (profile + quote + metrics +
        recent news) for a company/ticker query, concurrently.

        This is the primary entrypoint assumed by
        `app.langgraph.graph.company_research_node`.

        Args:
            query: Ticker symbol or free-text company name.

        Returns:
            A plain dict (via `CompanyOverview.to_dict()`) suitable for
            JSON serialization and LLM prompting.

        Raises:
            SymbolNotFoundError: if the query cannot be resolved to a
                known ticker at all.
        """
        resolved = await self.resolve_symbol(query)

        results = await asyncio.gather(
            self.get_company_profile(resolved),
            self.get_quote(resolved),
            self.get_financial_metrics(resolved),
            self.get_recent_news(resolved),
            return_exceptions=True,
        )

        profile_result, quote_result, metrics_result, news_result = results

        profile: Optional[CompanyProfile] = (
            profile_result if isinstance(profile_result, CompanyProfile) else None
        )
        quote: Optional[Quote] = quote_result if isinstance(quote_result, Quote) else None
        metrics: Optional[FinancialMetrics] = (
            metrics_result if isinstance(metrics_result, FinancialMetrics) else None
        )
        news: list[NewsItem] = news_result if isinstance(news_result, list) else []

        for label, result in (
            ("profile", profile_result),
            ("quote", quote_result),
            ("metrics", metrics_result),
            ("news", news_result),
        ):
            if isinstance(result, Exception):
                logger.warning(
                    "Partial failure fetching %s for %r: %s", label, resolved, result
                )

        if profile is None and quote is None and metrics is None and not news:
            # Every sub-lookup failed; surface a clear error rather than
            # an empty-looking overview.
            raise FinancialDataError(
                f"Unable to retrieve any financial data for {query!r}."
            )

        overview = CompanyOverview(
            symbol=resolved,
            profile=profile,
            quote=quote,
            metrics=metrics,
            news=news,
        )
        return overview.to_dict()