"""Free / freemium MCP-equivalent market stack for US + Asia-Pacific."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote as urlquote

import requests

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_UA = "Mozilla/5.0 (compatible; PredictorMCP/1.0; +https://github.com/eewnah1)"

DEFAULT_UNIVERSES: dict[str, dict[str, str]] = {
    "general": {
        "SPY": "SPY",
        "SPX": "^GSPC",
        "NDX": "^IXIC",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "GLD": "GLD",
        "US10Y": "^TNX",
    },
    "kospi": {"KS11": "^KS11", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX", "KOSPI": "^KS11"},
    "hst": {"HST": "HST.SI", "HSI": "^HSI", "HST_IDX": "^HSI", "SPX": "^GSPC", "VIX": "^VIX"},
    "srt": {"SRT": "SRT.SI", "ES3": "ES3.SI", "SPX": "^GSPC", "VIX": "^VIX"},
    "clr": {"CLR": "CLR.SI", "ES3": "ES3.SI", "SPX": "^GSPC", "VIX": "^VIX"},
    "sgx_g3b": {"G3B": "G3B.SI", "ES3": "ES3.SI", "SPX": "^GSPC", "VIX": "^VIX"},
    "sgx_cfa": {"CFA": "CFA.SI", "ES3": "ES3.SI", "SPX": "^GSPC", "VIX": "^VIX"},
    "slv": {"SLV": "SLV", "GLD": "GLD", "GC": "GC=F", "SPX": "^GSPC", "VIX": "^VIX"},
    "blackrock_gold": {"GLD": "GLD", "IAU": "IAU", "GC": "GC=F", "SPX": "^GSPC", "VIX": "^VIX"},
    "lion_korea": {"EWY": "EWY", "KS11": "^KS11", "SPX": "^GSPC", "VIX": "^VIX"},
    "mg_japan": {"EWJ": "EWJ", "N225": "^N225", "SPX": "^GSPC", "VIX": "^VIX"},
    "neuberger": {"IXP": "IXP", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX"},
    "first_sentier_bridge": {"AUDUSD": "AUDUSD=X", "EEM": "EEM", "SPX": "^GSPC", "VIX": "^VIX"},
    "schroder_multi_asset": {"EEM": "EEM", "ACWI": "ACWI", "SPX": "^GSPC", "VIX": "^VIX"},
    "aussuper": {"XJO": "^AXJO", "SPX": "^GSPC", "VIX": "^VIX", "AUDUSD": "AUDUSD=X"},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote(ticker: str, timeout: float = 5.0) -> dict[str, Any]:
    url = YAHOO_CHART.format(ticker=urlquote(ticker, safe=""))
    try:
        r = requests.get(
            url,
            params={"range": "5d", "interval": "1d", "events": "div,splits"},
            headers={"User-Agent": YAHOO_UA, "Accept": "application/json"},
            timeout=timeout,
        )
        if r.status_code != 200:
            return {"ticker": ticker, "http_status": r.status_code, "error": "http_failed"}
        payload = r.json()
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result:
            return {"ticker": ticker, "error": "empty_chart"}
        meta = result.get("meta") or {}
        ohlc = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = [c for c in (ohlc.get("close") or []) if c is not None]
        return {
            "ticker": ticker,
            "name": meta.get("symbol"),
            "last": closes[-1] if closes else meta.get("regularMarketPrice"),
            "prev": closes[-2] if len(closes) >= 2 else None,
            "source": "yahoo_chart_v8",
            "http_status": 200,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)[:120]}


def fetch_free_mcp_stack(universe: str = "general") -> dict[str, Any]:
    """Return a snapshot of free Yahoo Finance proxies for the chosen universe."""
    tickers = DEFAULT_UNIVERSES.get(universe, DEFAULT_UNIVERSES["general"])
    quotes = {k: _quote(v) for k, v in tickers.items()}
    ok = sum(1 for q in quotes.values() if q.get("http_status") == 200)
    return {
        "name": "free_mcp_stack",
        "universe": universe,
        "as_of": _now_iso(),
        "source": "yahoo_chart_v8",
        "quotes": quotes,
        "status": "ok" if ok >= 2 else "degraded",
        "note": "This is a free proxy. Set API keys for full TradingEconomics/Finnhub/Alpha Vantage feeds.",
    }


def overlay_prediction(signal: dict[str, Any], stack: dict[str, Any]) -> dict[str, Any]:
    """Optional overlay: attach the free MCP stack snapshot to a prediction response."""
    if not signal or not isinstance(signal, dict):
        return signal
    signal = dict(signal)
    signal["mcp_stack_snapshot"] = {
        "name": stack.get("name"),
        "as_of": stack.get("as_of"),
        "status": stack.get("status"),
        "universe": stack.get("universe"),
    }
    return signal
