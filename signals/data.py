"""Exchange data access via ccxt.

Kept isolated from the signal logic so tests can exercise the scanner with
synthetic data instead of hitting a live exchange.
"""
from __future__ import annotations

import pandas as pd

STABLECOIN_BASES = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "PYUSD", "EUR"}


def get_exchange(exchange_id: str = "binance"):
    """Instantiate a rate-limited ccxt exchange client."""
    import ccxt

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise ValueError(f"Unknown exchange id: {exchange_id!r}") from exc
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv(exchange, symbol: str, timeframe: str = "4h", limit: int = 300) -> pd.DataFrame:
    """Fetch OHLCV candles for a symbol and return a tidy DataFrame."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def top_symbols_by_volume(
    exchange,
    quote: str = "USDT",
    top_n: int = 20,
    exclude_stablecoins: bool = True,
) -> list[str]:
    """Return the top-N spot symbols quoted in `quote`, ranked by 24h quote volume."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    candidates = []
    for symbol, market in markets.items():
        if not market.get("active", True):
            continue
        if market.get("quote") != quote:
            continue
        if market.get("type") not in (None, "spot"):
            continue
        base = market.get("base", "")
        if exclude_stablecoins and base in STABLECOIN_BASES:
            continue
        ticker = tickers.get(symbol)
        volume = ticker.get("quoteVolume") if ticker else None
        if volume is None:
            continue
        candidates.append((symbol, volume))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [symbol for symbol, _ in candidates[:top_n]]
