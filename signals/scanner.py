"""Evaluates the five signal strategy against OHLCV data and ranks symbols.

The five signals implemented here are the ones most commonly cited by
experienced technical traders:

1. RSI(14)            - momentum / overbought-oversold
2. MACD(12,26,9)       - trend momentum crossovers
3. EMA 50/200 cross    - long-term trend direction (golden/death cross)
4. Bollinger Bands(20,2) - volatility breakout
5. Volume spike        - conviction behind a move
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data as data_mod
from . import indicators as ind

MIN_CANDLES = 50


@dataclass
class SignalResult:
    name: str
    verdict: str  # "bullish" | "bearish" | "neutral"
    score: int  # +1, -1, or 0
    detail: str


@dataclass
class ScanResult:
    symbol: str
    price: float
    composite_score: int
    verdict: str
    signals: list[SignalResult] = field(default_factory=list)


def _rsi_signal(close: pd.Series) -> SignalResult:
    series = ind.rsi(close, 14)
    latest = series.iloc[-1]
    if pd.isna(latest):
        return SignalResult("RSI(14)", "neutral", 0, "Not enough data")
    if latest < 30:
        return SignalResult("RSI(14)", "bullish", 1, f"Oversold at {latest:.1f}")
    if latest > 70:
        return SignalResult("RSI(14)", "bearish", -1, f"Overbought at {latest:.1f}")
    return SignalResult("RSI(14)", "neutral", 0, f"Neutral at {latest:.1f}")


def _macd_signal(close: pd.Series) -> SignalResult:
    macd_line, signal_line, _ = ind.macd(close)
    diff = macd_line - signal_line
    if diff.dropna().shape[0] < 2:
        return SignalResult("MACD(12,26,9)", "neutral", 0, "Not enough data")
    prev, curr = diff.iloc[-2], diff.iloc[-1]
    if prev <= 0 < curr:
        return SignalResult("MACD(12,26,9)", "bullish", 1, "Bullish crossover")
    if prev >= 0 > curr:
        return SignalResult("MACD(12,26,9)", "bearish", -1, "Bearish crossover")
    side = "above" if curr > 0 else "below"
    return SignalResult("MACD(12,26,9)", "neutral", 0, f"No fresh cross (MACD {side} signal line)")


def _ema_cross_signal(close: pd.Series) -> SignalResult:
    ema50 = ind.ema(close, 50)
    ema200 = ind.ema(close, 200)
    diff = ema50 - ema200
    if diff.dropna().shape[0] < 2:
        return SignalResult("EMA 50/200 Cross", "neutral", 0, "Not enough data")
    prev, curr = diff.iloc[-2], diff.iloc[-1]
    if prev <= 0 < curr:
        return SignalResult("EMA 50/200 Cross", "bullish", 1, "Golden cross just formed")
    if prev >= 0 > curr:
        return SignalResult("EMA 50/200 Cross", "bearish", -1, "Death cross just formed")
    if curr > 0:
        return SignalResult("EMA 50/200 Cross", "bullish", 1, "Uptrend (EMA50 > EMA200)")
    return SignalResult("EMA 50/200 Cross", "bearish", -1, "Downtrend (EMA50 < EMA200)")


def _bollinger_signal(close: pd.Series) -> SignalResult:
    upper, _, lower = ind.bollinger_bands(close, 20, 2.0)
    if upper.dropna().shape[0] < 2:
        return SignalResult("Bollinger Bands(20,2)", "neutral", 0, "Not enough data")
    prev_close, curr_close = close.iloc[-2], close.iloc[-1]
    prev_upper, curr_upper = upper.iloc[-2], upper.iloc[-1]
    prev_lower, curr_lower = lower.iloc[-2], lower.iloc[-1]
    if prev_close <= prev_upper and curr_close > curr_upper:
        return SignalResult("Bollinger Bands(20,2)", "bullish", 1, "Breakout above upper band")
    if prev_close >= prev_lower and curr_close < curr_lower:
        return SignalResult("Bollinger Bands(20,2)", "bearish", -1, "Breakdown below lower band")
    return SignalResult("Bollinger Bands(20,2)", "neutral", 0, "Trading inside the bands")


def _volume_spike_signal(close: pd.Series, open_: pd.Series, volume: pd.Series) -> SignalResult:
    ratio_series = ind.volume_spike_ratio(volume, 20)
    ratio = ratio_series.iloc[-1]
    if pd.isna(ratio):
        return SignalResult("Volume Spike", "neutral", 0, "Not enough data")
    if ratio >= 2.0 and close.iloc[-1] > open_.iloc[-1]:
        return SignalResult("Volume Spike", "bullish", 1, f"{ratio:.1f}x avg volume on an up candle")
    if ratio >= 2.0 and close.iloc[-1] < open_.iloc[-1]:
        return SignalResult("Volume Spike", "bearish", -1, f"{ratio:.1f}x avg volume on a down candle")
    return SignalResult("Volume Spike", "neutral", 0, f"{ratio:.2f}x average volume")


def evaluate_signals(df: pd.DataFrame) -> list[SignalResult]:
    """Run all five signals against an OHLCV DataFrame with
    open/high/low/close/volume columns."""
    close, open_, volume = df["close"], df["open"], df["volume"]
    return [
        _rsi_signal(close),
        _macd_signal(close),
        _ema_cross_signal(close),
        _bollinger_signal(close),
        _volume_spike_signal(close, open_, volume),
    ]


def composite_verdict(score: int) -> str:
    if score >= 3:
        return "Strong Buy"
    if score >= 1:
        return "Buy"
    if score <= -3:
        return "Strong Sell"
    if score <= -1:
        return "Sell"
    return "Neutral"


def scan_dataframe(symbol: str, df: pd.DataFrame) -> ScanResult:
    signals = evaluate_signals(df)
    score = sum(s.score for s in signals)
    return ScanResult(
        symbol=symbol,
        price=float(df["close"].iloc[-1]),
        composite_score=score,
        verdict=composite_verdict(score),
        signals=signals,
    )


def scan_symbols(
    exchange_id: str,
    symbols: list[str],
    timeframe: str = "4h",
    limit: int = 300,
) -> list[ScanResult]:
    """Fetch data for each symbol and evaluate signals, skipping any symbol
    that fails to fetch or lacks enough history."""
    exchange = data_mod.get_exchange(exchange_id)
    results = []
    for symbol in symbols:
        try:
            df = data_mod.fetch_ohlcv(exchange, symbol, timeframe, limit)
        except Exception:
            continue
        if len(df) < MIN_CANDLES:
            continue
        results.append(scan_dataframe(symbol, df))
    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def scan_market(
    exchange_id: str = "binance",
    quote: str = "USDT",
    top_n: int = 20,
    timeframe: str = "4h",
    limit: int = 300,
) -> list[ScanResult]:
    """Discover the top-N symbols by volume on an exchange and scan them."""
    exchange = data_mod.get_exchange(exchange_id)
    symbols = data_mod.top_symbols_by_volume(exchange, quote=quote, top_n=top_n)
    return scan_symbols(exchange_id, symbols, timeframe=timeframe, limit=limit)
