"""Evaluates the signal strategy against OHLCV data and ranks symbols.

The six signals implemented here are commonly cited by experienced
technical traders:

1. RSI(14)               - momentum / overbought-oversold
2. MACD(12,26,9)         - trend momentum crossovers
3. EMA 50/200 cross      - long-term trend direction (golden/death cross)
4. Bollinger Bands(20,2) - volatility breakout
5. Volume spike          - conviction behind a move
6. Double Top/Bottom     - reversal chart pattern with neckline confirmation
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


def _find_double_top(
    df: pd.DataFrame,
    window: int = 3,
    lookback: int = 120,
    tolerance: float = 0.025,
    min_depth: float = 0.02,
    min_gap: int = 5,
    max_gap: int = 60,
) -> dict | None:
    """Locate the most recent double-top candidate: two similar-height pivot
    highs with a meaningfully deeper trough between them. Returns None if no
    such pair exists in the lookback window."""
    sub = df.iloc[-lookback:] if len(df) > lookback else df
    high = sub["high"].reset_index(drop=True)
    low = sub["low"].reset_index(drop=True)
    close = sub["close"].reset_index(drop=True)

    mask = ind.pivot_highs(high, window)
    positions = mask[mask].index.tolist()
    if len(positions) < 2:
        return None
    i1, i2 = positions[-2], positions[-1]
    gap = i2 - i1
    if gap < min_gap or gap > max_gap:
        return None

    peak1, peak2 = high.iloc[i1], high.iloc[i2]
    avg_peak = (peak1 + peak2) / 2
    if avg_peak <= 0 or abs(peak1 - peak2) / avg_peak > tolerance:
        return None

    neckline = low.iloc[i1 : i2 + 1].min()
    if (avg_peak - neckline) / avg_peak < min_depth:
        return None

    after = close.iloc[i2 + 1 :]
    broken = after[after < neckline]
    confirmed = len(broken) > 0
    fresh = confirmed and after.iloc[-1] < neckline and (len(after) < 2 or after.iloc[-2] >= neckline)
    return {
        "kind": "double_top",
        "pivot2_pos": i2,
        "confirmed": confirmed,
        "fresh": fresh,
        "level": peak2,
        "neckline": neckline,
    }


def _find_double_bottom(
    df: pd.DataFrame,
    window: int = 3,
    lookback: int = 120,
    tolerance: float = 0.025,
    min_depth: float = 0.02,
    min_gap: int = 5,
    max_gap: int = 60,
) -> dict | None:
    """Mirror image of _find_double_top: two similar-depth pivot lows with a
    meaningfully higher peak between them."""
    sub = df.iloc[-lookback:] if len(df) > lookback else df
    high = sub["high"].reset_index(drop=True)
    low = sub["low"].reset_index(drop=True)
    close = sub["close"].reset_index(drop=True)

    mask = ind.pivot_lows(low, window)
    positions = mask[mask].index.tolist()
    if len(positions) < 2:
        return None
    i1, i2 = positions[-2], positions[-1]
    gap = i2 - i1
    if gap < min_gap or gap > max_gap:
        return None

    trough1, trough2 = low.iloc[i1], low.iloc[i2]
    avg_trough = (trough1 + trough2) / 2
    if avg_trough <= 0 or abs(trough1 - trough2) / avg_trough > tolerance:
        return None

    neckline = high.iloc[i1 : i2 + 1].max()
    if (neckline - avg_trough) / avg_trough < min_depth:
        return None

    after = close.iloc[i2 + 1 :]
    broken = after[after > neckline]
    confirmed = len(broken) > 0
    fresh = confirmed and after.iloc[-1] > neckline and (len(after) < 2 or after.iloc[-2] <= neckline)
    return {
        "kind": "double_bottom",
        "pivot2_pos": i2,
        "confirmed": confirmed,
        "fresh": fresh,
        "level": trough2,
        "neckline": neckline,
    }


def _double_pattern_signal(df: pd.DataFrame) -> SignalResult:
    name = "Double Top/Bottom"
    candidates = [c for c in (_find_double_top(df), _find_double_bottom(df)) if c is not None]
    if not candidates:
        return SignalResult(name, "neutral", 0, "No pattern detected")

    chosen = max(candidates, key=lambda c: c["pivot2_pos"])
    timing = "just broke" if chosen["fresh"] else "broke"

    if chosen["kind"] == "double_top":
        if chosen["confirmed"]:
            return SignalResult(
                name, "bearish", -1, f"Double top confirmed, neckline {timing} at {chosen['neckline']:.4g}"
            )
        return SignalResult(
            name, "neutral", 0, f"Double top forming near {chosen['level']:.4g} (neckline {chosen['neckline']:.4g})"
        )

    if chosen["confirmed"]:
        return SignalResult(
            name, "bullish", 1, f"Double bottom confirmed, neckline {timing} at {chosen['neckline']:.4g}"
        )
    return SignalResult(
        name, "neutral", 0, f"Double bottom forming near {chosen['level']:.4g} (neckline {chosen['neckline']:.4g})"
    )


def evaluate_signals(df: pd.DataFrame) -> list[SignalResult]:
    """Run all six signals against an OHLCV DataFrame with
    open/high/low/close/volume columns."""
    close, open_, volume = df["close"], df["open"], df["volume"]
    return [
        _rsi_signal(close),
        _macd_signal(close),
        _ema_cross_signal(close),
        _bollinger_signal(close),
        _volume_spike_signal(close, open_, volume),
        _double_pattern_signal(df),
    ]


RSI_BUY_MAX = 30
RSI_SELL_MIN = 60


def composite_verdict(
    score: int,
    signals: list[SignalResult] | None = None,
    rsi_value: float | None = None,
) -> str:
    """Buy and Sell are both hard-gated on the raw RSI(14) value, not just
    the RSI signal's own verdict:

    - RSI > 60 is required for Sell/Strong Sell. Double Top/Bottom confirmed
      bearish on top of that upgrades it to Strong Sell; otherwise it's Sell.
    - RSI < 30 is required for Buy/Strong Buy. Among those, the full
      composite score (all six signals) decides Strong Buy (score >= 3)
      vs. Buy.
    - RSI between 30 and 60 is Neutral, regardless of the other signals.

    Pass `rsi_value` to enable this gate. Without it (e.g. not enough
    history to compute RSI), this falls back to score-only thresholds.
    """
    if rsi_value is not None:
        pattern = None
        if signals is not None:
            pattern = next((s for s in signals if s.name == "Double Top/Bottom"), None)
        pattern_bearish = pattern is not None and pattern.verdict == "bearish"

        if rsi_value > RSI_SELL_MIN:
            return "Strong Sell" if pattern_bearish else "Sell"
        if rsi_value < RSI_BUY_MAX:
            return "Strong Buy" if score >= 3 else "Buy"
        return "Neutral"

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
    rsi_series = ind.rsi(df["close"], 14)
    rsi_value = None if pd.isna(rsi_series.iloc[-1]) else float(rsi_series.iloc[-1])
    return ScanResult(
        symbol=symbol,
        price=float(df["close"].iloc[-1]),
        composite_score=score,
        verdict=composite_verdict(score, signals, rsi_value),
        signals=signals,
    )


def scan_symbols(
    exchange_id: str,
    symbols: list[str],
    timeframe: str = "4h",
    limit: int = 300,
) -> list[ScanResult]:
    """Fetch data for each symbol and evaluate signals, skipping any symbol
    that fails to fetch or lacks enough history. Neutral verdicts are
    dropped -- only actionable Buy/Strong Buy/Sell/Strong Sell calls are
    returned."""
    exchange = data_mod.get_exchange(exchange_id)
    results = []
    for symbol in symbols:
        try:
            df = data_mod.fetch_ohlcv(exchange, symbol, timeframe, limit)
        except Exception:
            continue
        if len(df) < MIN_CANDLES:
            continue
        result = scan_dataframe(symbol, df)
        if result.verdict == "Neutral":
            continue
        results.append(result)
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
