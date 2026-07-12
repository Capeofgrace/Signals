"""Technical indicator math shared by the signal scanner.

All functions are pure and operate on pandas Series/DataFrames so they can be
unit tested without hitting any exchange.
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band (SMA), and lower band."""
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def volume_spike_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume divided by its rolling average (>1 means above average)."""
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


def pivot_highs(series: pd.Series, window: int = 3) -> pd.Series:
    """Boolean mask: True where a bar is strictly higher than every bar in the
    `window` bars before it and the `window` bars after it. Requires strict
    inequality on both sides so a flat run of equal values never counts as a
    pivot (ties don't identify a swing high)."""
    left_max = series.rolling(window).max().shift(1)
    right_max = series[::-1].rolling(window).max()[::-1].shift(-1)
    return (series > left_max) & (series > right_max)


def pivot_lows(series: pd.Series, window: int = 3) -> pd.Series:
    """Boolean mask: True where a bar is strictly lower than every bar in the
    `window` bars before it and the `window` bars after it."""
    left_min = series.rolling(window).min().shift(1)
    right_min = series[::-1].rolling(window).min()[::-1].shift(-1)
    return (series < left_min) & (series < right_min)
