import numpy as np
import pandas as pd
import pytest

from signals import indicators as ind


def test_ema_matches_pandas_ewm():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    result = ind.ema(series, 3)
    expected = series.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_sma_basic():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = ind.sma(series, 2)
    assert result.iloc[-1] == 4.5
    assert pd.isna(result.iloc[0])


def test_rsi_all_gains_is_100():
    series = pd.Series(range(1, 30), dtype=float)  # strictly increasing
    result = ind.rsi(series, 14)
    assert result.iloc[-1] == 100.0


def test_rsi_all_losses_is_0():
    series = pd.Series(range(30, 1, -1), dtype=float)  # strictly decreasing
    result = ind.rsi(series, 14)
    assert result.iloc[-1] == 0.0


def test_rsi_flat_series_is_neutral():
    series = pd.Series([50.0] * 30)
    result = ind.rsi(series, 14)
    # no gains or losses -> RS is NaN (0/0); should not raise
    assert result.iloc[-1] != result.iloc[-1] or 0 <= result.iloc[-1] <= 100


def test_macd_returns_three_series_of_equal_length():
    series = pd.Series(np.sin(np.linspace(0, 10, 100)) + 10)
    macd_line, signal_line, hist = ind.macd(series)
    assert len(macd_line) == len(signal_line) == len(hist) == len(series)
    pd.testing.assert_series_equal(hist, macd_line - signal_line)


def test_bollinger_bands_ordering():
    series = pd.Series(np.random.default_rng(0).normal(100, 5, 60))
    upper, mid, lower = ind.bollinger_bands(series, 20, 2.0)
    valid = ~upper.isna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_volume_spike_ratio():
    volume = pd.Series([10] * 20 + [50])
    ratio = ind.volume_spike_ratio(volume, 20)
    # rolling average of the trailing 20 bars includes the spike bar itself:
    # (19 * 10 + 50) / 20 = 12, so ratio = 50 / 12.
    expected_avg = (19 * 10 + 50) / 20
    assert ratio.iloc[-1] == pytest.approx(50 / expected_avg)


def test_pivot_highs_finds_single_peak():
    series = pd.Series([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])
    mask = ind.pivot_highs(series, window=3)
    assert list(mask[mask].index) == [3]


def test_pivot_lows_finds_single_trough():
    series = pd.Series([5.0, 4.0, 3.0, 1.0, 3.0, 4.0, 5.0])
    mask = ind.pivot_lows(series, window=3)
    assert list(mask[mask].index) == [3]


def test_pivot_highs_ignores_flat_runs():
    # A flat plateau has no directional peak; ties must not count as pivots.
    series = pd.Series([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    mask = ind.pivot_highs(series, window=3)
    assert not mask.any()


def test_atr_is_zero_for_a_perfectly_flat_series():
    n = 30
    close = pd.Series([100.0] * n)
    high = close.copy()
    low = close.copy()
    result = ind.atr(high, low, close, period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_atr_reflects_true_range_including_gaps():
    # A gap-up (low stays above the prior close) must still count via the
    # high-minus-prev-close leg of true range, not just high-minus-low.
    close = pd.Series([100.0] * 15 + [120.0])
    high = close.copy()
    low = close.copy()
    result = ind.atr(high, low, close, period=14)
    assert result.iloc[-1] > 0


def test_atr_scales_with_volatility():
    rng = np.random.default_rng(5)
    calm = pd.Series(100 + rng.normal(0, 0.2, 60))
    wild = pd.Series(100 + rng.normal(0, 5.0, 60))
    atr_calm = ind.atr(calm + 0.3, calm - 0.3, calm, period=14).iloc[-1]
    atr_wild = ind.atr(wild + 0.3, wild - 0.3, wild, period=14).iloc[-1]
    assert atr_wild > atr_calm
