import numpy as np
import pandas as pd

from signals import indicators as ind
from signals.scanner import SignalResult, composite_verdict, evaluate_signals, scan_dataframe


def _signals(**overrides):
    """Build a full 6-signal list with everything neutral by default,
    overridden by name -> (verdict, score) for the ones under test."""
    base = {
        "RSI(14)": ("neutral", 0),
        "MACD(12,26,9)": ("neutral", 0),
        "EMA 50/200 Cross": ("neutral", 0),
        "Bollinger Bands(20,2)": ("neutral", 0),
        "Volume Spike": ("neutral", 0),
        "Double Top/Bottom": ("neutral", 0),
    }
    base.update(overrides)
    return [SignalResult(name, verdict, score, "detail") for name, (verdict, score) in base.items()]


def _base_df(n=250, start=100.0, step=0.0, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    prices = start + np.arange(n) * step + rng.normal(0, noise, n)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1000.0),
        }
    )
    return df


def _double_top_df(break_below=True, fresh=False):
    up1 = np.linspace(90, 110, 10)
    down1 = np.linspace(110, 95, 8)[1:]
    up2 = np.linspace(95, 110, 8)[1:]
    if fresh:
        down2 = np.linspace(110, 96, 10)[1:]
        tail = np.array([90.0])
    elif break_below:
        down2 = np.linspace(110, 80, 14)[1:]
        tail = np.full(15, down2[-1])
    else:
        down2 = np.linspace(110, 100, 6)[1:]
        tail = np.full(15, down2[-1])
    prices = np.concatenate([up1, down1, up2, down2, tail])
    n = len(prices)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 1000.0),
        }
    )


def _double_bottom_df(break_above=True):
    down1 = np.linspace(110, 90, 10)
    up1 = np.linspace(90, 105, 8)[1:]
    down2 = np.linspace(105, 90, 8)[1:]
    if break_above:
        up2 = np.linspace(90, 120, 14)[1:]
    else:
        up2 = np.linspace(90, 100, 6)[1:]
    tail = np.full(15, up2[-1])
    prices = np.concatenate([down1, up1, down2, up2, tail])
    n = len(prices)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.full(n, 1000.0),
        }
    )


def test_evaluate_signals_returns_six_results():
    df = _base_df()
    signals = evaluate_signals(df)
    assert len(signals) == 6
    names = {s.name for s in signals}
    assert names == {
        "RSI(14)",
        "MACD(12,26,9)",
        "EMA 50/200 Cross",
        "Bollinger Bands(20,2)",
        "Volume Spike",
        "Double Top/Bottom",
    }


def test_double_top_confirmed_is_bearish():
    df = _double_top_df(break_below=True)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "bearish"
    assert pattern.score == -1
    assert "confirmed" in pattern.detail.lower()


def test_double_top_forming_is_neutral_but_flagged():
    df = _double_top_df(break_below=False)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "neutral"
    assert pattern.score == 0
    assert "forming" in pattern.detail.lower()


def test_double_top_fresh_break_is_flagged_as_fresh():
    df = _double_top_df(fresh=True)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "bearish"
    assert "just broke" in pattern.detail.lower()


def test_double_bottom_confirmed_is_bullish():
    df = _double_bottom_df(break_above=True)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "bullish"
    assert pattern.score == 1
    assert "confirmed" in pattern.detail.lower()


def test_double_bottom_forming_is_neutral_but_flagged():
    df = _double_bottom_df(break_above=False)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "neutral"
    assert pattern.score == 0
    assert "forming" in pattern.detail.lower()


def test_no_double_pattern_on_a_clean_trend():
    df = _base_df(n=60, start=100.0, step=0.8, noise=0.0)
    signals = {s.name: s for s in evaluate_signals(df)}
    pattern = signals["Double Top/Bottom"]
    assert pattern.verdict == "neutral"
    assert pattern.score == 0
    assert pattern.detail == "No pattern detected"


def test_rsi_oversold_signal_is_bullish():
    # Strong sustained downtrend drives RSI well below 30.
    df = _base_df(n=60, start=200.0, step=-1.5, noise=0.0)
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["RSI(14)"].verdict == "bullish"
    assert signals["RSI(14)"].score == 1


def test_rsi_overbought_signal_is_bearish():
    df = _base_df(n=60, start=50.0, step=1.5, noise=0.0)
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["RSI(14)"].verdict == "bearish"
    assert signals["RSI(14)"].score == -1


def test_ema_cross_uptrend_is_bullish():
    # Long steady uptrend -> EMA50 above EMA200.
    df = _base_df(n=260, start=100.0, step=0.3, noise=0.0)
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["EMA 50/200 Cross"].verdict == "bullish"


def test_ema_cross_downtrend_is_bearish():
    df = _base_df(n=260, start=300.0, step=-0.3, noise=0.0)
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["EMA 50/200 Cross"].verdict == "bearish"


def test_volume_spike_bullish_on_up_candle():
    df = _base_df(n=30, start=100.0, step=0.0, noise=0.0)
    df.loc[df.index[-1], "volume"] = 5000.0
    df.loc[df.index[-1], "open"] = 100.0
    df.loc[df.index[-1], "close"] = 105.0
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["Volume Spike"].verdict == "bullish"


def test_volume_spike_bearish_on_down_candle():
    df = _base_df(n=30, start=100.0, step=0.0, noise=0.0)
    df.loc[df.index[-1], "volume"] = 5000.0
    df.loc[df.index[-1], "open"] = 105.0
    df.loc[df.index[-1], "close"] = 100.0
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["Volume Spike"].verdict == "bearish"


def test_bollinger_breakout_bullish():
    df = _base_df(n=40, start=100.0, step=0.0, noise=0.3, seed=2)
    # Force a clean breakout above the recent range on the final candle.
    df.loc[df.index[-1], "close"] = df["close"].iloc[:-1].max() + 20
    signals = {s.name: s for s in evaluate_signals(df)}
    assert signals["Bollinger Bands(20,2)"].verdict == "bullish"


def test_buy_requires_rsi_below_30():
    signals = _signals()
    assert composite_verdict(1, signals, rsi_value=25.0) == "Buy"
    assert composite_verdict(4, signals, rsi_value=25.0) == "Strong Buy"


def test_buy_not_triggered_when_rsi_at_or_above_30():
    signals = _signals()
    # Even a strongly positive composite score doesn't matter once RSI
    # is out of the oversold zone.
    assert composite_verdict(5, signals, rsi_value=30.0) == "Neutral"
    assert composite_verdict(5, signals, rsi_value=45.0) == "Neutral"


def test_sell_requires_rsi_above_60():
    signals = _signals()
    assert composite_verdict(0, signals, rsi_value=65.0) == "Sell"


def test_sell_upgrades_to_strong_sell_with_confirmed_double_top():
    signals = _signals(**{"Double Top/Bottom": ("bearish", -1)})
    assert composite_verdict(0, signals, rsi_value=65.0) == "Strong Sell"


def test_sell_not_triggered_when_rsi_at_or_below_60():
    # Double top confirmed bearish is no longer sufficient on its own --
    # RSI > 60 is a hard requirement for Sell.
    signals = _signals(**{"Double Top/Bottom": ("bearish", -1)})
    assert composite_verdict(-1, signals, rsi_value=60.0) == "Neutral"
    assert composite_verdict(-1, signals, rsi_value=45.0) == "Neutral"


def test_composite_verdict_falls_back_to_score_thresholds_without_rsi():
    # rsi_value omitted (e.g. not enough history) -> legacy score-only path.
    assert composite_verdict(5) == "Strong Buy"
    assert composite_verdict(1) == "Buy"
    assert composite_verdict(0) == "Neutral"
    assert composite_verdict(-2) == "Sell"
    assert composite_verdict(-5) == "Strong Sell"


def test_scan_dataframe_aggregates_score():
    df = _base_df(n=260, start=100.0, step=0.3, noise=0.0)
    result = scan_dataframe("BTC/USDT", df)
    rsi_value = ind.rsi(df["close"], 14).iloc[-1]
    assert result.symbol == "BTC/USDT"
    assert result.price == df["close"].iloc[-1]
    assert result.composite_score == sum(s.score for s in result.signals)
    assert result.verdict == composite_verdict(result.composite_score, result.signals, float(rsi_value))
