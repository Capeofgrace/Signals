import numpy as np
import pandas as pd

from signals.scanner import composite_verdict, evaluate_signals, scan_dataframe


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


def test_evaluate_signals_returns_five_results():
    df = _base_df()
    signals = evaluate_signals(df)
    assert len(signals) == 5
    names = {s.name for s in signals}
    assert names == {
        "RSI(14)",
        "MACD(12,26,9)",
        "EMA 50/200 Cross",
        "Bollinger Bands(20,2)",
        "Volume Spike",
    }


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


def test_composite_verdict_thresholds():
    assert composite_verdict(5) == "Strong Buy"
    assert composite_verdict(1) == "Buy"
    assert composite_verdict(0) == "Neutral"
    assert composite_verdict(-2) == "Sell"
    assert composite_verdict(-5) == "Strong Sell"


def test_scan_dataframe_aggregates_score():
    df = _base_df(n=260, start=100.0, step=0.3, noise=0.0)
    result = scan_dataframe("BTC/USDT", df)
    assert result.symbol == "BTC/USDT"
    assert result.price == df["close"].iloc[-1]
    assert result.composite_score == sum(s.score for s in result.signals)
    assert result.verdict == composite_verdict(result.composite_score)
