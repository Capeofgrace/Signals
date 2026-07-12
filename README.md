# Signals — Crypto Coin Signal Scanner

Scans crypto markets and ranks coins using the six technical signals most
widely relied on by experienced traders. Works against any exchange
supported by [ccxt](https://github.com/ccxt/ccxt) (Binance, Coinbase, Kraken,
Bybit, etc.) using only public market data — no API keys required.

## The 6 signals

| # | Signal | What it measures | Bullish trigger | Bearish trigger |
|---|--------|-------------------|------------------|------------------|
| 1 | **RSI(14)** | Momentum / overbought-oversold | RSI < 30 (oversold) | RSI > 70 (overbought) |
| 2 | **MACD(12,26,9)** | Trend momentum | MACD line crosses above signal line | MACD line crosses below signal line |
| 3 | **EMA 50/200 Cross** | Long-term trend direction | Golden cross / EMA50 above EMA200 | Death cross / EMA50 below EMA200 |
| 4 | **Bollinger Bands(20,2)** | Volatility breakout | Close breaks above upper band | Close breaks below lower band |
| 5 | **Volume Spike** | Conviction behind a move | ≥2x average volume on an up candle | ≥2x average volume on a down candle |
| 6 | **Double Top/Bottom** | Reversal chart pattern | Double bottom confirmed (close breaks above the neckline) | Double top confirmed (close breaks below the neckline) |

Each signal contributes `+1` (bullish), `-1` (bearish), or `0` (neutral) to a
**composite score** in the range `-6..+6`.

**Buy / Strong Buy** is driven by the full composite score across all six
signals (`Strong Buy` at score ≥3, `Buy` at 1-2).

**Sell / Strong Sell is intentionally narrower.** It's driven only by RSI and
the Double Top/Bottom pattern, regardless of what the other four signals say:
both bearish → `Strong Sell`, either one alone → `Sell`, neither → falls
through to the Buy/Neutral check above. MACD, EMA cross, Bollinger, and
Volume never trigger a sell call on their own — they still display and still
count toward the composite score/ranking, they just don't gate the sell
verdict. One consequence: a coin can show a positive composite score (green
score bar) alongside a `Sell` badge if RSI + Double Top both flip bearish
while the other four stay bullish — that's the sell rule intentionally
overriding the broader score for that specific call.

**On the double top/bottom pattern:** two pivot highs (or lows) of similar
height within a lookback window, with a meaningfully deeper trough (or higher
peak) between them, form the pattern; it only scores once price closes past
the neckline (the trough for a top, the peak for a bottom) — an unconfirmed
pattern shows up as a neutral "forming" note so you know to watch it, but
doesn't move the composite score.

These are widely-used, well-documented signals — not a guarantee of
profitability. Always combine with your own risk management; this tool
surfaces signals, it does not give financial advice.

## Install

```bash
pip install -r requirements.txt
```

For running tests:

```bash
pip install -r requirements-dev.txt
```

## Usage

### CLI

Scan the top 20 USDT pairs by volume on Binance (default):

```bash
python -m signals.cli
```

Scan specific symbols on another exchange/timeframe:

```bash
python -m signals.cli --exchange kraken --symbols BTC/USDT,ETH/USDT --timeframe 1d
```

Options:

```
--exchange     ccxt exchange id (default: binance)
--quote        quote currency to auto-discover markets in (default: USDT)
--symbols      comma-separated symbols, overrides auto-discovery
--top-n        number of top-volume markets to scan (default: 20)
--timeframe    candle timeframe, e.g. 1h, 4h, 1d (default: 4h)
--limit        candles fetched per symbol (default: 300)
--output       table | json | csv (default: table)
--min-score    only show results with |composite_score| >= this value
```

If installed via `pip install .`, the same tool is available as `signals-scan`.

### Web dashboard

```bash
uvicorn signals.web.app:app --reload
```

Then open http://localhost:8000 — pick an exchange, quote currency,
timeframe, and market count, hit **Scan**, and get a ranked, color-coded
table with the detail behind every signal. The dashboard talks to a JSON
API at `GET /api/scan` (same query parameters as the CLI) that you can also
call directly or wire into your own tools.

## Project layout

```
signals/
  indicators.py   pure indicator math (RSI, MACD, EMA, Bollinger, volume ratio)
  data.py         ccxt wrapper: fetch OHLCV, discover top symbols by volume
  scanner.py      combines indicators into per-symbol signals + composite score
  cli.py          argparse CLI (table/json/csv output)
  web/app.py      FastAPI JSON API
  web/static/     dashboard frontend (vanilla HTML/JS)
tests/            unit tests using synthetic OHLCV data (no network required)
```

## Testing

```bash
pytest
```

Tests run entirely offline against synthetic OHLCV data — no exchange
connectivity is required to validate the signal logic.

## Notes

- Only spot markets are scanned by default; futures/margin symbols are
  filtered out.
- Common stablecoin bases (USDC, BUSD, DAI, etc.) are excluded from
  auto-discovery to avoid scanning stablecoin-to-stablecoin pairs.
- A symbol is skipped (not errored) if the exchange fails to return data or
  there isn't enough history (200+ candles recommended, for the EMA 200
  signal to be meaningful).
