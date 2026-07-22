"""Command-line entry point for the crypto signal scanner."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys

from .scanner import ScanResult, scan_market, scan_symbols

VERDICT_STYLE = {
    "Strong Buy": "bold green",
    "Buy": "green",
    "Neutral": "white",
    "Sell": "red",
    "Strong Sell": "bold red",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signals-scan",
        description="Scan crypto markets for the top trader signals: "
        "RSI, MACD, EMA 50/200 cross, Bollinger breakout, volume spike, "
        "and double top/bottom reversal patterns.",
    )
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    parser.add_argument("--quote", default="USDT", help="Quote currency to scan (default: USDT)")
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols to scan instead of auto-discovering "
        "top markets, e.g. BTC/USDT,ETH/USDT",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top-volume markets to scan (default: 20)")
    parser.add_argument("--timeframe", default="4h", help="Candle timeframe, e.g. 1h, 4h, 1d (default: 4h)")
    parser.add_argument("--limit", type=int, default=300, help="Number of candles to fetch per symbol (default: 300)")
    parser.add_argument(
        "--output",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--min-score", type=int, default=None, help="Only show results with |composite_score| >= this value")
    return parser


def _result_to_dict(result: ScanResult) -> dict:
    d = dataclasses.asdict(result)
    return d


def print_table(results: list[ScanResult]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_plain(results)
        return

    console = Console()
    table = Table(title="Crypto Signal Scan")
    table.add_column("Symbol", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Verdict")
    table.add_column("Exit (stop)", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Signals")

    for r in results:
        signal_text = "  ".join(f"{s.name}: {s.detail}" for s in r.signals)
        style = VERDICT_STYLE.get(r.verdict, "white")
        table.add_row(
            r.symbol,
            f"{r.price:,.4f}",
            f"{r.composite_score:+d}",
            f"[{style}]{r.verdict}[/{style}]",
            f"{r.exit_price:,.4f}" if r.exit_price is not None else "—",
            f"{r.target_price:,.4f}" if r.target_price is not None else "—",
            signal_text,
        )
    console.print(table)
    console.print(
        "[dim]Exit/Target are ATR(14)-based (1.5x/3x, ~2:1 reward:risk) reference levels, "
        "not guarantees. Not financial advice.[/dim]"
    )


def print_plain(results: list[ScanResult]) -> None:
    header = f"{'Symbol':<14}{'Price':>14}{'Score':>8}  {'Verdict':<12}{'Exit':>14}{'Target':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        exit_str = f"{r.exit_price:,.4f}" if r.exit_price is not None else "—"
        target_str = f"{r.target_price:,.4f}" if r.target_price is not None else "—"
        print(
            f"{r.symbol:<14}{r.price:>14,.4f}{r.composite_score:>+8d}  "
            f"{r.verdict:<12}{exit_str:>14}{target_str:>14}"
        )
        for s in r.signals:
            print(f"    - {s.name}: {s.detail}")
    print("\nExit/Target are ATR(14)-based reference levels (~2:1 reward:risk), not guarantees.")


def print_json(results: list[ScanResult]) -> None:
    print(json.dumps([_result_to_dict(r) for r in results], indent=2, default=str))


def print_csv(results: list[ScanResult]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "symbol",
            "price",
            "composite_score",
            "verdict",
            "exit_price",
            "target_price",
            "signal",
            "signal_verdict",
            "signal_detail",
        ]
    )
    for r in results:
        for s in r.signals:
            writer.writerow(
                [r.symbol, r.price, r.composite_score, r.verdict, r.exit_price, r.target_price, s.name, s.verdict, s.detail]
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
            results = scan_symbols(args.exchange, symbols, timeframe=args.timeframe, limit=args.limit)
        else:
            results = scan_market(
                exchange_id=args.exchange,
                quote=args.quote,
                top_n=args.top_n,
                timeframe=args.timeframe,
                limit=args.limit,
            )
    except Exception as exc:
        print(f"Error: could not reach {args.exchange} ({exc}).", file=sys.stderr)
        print("Check your network connection and the --exchange/--quote/--symbols values.", file=sys.stderr)
        return 1

    if args.min_score is not None:
        results = [r for r in results if abs(r.composite_score) >= args.min_score]

    if not results:
        print("No results (check exchange/symbol connectivity or lower --min-score).", file=sys.stderr)
        return 1

    if args.output == "json":
        print_json(results)
    elif args.output == "csv":
        print_csv(results)
    else:
        print_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
