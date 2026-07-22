"""FastAPI dashboard for the crypto signal scanner."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..scanner import scan_market, scan_symbols

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Crypto Signal Scanner")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/scan")
def api_scan(
    exchange: str = Query("binance"),
    quote: str = Query("USDT"),
    top_n: int = Query(20, ge=1, le=100),
    timeframe: str = Query("4h"),
    limit: int = Query(300, ge=60, le=1000),
    symbols: str | None = Query(None, description="Comma-separated list to override auto-discovery"),
):
    try:
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
            results = scan_symbols(exchange, symbol_list, timeframe=timeframe, limit=limit)
        else:
            results = scan_market(
                exchange_id=exchange,
                quote=quote,
                top_n=top_n,
                timeframe=timeframe,
                limit=limit,
            )
    except Exception as exc:  # surface exchange/network errors to the client
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return [dataclasses.asdict(r) for r in results]
