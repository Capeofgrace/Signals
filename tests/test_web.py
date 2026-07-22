from fastapi.testclient import TestClient

from signals.scanner import ScanResult, SignalResult
from signals.web import app as app_module


def test_serves_dashboard_html():
    client = TestClient(app_module.app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Crypto Signal Scanner" in res.text


def test_api_scan_with_explicit_symbols(monkeypatch):
    fake_result = ScanResult(
        symbol="BTC/USDT",
        price=65000.0,
        composite_score=2,
        verdict="Buy",
        signals=[SignalResult("RSI(14)", "bullish", 1, "Oversold at 25.0")],
    )

    def fake_scan_symbols(exchange_id, symbols, timeframe="4h", limit=300):
        assert symbols == ["BTC/USDT"]
        return [fake_result]

    monkeypatch.setattr(app_module, "scan_symbols", fake_scan_symbols)

    client = TestClient(app_module.app)
    res = client.get("/api/scan", params={"symbols": "BTC/USDT"})
    assert res.status_code == 200
    body = res.json()
    assert body[0]["symbol"] == "BTC/USDT"
    assert body[0]["verdict"] == "Buy"


def test_api_scan_surfaces_exchange_errors(monkeypatch):
    def fake_scan_market(**kwargs):
        raise RuntimeError("exchange unreachable")

    monkeypatch.setattr(app_module, "scan_market", fake_scan_market)

    client = TestClient(app_module.app)
    res = client.get("/api/scan")
    assert res.status_code == 502
    assert "exchange unreachable" in res.json()["detail"]
