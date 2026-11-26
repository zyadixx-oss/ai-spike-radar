"""Trading must be impossible: constants, outgoing-message allowlist, startup guard, API and source scan."""
import os
import re
from pathlib import Path

import pytest

from app import safety
from app.deriv.client import DerivClient


def test_flags_are_false():
    assert safety.ready_for_live is False
    assert safety.live_allowed is False
    assert safety.opened_trades is False
    assert safety.LIVE_TRADING is False
    st = safety.status()
    assert st["mode"] == "research_only" and not any(st[k] for k in ("ready_for_live", "live_allowed",
                                                                         "opened_trades", "LIVE_TRADING"))


@pytest.mark.parametrize("msg", [
    {"buy": "123", "price": 10}, {"sell": 1, "price": 0}, {"proposal": 1, "amount": 10, "contract_type": "CALL"},
    {"authorize": "token"}, {"buy_contract_for_multiple_accounts": "1"}, {"cancel": 1}, {"contract_update": 1},
    {"transfer_between_accounts": 1}, {"some_new_call": 1}, {"ticks_history": "X", "buy": 1},
])
def test_forbidden_messages_rejected(msg):
    with pytest.raises(safety.TradingForbiddenError):
        safety.check_outgoing_message(msg)


@pytest.mark.parametrize("msg", [
    {"ticks_history": "R_100", "end": "latest", "count": 10}, {"ticks": "R_100", "subscribe": 1},
    {"active_symbols": "brief"}, {"ping": 1}, {"forget_all": "ticks"},
])
def test_market_data_allowed(msg):
    safety.check_outgoing_message(msg)


async def test_client_refuses_before_network():
    c = DerivClient("ws://127.0.0.1:9", "1")  # never connected — refusal happens before any I/O
    with pytest.raises(safety.TradingForbiddenError):
        await c.request({"buy": "1", "price": 1})
    with pytest.raises(safety.TradingForbiddenError):
        await c.buy()
    with pytest.raises(safety.TradingForbiddenError):
        await c.sell()


def test_startup_guard_rejects_live_env(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(SystemExit):
        safety.startup_check()
    monkeypatch.delenv("LIVE_TRADING")
    monkeypatch.setenv("DERIV_API_TOKEN", "abc")
    with pytest.raises(SystemExit):
        safety.startup_check()
    monkeypatch.delenv("DERIV_API_TOKEN")
    safety.startup_check()


def test_api_trade_endpoint_forbidden(engine):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/api/trade")
        assert r.status_code == 403
        assert c.get("/api/safety").json()["LIVE_TRADING"] is False


def test_no_order_messages_in_source():
    """Static scan: no app code builds a Deriv request with a trading key (except the safety lists)."""
    root = Path(__file__).resolve().parents[1] / "app"
    pat = re.compile(r"""["'](buy|sell|proposal|authorize|buy_contract_for_multiple_accounts)["']\s*:""")
    offenders = []
    for f in root.rglob("*.py"):
        if f.name == "safety.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, offenders
    assert os.environ.get("LIVE_TRADING", "false").lower() != "true"
