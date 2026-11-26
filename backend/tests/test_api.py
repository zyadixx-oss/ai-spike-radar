"""Every page's API endpoint responds (empty DB = honest empty states, never fabricated numbers)."""
from fastapi.testclient import TestClient

ENDPOINTS = ["/api/health", "/api/safety", "/api/config", "/api/symbols", "/api/radar", "/api/spikes", "/api/research",
             "/api/arena", "/api/rankings/timeframes", "/api/rankings/models", "/api/predictions", "/api/forward",
             "/api/shadow", "/api/alerts", "/api/jobs", "/api/reports/summary", "/api/stats/overview", "/api/events",
             "/api/models"]


def test_endpoints(engine):
    from app.main import app
    with TestClient(app) as c:
        for e in ENDPOINTS:
            r = c.get(e)
            assert r.status_code == 200, (e, r.text[:300])
        assert c.get("/api/radar").json()["rows"] == []
        assert c.get("/api/predictions").json()["total"] == 0
        assert c.post("/api/jobs/train?symbol=X").json()["queued"] is True
        assert c.post("/api/jobs/buy").status_code == 400
        r = c.get("/api/candles?symbol=X&tf=M1")
        assert r.status_code == 200 and r.json()["candles"] == []
        assert c.get("/api/candles?symbol=X&tf=W1").status_code == 400
