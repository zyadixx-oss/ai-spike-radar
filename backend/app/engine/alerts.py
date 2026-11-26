"""In-app alerts + optional Telegram, with de-duplication (one alert per symbol/horizon/event window)."""
from __future__ import annotations

import logging
import time

import httpx

from .. import db
from ..config import settings

log = logging.getLogger("alerts")
LEVEL_ORDER = {"gray": 0, "green": 1, "yellow": 2, "orange": 3, "red": 4}


def format_alert(p: dict) -> str:
    perf = p.get("performance") or {}
    reasons = "; ".join(r.get("text") or f"{r['feature']}={r['value']:.4g} (z={r['z']:+.1f})"
                        for r in (p.get("reasons") or [])[:3])
    hz = p["horizon"]
    exp = f"{p['horizon_size']} ticks" if p.get("horizon_kind") == "tick" else f"{p['horizon_size']}s"
    bt = (p.get("best_timeframe") or {}) or {}
    lines = [
        f"⚠️ AI Spike Radar — {p['level'].upper()} [{p.get('display_name', p['symbol'])}]",
        f"Direction: {'UP (Boom)' if p['direction'] > 0 else 'DOWN (Crash)'} · Horizon: {hz} (expires in {exp})",
        f"Probability: {p['probability']:.2%} vs base rate {p.get('base_rate') or 0:.2%}",
        f"Model: {p.get('model_name')} (approved={perf.get('approved')})",
        f"Validation precision at this level: {_pct(perf.get('level_precision_val'))} · "
        f"false-alarm share: {_pct(perf.get('false_alarm_share_val'))}",
        f"Test BSS vs base rate: {_num(perf.get('test_bss_vs_base_rate'))} · Best TF: {bt.get('timeframe', 'n/a')}",
        f"Reasons: {reasons or 'n/a'}",
        "Research signal only — no trades are placed (LIVE_TRADING=false).",
    ]
    return "\n".join(lines)


def _pct(x):  # noqa: ANN001, ANN202
    return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"


def _num(x):  # noqa: ANN001, ANN202
    return f"{x:+.4f}" if isinstance(x, (int, float)) else "n/a"


class AlertManager:
    def __init__(self, engine=None):  # noqa: ANN001
        self.engine = engine or db.get_engine()
        self.pending_tg: list[tuple[int, str]] = []
        self.active: dict[tuple[str, str], float] = {}  # (symbol, horizon) -> suppress-until epoch

    def consider(self, prediction_id: int, p: dict, issued_epoch: int) -> int | None:
        min_lvl = LEVEL_ORDER.get(settings.alert_min_level, 3)
        if LEVEL_ORDER.get(p.get("level", "gray"), 0) < min_lvl or not p.get("approved"):
            return None
        key = (p["symbol"], p["horizon"])
        if self.active.get(key, 0) > issued_epoch:
            return None  # same event window already alerted
        window = p["horizon_size"] if p.get("horizon_kind") == "sec" else int(p["horizon_size"] * 2)
        bucket = issued_epoch // max(window, 1)
        dedupe = f"{p['symbol']}:{p['horizon']}:{bucket}"
        msg = format_alert(p)
        try:
            with self.engine.begin() as conn:
                r = conn.execute(db.alerts.insert().values(created_at=time.time(), symbol=p["symbol"],
                                                           horizon=p["horizon"], level=p["level"],
                                                           prediction_id=prediction_id, dedupe_key=dedupe, message=msg))
                aid = int(r.inserted_primary_key[0])
        except Exception:  # noqa: BLE001 — unique dedupe key => duplicate
            return None
        self.active[key] = issued_epoch + window
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self.pending_tg.append((aid, msg))
        return aid

    async def flush_telegram(self) -> None:
        if not self.pending_tg:
            return
        items, self.pending_tg = self.pending_tg, []
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            for aid, msg in items:
                err = None
                try:
                    r = await client.post(url, json={"chat_id": settings.telegram_chat_id, "text": msg})
                    if r.status_code != 200:
                        err = f"HTTP {r.status_code}: {r.text[:200]}"
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                with self.engine.begin() as conn:
                    conn.execute(db.alerts.update().where(db.alerts.c.id == aid)
                                 .values(telegram_sent=err is None, telegram_error=err))
