"""Discover Boom/Crash symbols from Deriv `active_symbols` by display name — never by assumed codes."""
from __future__ import annotations

import re
import time

from ..config import TARGET_INDICES

_NAME_RE = re.compile(r"^\s*(boom|crash)\s+(\d+)\s*(index)?\s*$", re.IGNORECASE)


def match_boom_crash(active: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (rows for found target indices, list of missing target display names)."""
    wanted = {(f.lower(), n) for f, n in TARGET_INDICES}
    found: dict[tuple[str, int], dict] = {}
    for s in active:
        name = str(s.get("display_name", ""))
        m = _NAME_RE.match(name)
        if not m:
            continue
        fam, n = m.group(1).lower(), int(m.group(2))
        if (fam, n) not in wanted:
            continue
        # Prefer the plain index (some platforms list variants); first match wins deterministically
        key = (fam, n)
        if key in found:
            continue
        now = time.time()
        found[key] = {
            "symbol": s["symbol"],
            "display_name": name,
            "family": fam,
            "n": n,
            "direction": 1 if fam == "boom" else -1,
            "pip": float(s["pip"]) if s.get("pip") is not None else None,
            "market": s.get("market"),
            "submarket": s.get("submarket"),
            "available": True,
            "exchange_is_open": bool(s.get("exchange_is_open", 1)),
            "discovered_at": now,
            "last_seen_at": now,
        }
    missing = [f"{f} {n} Index" for f, n in TARGET_INDICES if (f.lower(), n) not in found]
    ordered = [found[(f.lower(), n)] for f, n in TARGET_INDICES if (f.lower(), n) in found]
    return ordered, missing
