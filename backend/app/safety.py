"""Hard safety constraints. Research / monitoring / virtual testing ONLY.

These values are module constants, are not read from the environment and are asserted at
startup. Any code path that tries to place an order must go through `assert_no_trading`,
which always raises.
"""
from __future__ import annotations

import os

ready_for_live = False
live_allowed = False
opened_trades = False
LIVE_TRADING = False

# The ONLY Deriv API message types this application is permitted to send (read-only market data).
ALLOWED_DERIV_CALLS = frozenset({
    "active_symbols", "ticks_history", "ticks", "forget", "forget_all", "ping", "time", "website_status",
})

# Explicitly listed for clarity/testing — all are rejected (anything not in the allowlist is rejected anyway).
FORBIDDEN_DERIV_CALLS = frozenset({
    "buy", "sell", "proposal", "proposal_open_contract", "buy_contract_for_multiple_accounts",
    "sell_contract_for_multiple_accounts", "sell_expired", "cancel", "contract_update", "authorize",
    "transfer_between_accounts", "topup_virtual", "paymentagent_transfer", "cashier", "mt5_deposit",
    "mt5_new_account", "p2p_order_create", "new_account_real", "set_settings",
})


class TradingForbiddenError(RuntimeError):
    """Raised whenever anything attempts to trade or authenticate."""


def assert_no_trading(action: str = "trade") -> None:
    raise TradingForbiddenError(
        f"'{action}' is forbidden: this system is research-only (LIVE_TRADING=false, live_allowed=false)."
    )


def check_outgoing_message(msg: dict) -> None:
    """Validate an outgoing Deriv API request. Raises TradingForbiddenError if not a read-only call."""
    keys = set(msg.keys()) - {"req_id", "passthrough", "subscribe", "echo_req"}
    call_keys = keys & (ALLOWED_DERIV_CALLS | FORBIDDEN_DERIV_CALLS)
    if keys & FORBIDDEN_DERIV_CALLS:
        assert_no_trading(",".join(sorted(keys & FORBIDDEN_DERIV_CALLS)))
    if not call_keys:
        # Unknown request type: reject rather than risk it.
        assert_no_trading("unknown-call:" + ",".join(sorted(keys)))


def status() -> dict:
    return {
        "ready_for_live": ready_for_live,
        "live_allowed": live_allowed,
        "opened_trades": opened_trades,
        "LIVE_TRADING": LIVE_TRADING,
        "mode": "research_only",
        "allowed_deriv_calls": sorted(ALLOWED_DERIV_CALLS),
    }


def startup_check() -> None:
    flags = (ready_for_live, live_allowed, opened_trades, LIVE_TRADING)
    if any(flags):
        raise SystemExit("SAFETY VIOLATION: live trading flags must all be False")
    for env_name in ("LIVE_TRADING", "LIVE_ALLOWED", "READY_FOR_LIVE"):
        v = os.environ.get(env_name, "false").strip().lower()
        if v in ("1", "true", "yes", "on"):
            raise SystemExit(f"SAFETY VIOLATION: environment variable {env_name} attempts to enable live trading")
    if os.environ.get("DERIV_API_TOKEN"):
        raise SystemExit("SAFETY VIOLATION: no API token may be configured; this app never authenticates")
