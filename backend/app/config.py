"""Runtime configuration (env-driven). Safety flags live in app/safety.py and are NOT configurable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Requested instruments (display names). Programmatic symbols are DISCOVERED, never assumed.
TARGET_INDICES = [
    ("Boom", 300), ("Boom", 500), ("Boom", 600), ("Boom", 900), ("Boom", 1000),
    ("Crash", 300), ("Crash", 500), ("Crash", 600), ("Crash", 900), ("Crash", 1000),
]

# Timeframes in seconds. 0 == raw tick.
TIMEFRAMES: dict[str, int] = {
    "tick": 0, "5s": 5, "10s": 10, "15s": 15, "30s": 30,
    "M1": 60, "M2": 120, "M3": 180, "M5": 300, "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}
# Granularities Deriv serves directly as candles (others are built from ticks).
DERIV_NATIVE_GRANULARITIES = {60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400}

# Prediction horizons: (key, kind, size)
HORIZONS: list[tuple[str, str, int]] = [
    ("10s", "sec", 10), ("30s", "sec", 30), ("1m", "sec", 60), ("3m", "sec", 180),
    ("5m", "sec", 300), ("15m", "sec", 900), ("50t", "tick", 50), ("100t", "tick", 100), ("200t", "tick", 200),
]
HORIZON_MAP = {k: (kind, n) for k, kind, n in HORIZONS}


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///./data/spike_radar.db"))
    deriv_ws_url: str = field(default_factory=lambda: _env("DERIV_WS_URL", "wss://api.derivws.com/trading/v1/options/ws/public"))
    deriv_app_id: str = field(default_factory=lambda: _env("DERIV_APP_ID", "1089"))
    deriv_ws_proxy: str = field(default_factory=lambda: _env("DERIV_WS_PROXY", "auto"))  # auto | none | URL
    deriv_max_rps: float = field(default_factory=lambda: _float("DERIV_MAX_RPS", 2.0))
    max_backfill_days: int = field(default_factory=lambda: _int("MAX_BACKFILL_DAYS", 30))
    gap_factor: float = field(default_factory=lambda: _float("GAP_FACTOR", 10.0))
    gap_min_seconds: float = field(default_factory=lambda: _float("GAP_MIN_SECONDS", 15.0))

    spike_k: float = field(default_factory=lambda: _float("SPIKE_K", 8.0))

    # Research / training limits
    min_spikes_for_training: int = field(default_factory=lambda: _int("MIN_SPIKES_FOR_TRAINING", 60))
    max_train_rows: int = field(default_factory=lambda: _int("MAX_TRAIN_ROWS", 60000))
    sample_stride: int = field(default_factory=lambda: _int("SAMPLE_STRIDE", 3))
    bootstrap_iters: int = field(default_factory=lambda: _int("BOOTSTRAP_ITERS", 300))
    feature_timeframes: list[str] = field(default_factory=lambda: _env(
        "FEATURE_TIMEFRAMES", "5s,15s,30s,M1,M5,M15,H1").split(","))
    enable_deep_models: bool = field(default_factory=lambda: _bool("ENABLE_DEEP_MODELS", True))

    # Virtual (simulated) trading costs — assumption, configurable
    sim_cost_per_trade_pct: float = field(default_factory=lambda: _float("SIM_COST_PER_TRADE_PCT", 0.02))

    # Schedules (seconds)
    retrain_interval: int = field(default_factory=lambda: _int("RETRAIN_INTERVAL", 6 * 3600))
    arena_interval: int = field(default_factory=lambda: _int("ARENA_INTERVAL", 12 * 3600))
    research_interval: int = field(default_factory=lambda: _int("RESEARCH_INTERVAL", 3 * 3600))
    spike_scan_interval: int = field(default_factory=lambda: _int("SPIKE_SCAN_INTERVAL", 300))
    evaluate_interval: int = field(default_factory=lambda: _int("EVALUATE_INTERVAL", 30))
    radar_interval: float = field(default_factory=lambda: _float("RADAR_INTERVAL", 1.0))

    # Process roles
    run_collector: bool = field(default_factory=lambda: _bool("RUN_COLLECTOR", True))
    run_worker: bool = field(default_factory=lambda: _bool("RUN_WORKER", True))
    slow_jobs_in_process: bool = field(default_factory=lambda: _bool("SLOW_JOBS_IN_PROCESS", True))
    run_radar: bool = field(default_factory=lambda: _bool("RUN_RADAR", True))

    # Alerts
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))
    alert_min_level: str = field(default_factory=lambda: _env("ALERT_MIN_LEVEL", "orange"))

    cors_origins: list[str] = field(default_factory=lambda: _env("CORS_ORIGINS", "*").split(","))
    models_dir: str = field(default_factory=lambda: _env("MODELS_DIR", "./data/models"))


settings = Settings()
