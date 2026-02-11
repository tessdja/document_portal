# utils/timezone.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_TZ = "UTC"  # safe, always available

def get_tz_name() -> str:
    return os.getenv("APP_TIMEZONE", DEFAULT_TZ)

def get_tz() -> ZoneInfo:
    tz_name = get_tz_name()
    try:
        return ZoneInfo(tz_name)
    except Exception:
        # Fallback to UTC if env var is invalid
        return ZoneInfo(DEFAULT_TZ)

def now_local() -> datetime:
    """Now in APP_TIMEZONE (or UTC fallback)."""
    return datetime.now(get_tz())

def now_utc() -> datetime:
    """Now in UTC (useful for storage, logs, stable filenames)."""
    return datetime.now(timezone.utc)
