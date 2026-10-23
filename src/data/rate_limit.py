"""Per-minute + per-day request budget, persisted on disk so restarts don't reset it.

Alpha Vantage's free tier allows 25 requests/day and 5/minute. Going over returns a polite
JSON "Note" instead of data, so we stop *before* calling and fall back to the next provider.
"""

from __future__ import annotations

import time
from datetime import date

import diskcache


class RateLimiter:
    def __init__(self, cache: diskcache.Cache, name: str, per_minute: int, per_day: int):
        self.cache, self.name = cache, name
        self.per_minute, self.per_day = per_minute, per_day

    def _keys(self) -> tuple[str, str]:
        return f"rl:{self.name}:minute", f"rl:{self.name}:day:{date.today().isoformat()}"

    def remaining_today(self) -> int:
        return max(self.per_day - int(self.cache.get(self._keys()[1], 0)), 0)

    def try_acquire(self) -> bool:
        """Record one request if both budgets allow it; False means "don't call now"."""
        minute_key, day_key = self._keys()
        now = time.time()
        with self.cache.transact():
            recent = [t for t in self.cache.get(minute_key, []) if now - t < 60]
            used_today = int(self.cache.get(day_key, 0))
            if len(recent) >= self.per_minute or used_today >= self.per_day:
                return False
            self.cache.set(minute_key, [*recent, now], expire=120)
            self.cache.set(day_key, used_today + 1, expire=2 * 86_400)
        return True
