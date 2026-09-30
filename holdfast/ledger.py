"""Idempotency ledger: the same action never runs twice in one conversation.

* An identical call (same tool, semantically equal arguments) that already
  completed returns the cached result instead of executing again.
* An identical call that is still running joins that execution.
* Failed executions are not recorded, so a retry really retries.
* "Slots" let a domain say that one action supersedes another (e.g. a new
  navigation destination replaces the old route) without double-acting.

Each conversation (LiveKit room) gets a fresh ledger: nothing is cached
across scenarios.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional


def normalize_value(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items() if x is not None}
    if isinstance(v, (list, tuple)):
        return [normalize_value(x) for x in v]
    return v


def canonical_key(name: str, args: dict[str, Any]) -> str:
    return name + ":" + json.dumps(normalize_value(args), sort_keys=True, default=str)


class ActionLedger:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._done: dict[str, Any] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._slots: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []

    def has_completed(self, name: str, args: dict[str, Any]) -> bool:
        return canonical_key(name, args) in self._done

    async def run(
        self, name: str, args: dict[str, Any], fn: Callable[[], Awaitable[Any]]
    ) -> tuple[Any, str]:
        key = canonical_key(name, args)
        if key in self._done:
            self._record(name, args, "duplicate")
            return self._done[key], "duplicate"
        if key in self._inflight:
            result = await asyncio.shield(self._inflight[key])
            self._record(name, args, "joined")
            return result, "joined"

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await fn()
        except BaseException as e:
            self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(e)
                fut.exception()  # mark retrieved; joiners re-raise via shield
            self._record(name, args, "failed")
            raise
        self._inflight.pop(key, None)
        self._done[key] = result
        if not fut.done():
            fut.set_result(result)
        self._record(name, args, "executed")
        return result, "executed"

    # ── supersede semantics ─────────────────────────────────────────
    def claim_slot(self, slot: str, key: str) -> Optional[str]:
        """Make ``key`` the owner of ``slot``; return the previous owner, if any."""
        prev = self._slots.get(slot)
        self._slots[slot] = key
        return prev

    def slot_owner(self, slot: str) -> Optional[str]:
        return self._slots.get(slot)

    def _record(self, name: str, args: dict[str, Any], status: str) -> None:
        self.history.append({"t": self._clock(), "tool": name, "args": args, "status": status})
