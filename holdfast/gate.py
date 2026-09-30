"""The commit gate: decides whether a tool call may execute *now*.

A realtime model can emit a tool call before the user has finished (or while
a correction is on its way). Executing that call is irreversible for scoring
and for the real world. The gate holds each call until the user has been
silent for a short settle window, then:

* STALE     - the user kept speaking after the call was issued (or barged in),
              so the arguments may be out of date. Nothing executes; the model
              is told to re-issue with the final values.
* CHALLENGE - the transcript suggests an argument value was self-corrected.
              Nothing executes; the model is asked to re-check once. Issuing
              the same value again is treated as confirmation.
* EXECUTE   - safe to run.

LiveKit does not cancel an in-flight tool when the user barges in (it waits
for the tool to finish), so this check has to live inside the tool layer.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .ledger import normalize_value
from .repair import describe, find_reparanda
from .timeline import ConversationTimeline

log = logging.getLogger("holdfast.gate")


class Verdict(enum.Enum):
    EXECUTE = "execute"
    STALE = "stale"
    CHALLENGE = "challenge"


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""
    waited_s: float = 0.0


@dataclass
class GatePolicy:
    settle_read_s: float = 0.35  # silence required before read-only calls run
    settle_write_s: float = 0.9  # silence required before state-changing calls run
    min_resume_s: float = 0.3  # user speech after the call shorter than this is noise
    max_wait_s: float = 10.0  # never hold a call longer than this
    max_stale_per_tool: int = 2  # after this many STALE verdicts for a tool, execute anyway
    repair_check: bool = True
    transcript_wait_s: float = 0.8  # mutating calls: wait this long for a pending transcript
    poll_s: float = 0.05

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GatePolicy":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class CommitGate:
    timeline: ConversationTimeline
    policy: GatePolicy = field(default_factory=GatePolicy)
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _stale_counts: Counter = field(default_factory=Counter)
    _challenged: set = field(default_factory=set)

    async def admit(
        self,
        name: str,
        args: dict[str, Any],
        *,
        mutating: bool,
        interrupted: Callable[[], bool],
    ) -> Decision:
        p = self.policy
        t0 = self.clock()
        deadline = t0 + p.max_wait_s
        settle = p.settle_write_s if mutating else p.settle_read_s

        # 1) hold until the user has been quiet for `settle` seconds
        while True:
            if interrupted():
                return self._stale(name, "the user barged in while this call was pending", t0)
            now = self.clock()
            if (
                not self.timeline.user_speaking
                and self.timeline.speech_after(t0, now) >= p.min_resume_s
                and self._stale_counts[name] < p.max_stale_per_tool
            ):
                # the user added speech after the call and has now paused: report stale
                # right away so the model can re-issue with the complete request
                return self._stale(name, "the user kept speaking after this call was issued", t0)
            if not self.timeline.user_speaking and self.timeline.silence_duration(now) >= settle:
                break
            if now >= deadline:
                break
            if self.timeline.user_speaking:
                step = p.poll_s
            else:
                step = min(p.poll_s, max(settle - self.timeline.silence_duration(now), 0.001))
            await self.sleep(min(step, max(deadline - now, 0.001)))

        # 2) did the user keep talking after the model issued this call?
        if self.timeline.speech_after(t0) >= p.min_resume_s:
            if self._stale_counts[name] < p.max_stale_per_tool:
                return self._stale(name, "the user kept speaking after this call was issued", t0)
            log.warning("stale cap reached for %s; executing", name)

        # 3) self-correction check against the transcript
        if p.repair_check:
            if mutating and self.timeline.transcript_pending():
                t_wait = self.clock() + p.transcript_wait_s
                while self.timeline.transcript_pending() and self.clock() < t_wait:
                    if interrupted():
                        return self._stale(name, "the user barged in while this call was pending", t0)
                    await self.sleep(p.poll_s)
            hits = [
                h
                for h in find_reparanda(self.timeline.user_text(), args)
                if (name, h.arg, normalize_value(h.value)) not in self._challenged
            ]
            if hits:
                for h in hits:
                    self._challenged.add((name, h.arg, normalize_value(h.value)))
                return Decision(
                    Verdict.CHALLENGE,
                    f"possible self-correction: {describe(hits)}",
                    self.clock() - t0,
                )

        return Decision(Verdict.EXECUTE, "", self.clock() - t0)

    def _stale(self, name: str, reason: str, t0: float) -> Decision:
        self._stale_counts[name] += 1
        return Decision(Verdict.STALE, reason, self.clock() - t0)
