"""Reply watchdog: the agent must never go silent.

Two failure modes seen in the FDB-v3 baselines are covered:

* "silent worker" - tools ran, results came back, but the model never spoke
  (the response-quality judge scores the spoken transcript, so this loses the
  scenario even when every tool call was right). After ``after_tools_s`` of
  silence following tool completion, the watchdog asks the model to report.
* no response - the user finished a real (transcribed) request and nothing
  happened: no speech, no tool activity. After ``no_response_s`` the watchdog
  asks the model to handle the request.

Nudges never fire while the user or the agent is speaking, and are capped per
conversation. The idempotency ledger makes a nudge that re-issues a tool call
harmless.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from .timeline import ConversationTimeline

log = logging.getLogger("holdfast.watchdog")

REPORT_PROMPT = (
    "You received tool results but have not spoken yet. In one or two short sentences, tell the "
    "user the outcome of every action you completed, using the exact values from the tool "
    "results. If a step the user asked for has not been done yet, call that tool now instead."
)
RESPOND_PROMPT = (
    "The user finished speaking and you have not responded. Handle their complete request now: "
    "call the needed tools in the order they asked (using their final, corrected values), then "
    "tell them the results."
)


class ReplyWatchdog:
    def __init__(
        self,
        timeline: ConversationTimeline,
        nudge: Callable[[str], None],
        *,
        after_tools_s: float = 2.5,
        no_response_s: float = 6.0,
        max_nudges: int = 2,
        enabled: bool = True,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.tl = timeline
        self._nudge = nudge
        self.after_tools_s = after_tools_s
        self.no_response_s = no_response_s
        self.max_nudges = max_nudges
        self.enabled = enabled
        self.clock = clock
        self.sleep = sleep
        self.nudges = 0
        self._tasks: set[asyncio.Task] = set()
        self._deferred: list[Callable[[], Awaitable[None]]] = []

    # ── triggers ────────────────────────────────────────────────────
    def tools_finished(self) -> None:
        if self.enabled:
            t = self.clock()
            self._schedule(lambda: self._after_tools(t))

    def user_turn_ended(self) -> None:
        if self.enabled:
            t = self.clock()
            self._schedule(lambda: self._no_response(t))

    async def drain(self) -> None:
        while self._deferred:
            await self._deferred.pop(0)()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ── checks ──────────────────────────────────────────────────────
    async def _after_tools(self, t: float) -> None:
        await self.sleep(self.after_tools_s)
        tl = self.tl
        if tl.user_speaking or tl.agent_state in ("speaking", "thinking") or tl.agent_spoke_after(t):
            return
        if tl.tool_activity_after(t):  # the model is still working (e.g. a re-issued call)
            return
        if tl.speech_after(t) >= 0.3:  # the user said more; the model will answer that turn
            return
        self._fire(REPORT_PROMPT, "tools finished but agent stayed silent")

    async def _no_response(self, t: float) -> None:
        await self.sleep(self.no_response_s)
        tl = self.tl
        if tl.user_speaking or tl.agent_state in ("speaking", "thinking"):
            return
        if tl.agent_spoke_after(t) or tl.tool_activity_after(t):
            return
        last_agent = tl.last_agent_speech_start()
        if not tl.final_transcript_after(last_agent if last_agent is not None else float("-inf")):
            return  # nothing new was actually said (e.g. VAD fired on noise)
        self._fire(RESPOND_PROMPT, "user turn ended without any agent reaction")

    def _fire(self, prompt: str, why: str) -> None:
        if self.nudges >= self.max_nudges:
            return
        self.nudges += 1
        log.info("watchdog nudge (%s)", why)
        try:
            self._nudge(prompt)
        except Exception:
            log.exception("watchdog nudge failed")

    def _schedule(self, fn: Callable[[], Awaitable[None]]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._deferred.append(fn)
            return
        task = loop.create_task(fn())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
