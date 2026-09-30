"""A small, testable record of what happened in the conversation and when.

Fed by LiveKit session events (user/agent state, transcripts, tool activity);
read by the commit gate and the reply watchdog. All times come from an
injectable clock so behaviour can be tested deterministically.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional


class ConversationTimeline:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.user_speaking = False
        self._segments: list[list[Optional[float]]] = []  # [start, end|None]
        self._finals: list[tuple[float, str]] = []
        self._interim = ""
        self.agent_state = "initializing"
        self._agent_speech_starts: list[float] = []
        self._tool_activity: list[float] = []

    # ── user speech (VAD) ───────────────────────────────────────────
    def on_user_state(self, state: str, at: Optional[float] = None) -> None:
        at = self._clock() if at is None else at
        if state == "speaking":
            if not self.user_speaking:
                self.user_speaking = True
                self._segments.append([at, None])
        elif self.user_speaking:
            self.user_speaking = False
            self._segments[-1][1] = at

    def silence_duration(self, now: Optional[float] = None) -> float:
        now = self._clock() if now is None else now
        if self.user_speaking:
            return 0.0
        end = self.last_user_speech_end()
        return math.inf if end is None else max(0.0, now - end)

    def last_user_speech_end(self) -> Optional[float]:
        for start, end in reversed(self._segments):
            if end is not None:
                return end
        return None

    def speech_after(self, t: float, now: Optional[float] = None) -> float:
        """Seconds of user speech that happened after time ``t``."""
        now = self._clock() if now is None else now
        total = 0.0
        for start, end in self._segments:
            s = max(start, t)
            e = min(now if end is None else end, now)
            if e > s:
                total += e - s
        return total

    # ── transcripts ─────────────────────────────────────────────────
    def on_transcript(self, text: str, is_final: bool, at: Optional[float] = None) -> None:
        at = self._clock() if at is None else at
        text = (text or "").strip()
        if is_final:
            if text:
                self._finals.append((at, text))
            self._interim = ""
        elif text:
            self._interim = text

    def user_text(self) -> str:
        parts = [t for _, t in self._finals]
        if self._interim:
            parts.append(self._interim)
        return " ".join(parts)

    def last_final_at(self) -> Optional[float]:
        return self._finals[-1][0] if self._finals else None

    def transcript_pending(self) -> bool:
        """True when the user's latest speech has no final transcript yet."""
        end = self.last_user_speech_end()
        if end is None:
            return False
        last = self.last_final_at()
        return last is None or last < end

    # ── agent side ──────────────────────────────────────────────────
    def on_agent_state(self, state: str, at: Optional[float] = None) -> None:
        at = self._clock() if at is None else at
        if state == "speaking" and self.agent_state != "speaking":
            self._agent_speech_starts.append(at)
        self.agent_state = state

    def on_tool_activity(self, at: Optional[float] = None) -> None:
        self._tool_activity.append(self._clock() if at is None else at)

    def last_agent_speech_start(self) -> Optional[float]:
        return self._agent_speech_starts[-1] if self._agent_speech_starts else None

    def agent_spoke_after(self, t: float) -> bool:
        return any(s > t for s in self._agent_speech_starts)

    def tool_activity_after(self, t: float) -> bool:
        return any(s > t for s in self._tool_activity)

    def final_transcript_after(self, t: float) -> bool:
        return any(at > t for at, _ in self._finals)
