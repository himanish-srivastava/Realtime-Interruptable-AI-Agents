"""Assembles the Holdfast control layer for one conversation and wires it to a
LiveKit ``AgentSession``. One runtime per room: nothing is shared across
conversations, so no state leaks between benchmark scenarios."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .gate import CommitGate, GatePolicy
from .ledger import ActionLedger
from .timeline import ConversationTimeline
from .toolkit import GatedToolkit, ToolSpec
from .watchdog import ReplyWatchdog

log = logging.getLogger("holdfast.runtime")


def build_realtime_model(cfg: dict[str, Any]):
    """OpenAI Realtime model configured from ``cfg`` (the declared provider)."""
    from livekit.plugins import openai as lk_openai
    from openai.types import realtime as rt
    from openai.types.realtime import realtime_audio_input_turn_detection as td

    m, t = cfg["model"], cfg["turn_detection"]
    if t.get("type", "semantic_vad") == "semantic_vad":
        turn = td.SemanticVad(
            type="semantic_vad", eagerness=t.get("eagerness", "medium"),
            create_response=True, interrupt_response=True,
        )
    else:
        turn = td.ServerVad(
            type="server_vad", silence_duration_ms=int(t.get("silence_duration_ms", 500)),
            create_response=True, interrupt_response=True,
        )
    kwargs: dict[str, Any] = dict(
        model=m["name"],
        voice=m.get("voice", "coral"),
        turn_detection=turn,
        input_audio_transcription=rt.AudioTranscription(
            model=m.get("transcription_model", "gpt-4o-mini-transcribe"),
            language=m.get("transcription_language") or None,
        ),
    )
    if m.get("noise_reduction"):
        kwargs["input_audio_noise_reduction"] = m["noise_reduction"]
    return lk_openai.realtime.RealtimeModel(**kwargs)


class LatencyHeartbeat:
    """Writes the reference agent's ``LATENCY_TRACK_JSON`` line (informational;
    run_tool_benchmark.py copies it into result files, scoring does not use it)."""

    def __init__(self, path: Optional[str], room: str) -> None:
        self.path = Path(path) if path else None
        self.room = room
        self.user_done_at = 0.0
        self.tool_start_at = 0.0
        self.tool_end_at = 0.0
        self.written = False

    def user_input(self) -> None:
        if not self.user_done_at:
            self.user_done_at = time.time()

    def tool_executed(self, started: float, ended: float) -> None:
        if not self.tool_start_at:
            self.tool_start_at, self.tool_end_at = started, ended

    def agent_speaking(self) -> None:
        if self.written or not self.user_done_at or self.path is None:
            return
        agent_start = time.time()
        metrics = {
            "room": self.room,
            "tool": "holdfast",
            "reasoning": round((self.tool_start_at - self.user_done_at) if self.tool_start_at else 0.0, 3),
            "execution": round((self.tool_end_at - self.tool_start_at) if self.tool_start_at else 0.0, 3),
            "synthesis": round(agent_start - (self.tool_end_at or self.user_done_at), 3),
            "total": round(agent_start - self.user_done_at, 3),
            "agent_start_at": agent_start,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("LATENCY_TRACK_JSON: " + json.dumps(metrics) + "\n")
        self.written = True


class HoldfastRuntime:
    def __init__(self, cfg: dict[str, Any], *, room_name: str, specs: list[ToolSpec]) -> None:
        self.cfg = cfg
        logs = cfg.get("logs", {})
        self.timeline = ConversationTimeline()
        self.gate = CommitGate(self.timeline, GatePolicy.from_dict(cfg.get("gate")))
        self.ledger = ActionLedger()
        self.heartbeat = LatencyHeartbeat(logs.get("heartbeat_log"), room_name)
        self.toolkit = GatedToolkit(
            specs, self.gate, self.ledger, room_name=room_name,
            call_log_path=logs.get("call_log"), audit_log_path=logs.get("audit_log"),
        )
        self._wrap_executor_for_heartbeat()
        wd = cfg.get("watchdog", {})
        self._session = None
        self.watchdog = ReplyWatchdog(
            self.timeline, self._nudge,
            after_tools_s=float(wd.get("after_tools_s", 2.5)),
            no_response_s=float(wd.get("no_response_s", 6.0)),
            max_nudges=int(wd.get("max_nudges", 2)),
            enabled=bool(wd.get("enabled", True)),
        )

    def _wrap_executor_for_heartbeat(self) -> None:
        original = self.toolkit._write_call

        def write_call(name, args, t_start, t_end):
            original(name, args, t_start, t_end)
            self.heartbeat.tool_executed(t_start, t_end)

        self.toolkit._write_call = write_call  # type: ignore[method-assign]

    def _nudge(self, prompt: str) -> None:
        if self._session is not None:
            self._session.generate_reply(instructions=prompt)

    def attach(self, session: Any) -> None:
        self._session = session
        tl, wd, hb = self.timeline, self.watchdog, self.heartbeat

        def on_user_state(ev):
            tl.on_user_state(ev.new_state)
            if ev.old_state == "speaking" and ev.new_state != "speaking":
                wd.user_turn_ended()

        def on_transcript(ev):
            tl.on_transcript(ev.transcript, ev.is_final)
            hb.user_input()

        def on_agent_state(ev):
            tl.on_agent_state(ev.new_state)
            if ev.new_state == "speaking":
                hb.agent_speaking()

        session.on("user_state_changed", on_user_state)
        session.on("user_input_transcribed", on_transcript)
        session.on("agent_state_changed", on_agent_state)
        session.on("function_tools_executed", lambda ev: wd.tools_finished())
