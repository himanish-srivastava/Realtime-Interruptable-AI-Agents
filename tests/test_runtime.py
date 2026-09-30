import asyncio
import json
import os

from holdfast.config import load_config
from holdfast.prompts import build_instructions
from holdfast.runtime import HoldfastRuntime, build_realtime_model
from holdfast.toolkit import ToolSpec


def test_default_config_loads_with_expected_sections():
    cfg = load_config()
    assert cfg["provider_id"] == "holdfast"
    assert cfg["tools"]["schema"] == "extended"
    assert cfg["gate"]["settle_write_s"] > cfg["gate"]["settle_read_s"]


def test_config_overrides_from_env_are_typed(monkeypatch):
    monkeypatch.setenv("HOLDFAST_SET", "gate.settle_write_s=1.25, tools.schema=template, watchdog.enabled=false")
    cfg = load_config()
    assert cfg["gate"]["settle_write_s"] == 1.25
    assert cfg["tools"]["schema"] == "template"
    assert cfg["watchdog"]["enabled"] is False


def test_config_file_override(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("gate:\n  settle_read_s: 0.1\n")
    monkeypatch.setenv("HOLDFAST_CONFIG", str(p))
    cfg = load_config()
    assert cfg["gate"]["settle_read_s"] == 0.1
    assert cfg["gate"]["settle_write_s"] > 0  # defaults still merged


def test_instructions_cover_repairs_ordering_and_no_early_claims():
    text = build_instructions(acknowledge=False).lower()
    for phrase in ("last stated value wins", "in the order", "never say something is done", "not_executed"):
        assert phrase in text
    assert "without saying anything first" in text
    assert "on it" in build_instructions(acknowledge=True).lower()


def test_realtime_model_is_built_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    cfg = load_config()
    model = build_realtime_model(cfg)
    assert type(model).__name__ == "RealtimeModel"


class FakeEvent:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSession:
    def __init__(self):
        self.handlers = {}
        self.replies = []

    def on(self, name, cb=None):
        self.handlers.setdefault(name, []).append(cb)
        return cb

    def emit(self, name, ev):
        for cb in self.handlers.get(name, []):
            cb(ev)

    def generate_reply(self, **kw):
        self.replies.append(kw)


def make_runtime(tmp_path, **cfg_over):
    cfg = load_config()
    cfg["logs"] = {"call_log": str(tmp_path / "calls.log"), "heartbeat_log": str(tmp_path / "hb.log"),
                   "audit_log": str(tmp_path / "audit.log")}
    cfg["watchdog"]["after_tools_s"] = 0.01
    cfg["watchdog"]["no_response_s"] = 0.01
    for k, v in cfg_over.items():
        cfg[k].update(v)

    async def handler(args):
        return {"status": "success"}

    spec = ToolSpec("track_order", "d", {"type": "object", "properties": {"order_id": {"type": "string"}},
                                          "required": ["order_id"]}, False, handler)
    return HoldfastRuntime(cfg, room_name="eval-1", specs=[spec])


def test_session_events_feed_the_timeline(tmp_path):
    rt = make_runtime(tmp_path)
    s = FakeSession()
    rt.attach(s)
    s.emit("user_state_changed", FakeEvent(old_state="listening", new_state="speaking"))
    assert rt.timeline.user_speaking
    s.emit("user_input_transcribed", FakeEvent(transcript="track order Q1", is_final=True))
    s.emit("user_state_changed", FakeEvent(old_state="speaking", new_state="listening"))
    s.emit("agent_state_changed", FakeEvent(old_state="listening", new_state="thinking"))
    assert not rt.timeline.user_speaking
    assert rt.timeline.user_text() == "track order Q1"
    assert rt.timeline.agent_state == "thinking"


def test_watchdog_nudges_through_session_generate_reply(tmp_path):
    rt = make_runtime(tmp_path)
    s = FakeSession()

    async def go():
        rt.attach(s)
        s.emit("agent_state_changed", FakeEvent(old_state="thinking", new_state="listening"))
        s.emit("function_tools_executed", FakeEvent(function_calls=[], function_call_outputs=[]))
        await asyncio.sleep(0.05)
        await rt.watchdog.drain()

    asyncio.run(go())
    assert len(s.replies) == 1 and "instructions" in s.replies[0]


def test_heartbeat_line_is_written_in_the_reference_format(tmp_path):
    rt = make_runtime(tmp_path, watchdog={"enabled": False})
    s = FakeSession()

    async def go():
        rt.attach(s)
        s.emit("user_input_transcribed", FakeEvent(transcript="track order Q1", is_final=True))
        await rt.toolkit.invoke("track_order", {"order_id": "Q1"})
        s.emit("agent_state_changed", FakeEvent(old_state="thinking", new_state="speaking"))

    asyncio.run(go())
    lines = [l for l in (tmp_path / "hb.log").read_text().splitlines() if l.startswith("LATENCY_TRACK_JSON: ")]
    assert len(lines) == 1
    m = json.loads(lines[0][len("LATENCY_TRACK_JSON: "):])
    assert m["room"] == "eval-1"
    assert {"reasoning", "execution", "synthesis", "total", "agent_start_at"} <= set(m)
