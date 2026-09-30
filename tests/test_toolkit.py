import asyncio
import json

from livekit.agents import RunContext
from livekit.agents.llm.tool_context import RawFunctionTool
from livekit.agents.llm.utils import prepare_function_arguments

from holdfast.gate import CommitGate, GatePolicy
from holdfast.ledger import ActionLedger
from holdfast.timeline import ConversationTimeline
from holdfast.toolkit import GatedToolkit, ToolSpec

FAST = GatePolicy(settle_read_s=0.0, settle_write_s=0.0, transcript_wait_s=0.0, poll_s=0.001)


def commute_spec(calls):
    async def handler(args):
        calls.append(args)
        return {"status": "success", "duration_mins": 25}

    return ToolSpec(
        name="calculate_commute",
        description="Commute time.",
        parameters={
            "type": "object",
            "properties": {
                "origin_address": {"type": "string"},
                "destination_address": {"type": "string"},
                "mode": {"type": "string", "default": "driving"},
                "max_minutes": {"type": "integer"},
                "avoid_tolls": {"type": "boolean"},
            },
            "required": ["origin_address", "destination_address"],
        },
        mutating=False,
        handler=handler,
    )


def make(tmp_path, specs, policy=FAST, speaking_after=None):
    tl = ConversationTimeline()
    gate = CommitGate(tl, policy)
    kit = GatedToolkit(
        specs,
        gate,
        ActionLedger(),
        room_name="eval-abc",
        call_log_path=tmp_path / "calls.log",
        audit_log_path=tmp_path / "audit.log",
    )
    return kit, tl


def log_lines(tmp_path):
    p = tmp_path / "calls.log"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def test_executed_call_is_logged_in_the_exact_harness_format(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    out = asyncio.run(kit.invoke("calculate_commute", {"origin_address": "APT1", "destination_address": "Downtown"}))
    assert json.loads(out) == {"status": "success", "duration_mins": 25}
    (line,) = log_lines(tmp_path)
    assert set(line) == {"room", "call"}
    assert line["room"] == "eval-abc"
    assert set(line["call"]) == {"function", "args", "timestamp_start", "timestamp_end"}
    assert line["call"]["function"] == "calculate_commute"
    assert line["call"]["args"] == {"origin_address": "APT1", "destination_address": "Downtown", "mode": "driving"}
    assert line["call"]["timestamp_start"] <= line["call"]["timestamp_end"]


def test_arguments_are_cleaned_coerced_and_unknown_keys_dropped(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    asyncio.run(
        kit.invoke(
            "calculate_commute",
            {
                "origin_address": " APT1 ",
                "destination_address": "Downtown",
                "mode": "",
                "max_minutes": "30",
                "avoid_tolls": "true",
                "made_up": "x",
            },
        )
    )
    assert calls == [
        {
            "origin_address": "APT1",
            "destination_address": "Downtown",
            "mode": "driving",
            "max_minutes": 30,
            "avoid_tolls": True,
        }
    ]


def test_missing_required_argument_is_reported_and_nothing_runs(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    out = json.loads(asyncio.run(kit.invoke("calculate_commute", {"origin_address": "APT1"})))
    assert out["status"] == "not_executed"
    assert "destination_address" in out["reason"]
    assert calls == [] and log_lines(tmp_path) == []


def test_stale_call_does_not_execute_or_log(tmp_path):
    calls = []
    kit, tl = make(tmp_path, [commute_spec(calls)])
    out = json.loads(
        asyncio.run(
            kit.invoke(
                "calculate_commute",
                {"origin_address": "A", "destination_address": "B"},
                interrupted=lambda: True,
            )
        )
    )
    assert out["status"] == "not_executed"
    assert "call this tool again" in out["instruction"]
    assert calls == [] and log_lines(tmp_path) == []


def test_challenged_call_does_not_execute_until_confirmed(tmp_path):
    calls = []
    kit, tl = make(tmp_path, [commute_spec(calls)])
    tl.on_transcript("commute from APT1 to Harbor, sorry, to Downtown", is_final=True)
    args = {"origin_address": "APT1", "destination_address": "Harbor"}
    first = json.loads(asyncio.run(kit.invoke("calculate_commute", args)))
    assert first["status"] == "not_executed" and "Harbor" in first["reason"]
    assert calls == []
    asyncio.run(kit.invoke("calculate_commute", args))  # model re-confirms the same value
    assert len(calls) == 1


def test_duplicate_call_returns_cached_result_and_is_logged_once(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    args = {"origin_address": "APT1", "destination_address": "Downtown"}
    asyncio.run(kit.invoke("calculate_commute", args))
    second = json.loads(asyncio.run(kit.invoke("calculate_commute", dict(args, mode="driving"))))
    assert second["status"] == "already_done"
    assert second["result"] == {"status": "success", "duration_mins": 25}
    assert len(calls) == 1 and len(log_lines(tmp_path)) == 1


def test_unknown_tool_is_reported_not_raised(tmp_path):
    kit, _ = make(tmp_path, [commute_spec([])])
    out = json.loads(asyncio.run(kit.invoke("nope", {})))
    assert out["status"] == "not_executed"


def test_audit_log_records_every_decision(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    asyncio.run(kit.invoke("calculate_commute", {"origin_address": "A", "destination_address": "B"}))
    asyncio.run(kit.invoke("calculate_commute", {"origin_address": "A", "destination_address": "B"}))
    audit = [json.loads(line) for line in (tmp_path / "audit.log").read_text().splitlines()]
    assert [a["outcome"] for a in audit] == ["executed", "duplicate"]
    assert all(a["room"] == "eval-abc" for a in audit)


def test_logged_args_can_differ_from_backend_args(tmp_path):
    """A backend shim may add internal defaults; the log records what the agent asked for."""
    seen = []

    async def handler(args):
        seen.append(args)
        return {"ok": True}

    spec = ToolSpec(
        name="search_apartments",
        description="d",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        mutating=False,
        handler=handler,
    )
    kit, _ = make(tmp_path, [spec])
    asyncio.run(kit.invoke("search_apartments", {"city": "Austin"}))
    assert log_lines(tmp_path)[0]["call"]["args"] == {"city": "Austin"}


def test_on_executed_callback_fires_after_real_execution_only(tmp_path):
    calls, fired = [], []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    kit.on_executed = lambda name, args, result: fired.append(name)
    asyncio.run(kit.invoke("calculate_commute", {"origin_address": "A", "destination_address": "B"}))
    asyncio.run(kit.invoke("calculate_commute", {"origin_address": "A", "destination_address": "B"}))
    assert fired == ["calculate_commute"]


class FakeHandle:
    interrupted = False


class FakeCtx(RunContext):
    def __init__(self):
        self._h = FakeHandle()

    @property
    def speech_handle(self):
        return self._h


def test_livekit_tools_bind_and_route_through_the_toolkit(tmp_path):
    calls = []
    kit, _ = make(tmp_path, [commute_spec(calls)])
    (tool,) = kit.livekit_tools()
    assert isinstance(tool, RawFunctionTool)
    args, kwargs = prepare_function_arguments(
        fnc=tool,
        json_arguments=json.dumps({"origin_address": "A", "destination_address": "B"}),
        call_ctx=FakeCtx(),
    )
    out = asyncio.run(tool(*args, **kwargs))
    assert json.loads(out)["status"] == "success"
    assert len(calls) == 1


def test_livekit_tool_schema_matches_spec(tmp_path):
    kit, _ = make(tmp_path, [commute_spec([])])
    (tool,) = kit.livekit_tools()
    from livekit.agents.llm.tool_context import get_raw_function_info

    schema = get_raw_function_info(tool).raw_schema
    assert schema["name"] == "calculate_commute"
    assert schema["parameters"]["required"] == ["origin_address", "destination_address"]
