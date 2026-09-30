"""Gated tools: every tool call goes through the commit gate and the ledger.

The toolkit turns plain ``ToolSpec`` definitions into LiveKit raw-schema
function tools. For each call it:

1. cleans the model's arguments against the JSON schema (drops empty and
   unknown keys, coerces "30" -> 30 / "true" -> True, fills declared defaults),
2. asks the commit gate whether the call may run now,
3. runs it through the idempotency ledger (so an action never runs twice),
4. writes the executed call to the benchmark's telemetry log in exactly the
   format the FDB-v3 harness parses, and a richer audit line for debugging.

Only calls that actually executed are written to the telemetry log, which is
what the harness scores. Calls the gate held back never touched the backend.

NOTE: no ``from __future__ import annotations`` here: LiveKit inspects the
``context: RunContext`` annotation of the generated tool at runtime.
"""

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from livekit.agents import RunContext, function_tool

from .gate import CommitGate, Verdict
from .ledger import ActionLedger

log = logging.getLogger("holdfast.toolkit")

DEFAULT_CALL_LOG = Path("/tmp/agent_tool_calls.log")  # read by FDB-v3 run_tool_benchmark.py

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema object
    mutating: bool
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    # Optional world-state folded into the idempotency key, for actions whose
    # meaning depends on state (e.g. "text my ETA" after a reroute is a new action).
    dedupe_context: Optional[Callable[[Dict[str, Any]], Any]] = None


def _coerce(value: Any, prop: Dict[str, Any]) -> Any:
    kind = prop.get("type")
    try:
        if kind == "string":
            return str(value).strip()
        if kind == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                value = float(value.replace(",", "").strip())
            f = float(value)
            return int(f) if f.is_integer() else value
        if kind == "number":
            if isinstance(value, str):
                return float(value.replace(",", "").strip())
            return float(value) if not isinstance(value, bool) else value
        if kind == "boolean":
            if isinstance(value, str):
                s = value.strip().lower()
                if s in _TRUE:
                    return True
                if s in _FALSE:
                    return False
            return value
    except (TypeError, ValueError):
        return value
    return value


def clean_args(spec: ToolSpec, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    props = spec.parameters.get("properties", {})
    out: Dict[str, Any] = {}
    for key, prop in props.items():
        v = raw.get(key)
        if v is not None:
            v = _coerce(v, prop)
        if v is None or (isinstance(v, str) and v == ""):
            if "default" in prop:
                out[key] = prop["default"]
            continue
        out[key] = v
    missing = [k for k in spec.parameters.get("required", []) if k not in out]
    return out, missing


def model_schema(spec: ToolSpec) -> Dict[str, Any]:
    """Schema sent to the model: defaults moved into descriptions."""
    params = copy.deepcopy(spec.parameters)
    for prop in params.get("properties", {}).values():
        if "default" in prop:
            d = prop.pop("default")
            prop["description"] = (prop.get("description", "") + f" Defaults to {d!r} if not stated.").strip()
    params.setdefault("additionalProperties", False)
    return {"name": spec.name, "description": spec.description, "parameters": params}


def _msg(status: str, reason: str, instruction: str = "", **extra: Any) -> str:
    body: Dict[str, Any] = {"status": status, "reason": reason}
    if instruction:
        body["instruction"] = instruction
    body.update(extra)
    return json.dumps(body, default=str)


STALE_INSTRUCTION = (
    "Not executed, nothing happened. Do not mention this to the user. Wait until the user "
    "has finished their complete request, then call this tool again with the FINAL values "
    "they stated (apply any corrections they made)."
)
CHALLENGE_INSTRUCTION = (
    "Not executed yet. Re-check the user's exact words. If they corrected this value, call "
    "this tool again with the corrected value. If the value is right, call this tool again "
    "with exactly the same arguments to confirm. Do not mention this check to the user."
)
DUPLICATE_INSTRUCTION = (
    "This exact action already completed earlier in this conversation. Do not repeat it; "
    "use this result."
)


class GatedToolkit:
    def __init__(
        self,
        specs: List[ToolSpec],
        gate: CommitGate,
        ledger: ActionLedger,
        *,
        room_name: str,
        call_log_path: Union[str, Path, None] = DEFAULT_CALL_LOG,
        audit_log_path: Union[str, Path, None] = None,
        clock: Callable[[], float] = time.time,
        on_executed: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
    ) -> None:
        self.specs = {s.name: s for s in specs}
        self.gate = gate
        self.ledger = ledger
        self.room_name = room_name
        self.call_log_path = Path(call_log_path) if call_log_path else None
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.clock = clock
        self.on_executed = on_executed
        self._lock = threading.Lock()

    # ── main entry ──────────────────────────────────────────────────
    async def invoke(
        self,
        name: str,
        raw_args: Dict[str, Any],
        interrupted: Callable[[], bool] = lambda: False,
    ) -> str:
        spec = self.specs.get(name)
        if spec is None:
            self._audit(name, raw_args, "unknown_tool", "")
            return _msg("not_executed", f"unknown tool {name!r}")

        self.gate.timeline.on_tool_activity()
        args, missing = clean_args(spec, raw_args or {})
        if missing:
            self._audit(name, args, "missing_args", ",".join(missing))
            return _msg(
                "not_executed",
                f"missing required argument(s): {', '.join(missing)}",
                "If the user stated these values, call again including them. Only ask the "
                "user if a value was never mentioned.",
            )

        decision = await self.gate.admit(name, args, mutating=spec.mutating, interrupted=interrupted)
        if decision.verdict is Verdict.STALE:
            self._audit(name, args, "stale", decision.reason, decision.waited_s)
            return _msg("not_executed", decision.reason, STALE_INSTRUCTION)
        if decision.verdict is Verdict.CHALLENGE:
            self._audit(name, args, "challenge", decision.reason, decision.waited_s)
            return _msg("not_executed", decision.reason, CHALLENGE_INSTRUCTION)

        async def execute() -> Any:
            t_start = self.clock()
            result = await spec.handler(dict(args))
            t_end = self.clock()
            self._write_call(name, args, t_start, t_end)
            return result

        try:
            key_args = args if spec.dedupe_context is None else {**args, "__context__": spec.dedupe_context(args)}
            result, status = await self.ledger.run(name, key_args, execute)
        except Exception as e:  # backend failure: report, do not crash the session
            log.exception("tool %s failed", name)
            self._audit(name, args, "error", repr(e), decision.waited_s)
            return _msg("error", f"{type(e).__name__}: {e}", "Tell the user briefly that this step failed.")

        self._audit(name, args, status, "", decision.waited_s)
        self.gate.timeline.on_tool_activity()
        if status == "executed":
            if self.on_executed:
                try:
                    self.on_executed(name, args, result)
                except Exception:
                    log.exception("on_executed callback failed")
            return json.dumps(result, default=str)
        return _msg("already_done", "identical action already completed", DUPLICATE_INSTRUCTION, result=result)

    # ── LiveKit adapter ─────────────────────────────────────────────
    def livekit_tools(self) -> list:
        return [self._make_tool(spec) for spec in self.specs.values()]

    def _make_tool(self, spec: ToolSpec):
        toolkit = self

        async def _tool(raw_arguments: dict, context: RunContext):
            handle = getattr(context, "speech_handle", None)

            def interrupted() -> bool:
                return bool(getattr(handle, "interrupted", False)) if handle is not None else False

            return await toolkit.invoke(spec.name, raw_arguments, interrupted=interrupted)

        _tool.__name__ = spec.name
        return function_tool(_tool, raw_schema=model_schema(spec))

    # ── logs ────────────────────────────────────────────────────────
    def _append(self, path: Optional[Path], obj: Dict[str, Any]) -> None:
        if path is None:
            return
        line = json.dumps(obj, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def _write_call(self, name: str, args: Dict[str, Any], t_start: float, t_end: float) -> None:
        self._append(
            self.call_log_path,
            {
                "room": self.room_name,
                "call": {"function": name, "args": args, "timestamp_start": t_start, "timestamp_end": t_end},
            },
        )

    def _audit(self, name: str, args: Any, outcome: str, reason: str, waited_s: float = 0.0) -> None:
        self._append(
            self.audit_log_path,
            {
                "t": self.clock(),
                "room": self.room_name,
                "tool": name,
                "args": args,
                "outcome": outcome,
                "reason": reason,
                "waited_s": round(waited_s, 3),
            },
        )
