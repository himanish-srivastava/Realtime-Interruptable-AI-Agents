#!/usr/bin/env python3
"""Extension use case: hands-free in-car assistant (Holdfast Drive).

Same control layer as the benchmark agent (commit gate, idempotency ledger,
reply watchdog), different tools: navigation, places along the route, ETA texts.

    python agents/drive_agent.py console   # local mic + speaker, only OPENAI_API_KEY needed
    python agents/drive_agent.py dev       # via LiveKit Cloud (e.g. the Agents Playground)

Things to try (see README "Extension"):
  "Take me to the airport... actually no, Central Station."
  "Text Maya my ETA."  then  "Change of plan, go to Harbor Market."  then  "Text Maya my ETA."
  "Find a charger on the way and add the closest one."
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from livekit import agents  # noqa: E402
from livekit.agents import Agent, AgentServer, AgentSession  # noqa: E402

from extension.dashboard import render  # noqa: E402
from extension.drive_tools import drive_tool_specs  # noqa: E402
from extension.nav_backend import NavBackend  # noqa: E402
from holdfast.config import load_config  # noqa: E402
from holdfast.prompts import build_instructions  # noqa: E402
from holdfast.runtime import HoldfastRuntime, build_realtime_model  # noqa: E402

DRIVE_RULES = """\
IN-CAR CONTEXT
You are the car's voice assistant and the user is driving. Keep every reply to one or two short \
sentences. Use set_destination with the final place the driver named; a new destination replaces \
the current route (never keep two). Text a contact only when asked. For "find X on the way", call \
find_nearby, and call add_stop with the returned id only if the driver wants to stop there. \
Known places: City Airport, Central Station, Harbor Market, Tech Park, City Hospital, Home."""

logging.basicConfig(level=os.getenv("HOLDFAST_LOG_LEVEL", "WARNING"))
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    cfg = load_config()
    logs_dir = ROOT / "results" / "drive_logs"
    cfg["logs"] = {"call_log": str(logs_dir / "calls.jsonl"), "heartbeat_log": None,
                   "audit_log": str(logs_dir / "audit.jsonl")}
    nav = NavBackend(route_delay_s=float(os.getenv("DRIVE_ROUTE_DELAY_S", "2.0")), on_change=render)
    runtime = HoldfastRuntime(cfg, room_name=ctx.room.name, specs=drive_tool_specs(nav))
    session = AgentSession(llm=build_realtime_model(cfg), tools=runtime.toolkit.livekit_tools())
    runtime.attach(session)
    await session.start(room=ctx.room, agent=Agent(instructions=build_instructions(acknowledge=True, extra=DRIVE_RULES)))
    render(nav.snapshot())


if __name__ == "__main__":
    agents.cli.run_app(server)
