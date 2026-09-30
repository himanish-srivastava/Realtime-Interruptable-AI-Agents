#!/usr/bin/env python3
"""Holdfast agent for FDB-v3: drop-in replacement for the reference
``lk_agent_tool.py`` (same room dispatch, same telemetry files).

    python agents/fdb_agent.py start            # what reproduce.sh runs
    python agents/fdb_agent.py dev              # hot-reload development mode
    python agents/fdb_agent.py console          # talk to it locally (mic/speaker)

Declared model provider: OpenAI Realtime API (``config/holdfast.yaml`` model.name).
Needs LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET and OPENAI_API_KEY.
"""

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

FDB_V3_DIR = Path(os.getenv("FDB_V3_DIR", ROOT / "third_party" / "Full-Duplex-Bench" / "v3")).resolve()
sys.path.insert(0, str(FDB_V3_DIR))
load_dotenv(ROOT / ".env")
load_dotenv(FDB_V3_DIR / ".env.local")

from holdfast.config import load_config  # noqa: E402

CFG = load_config()

# Same optional CLI flag as the reference agent; stripped before LiveKit parses argv.
LATENCY_PROFILE = CFG["tools"].get("latency_profile", "instant")
if "--latency" in sys.argv:
    i = sys.argv.index("--latency")
    if i + 1 < len(sys.argv):
        LATENCY_PROFILE = sys.argv[i + 1]
        del sys.argv[i : i + 2]

from livekit import agents  # noqa: E402
from livekit.agents import Agent, AgentServer, AgentSession  # noqa: E402

from holdfast.fdb_tools import fdb_tool_specs  # noqa: E402
from holdfast.prompts import build_instructions  # noqa: E402
from holdfast.runtime import HoldfastRuntime, build_realtime_model  # noqa: E402

try:
    from mock_apis import MockAPIRegistry  # the benchmark's own mock backend
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"mock_apis.py not found under FDB_V3_DIR={FDB_V3_DIR}: {e}")

logging.basicConfig(level=os.getenv("HOLDFAST_LOG_LEVEL", "INFO"))
log = logging.getLogger("holdfast.fdb_agent")

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    room = ctx.room.name
    with open(CFG["logs"]["heartbeat_log"], "a") as f:
        f.write(f"!!! AGENT JOINING ROOM: {room} at {time.ctime()} !!!\n")
    log.info("joining room %s (schema=%s, latency=%s)", room, CFG["tools"]["schema"], LATENCY_PROFILE)

    # fresh backend + control layer per conversation: nothing is cached across scenarios
    registry = MockAPIRegistry(latency_profile=LATENCY_PROFILE)
    runtime = HoldfastRuntime(CFG, room_name=room, specs=fdb_tool_specs(registry, schema=CFG["tools"]["schema"]))

    session = AgentSession(llm=build_realtime_model(CFG), tools=runtime.toolkit.livekit_tools())
    runtime.attach(session)
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=build_instructions(acknowledge=bool(CFG["prompt"].get("acknowledge")))),
    )
    log.info("agent listening in %s", room)


if __name__ == "__main__":
    agents.cli.run_app(server)
