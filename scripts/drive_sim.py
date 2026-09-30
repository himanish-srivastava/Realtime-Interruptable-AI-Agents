#!/usr/bin/env python3
"""Offline, scripted run of the in-car extension (no API keys). Replays a user
speech timeline and scripted model calls through the real Holdfast control
layer and nav backend, printing the dashboard. It checks the control logic;
the real conversational demo is `python agents/drive_agent.py console`."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extension.dashboard import render  # noqa: E402
from extension.drive_tools import drive_tool_specs  # noqa: E402
from extension.nav_backend import NavBackend  # noqa: E402
from holdfast.gate import CommitGate, GatePolicy  # noqa: E402
from holdfast.ledger import ActionLedger  # noqa: E402
from holdfast.timeline import ConversationTimeline  # noqa: E402
from holdfast.toolkit import GatedToolkit  # noqa: E402


async def main() -> int:
    nav = NavBackend(route_delay_s=0.5, search_delay_s=0.3, on_change=lambda s: render(s, sys.stdout))
    tl = ConversationTimeline()
    kit = GatedToolkit(drive_tool_specs(nav), CommitGate(tl, GatePolicy(settle_write_s=0.3, min_resume_s=0.1)),
                       ActionLedger(), room_name="drive-sim", call_log_path=None)

    async def say(text, seconds):
        print(f"\nUSER: {text}")
        tl.on_user_state("speaking")
        await asyncio.sleep(seconds)
        tl.on_user_state("listening")
        tl.on_transcript(text, is_final=True)

    async def model(name, args):
        out = json.loads(await kit.invoke(name, args))
        print(f"  tool {name}({args}) -> {out.get('status')}" + (f": {out.get('reason')}" if out.get("reason") else ""))
        return out

    # 1) destination corrected mid-utterance; the model fires early on "airport"
    early = asyncio.create_task(model("set_destination", {"destination": "airport"}))
    await say("Take me to the airport... actually, no, Central Station", 1.0)
    await early
    await model("set_destination", {"destination": "Central Station"})
    # 2) ETA text, a repeated request, then a reroute and an update
    await say("Text Maya my ETA", 0.4)
    await model("send_eta", {"contact": "Maya"})
    await model("send_eta", {"contact": "Maya"})  # model repeats itself
    await say("Change of plan, go to Harbor Market, and tell Maya", 0.8)
    await model("set_destination", {"destination": "Harbor Market"})
    await model("send_eta", {"contact": "Maya"})
    # 3) async search + chained stop
    await say("Find a charger on the way and add the closest one", 0.6)
    found = await model("find_nearby", {"category": "charging"})
    await model("add_stop", {"place_id": found["results"][0]["id"]})

    ok = (nav.route_count == 2 and [r["destination"] for r in nav.superseded] == ["Central Station"]
          and [m["kind"] for m in nav.messages] == ["initial", "update"] and nav.active_route["stops"] == ["CHG-1"])
    print("\nCHECK:", "OK - one route at a time, no duplicate texts, stale call never executed" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
