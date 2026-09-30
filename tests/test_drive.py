import asyncio
import json

from extension.drive_tools import drive_tool_specs
from extension.nav_backend import NavBackend
from holdfast.gate import CommitGate, GatePolicy
from holdfast.ledger import ActionLedger
from holdfast.timeline import ConversationTimeline
from holdfast.toolkit import GatedToolkit

POLICY = GatePolicy(settle_read_s=0.05, settle_write_s=0.1, min_resume_s=0.05, poll_s=0.005, transcript_wait_s=0.0)


def make(route_delay_s=0.0):
    nav = NavBackend(route_delay_s=route_delay_s, search_delay_s=0.0)
    tl = ConversationTimeline()
    kit = GatedToolkit(drive_tool_specs(nav), CommitGate(tl, POLICY), ActionLedger(), room_name="car", call_log_path=None)
    return nav, tl, kit


def call(kit, name, args):
    return json.loads(asyncio.run(kit.invoke(name, args)))


def test_setting_a_destination_creates_one_active_route():
    nav, _, kit = make()
    out = call(kit, "set_destination", {"destination": "the airport"})
    assert out["status"] == "success" and out["destination"] == "City Airport"
    assert nav.active_route["destination"] == "City Airport"


def test_changing_destination_supersedes_instead_of_stacking_routes():
    nav, _, kit = make()
    call(kit, "set_destination", {"destination": "airport"})
    out = call(kit, "set_destination", {"destination": "Central Station"})
    assert out["replaced"] == "City Airport"
    assert nav.active_route["destination"] == "Central Station"
    assert [r["destination"] for r in nav.superseded] == ["City Airport"]


def test_same_destination_twice_is_not_a_new_route():
    nav, _, kit = make()
    call(kit, "set_destination", {"destination": "airport"})
    call(kit, "set_destination", {"destination": "Airport"})  # ledger: identical after normalization
    assert nav.route_count == 1


def test_eta_message_is_sent_once_per_route_and_updated_after_a_reroute():
    nav, _, kit = make()
    call(kit, "set_destination", {"destination": "airport"})
    first = call(kit, "send_eta", {"contact": "Maya"})
    again = nav_send = asyncio.run(nav.send_eta("Maya"))  # backend-level repeat, bypassing the ledger
    assert first["status"] == "sent" and again["status"] == "already_sent"
    call(kit, "set_destination", {"destination": "harbor market"})
    upd = call(kit, "send_eta", {"contact": "Maya"})
    assert upd["status"] == "sent" and upd["kind"] == "update"
    assert [m["kind"] for m in nav.messages] == ["initial", "update"]


def test_unknown_destination_changes_nothing():
    nav, _, kit = make()
    out = call(kit, "set_destination", {"destination": "Atlantis"})
    assert out["status"] == "error" and "known_places" in out
    assert nav.active_route is None


def test_find_nearby_then_add_stop_chains_on_the_returned_id():
    nav, _, kit = make()
    call(kit, "set_destination", {"destination": "tech park"})
    found = call(kit, "find_nearby", {"category": "charging"})
    pid = found["results"][0]["id"]
    out = call(kit, "add_stop", {"place_id": pid})
    assert out["status"] == "success"
    assert nav.active_route["stops"] == [pid]
    assert out["eta_min"] > found["current_eta_min"]


def test_mid_utterance_destination_change_sets_only_the_final_destination():
    nav, tl, kit = make(route_delay_s=0.01)

    async def scenario():
        tl.on_user_state("speaking")  # "take me to the airport... actually, Central Station"
        early = asyncio.create_task(kit.invoke("set_destination", {"destination": "airport"}))
        await asyncio.sleep(0.2)
        tl.on_user_state("listening")
        tl.on_transcript("take me to the airport actually central station", is_final=True)
        first = json.loads(await early)
        final = json.loads(await kit.invoke("set_destination", {"destination": "Central Station"}))
        return first, final

    first, final = asyncio.run(scenario())
    assert first["status"] == "not_executed"
    assert final["status"] == "success"
    assert nav.route_count == 1 and nav.superseded == []


def test_state_changes_notify_the_dashboard():
    events = []
    nav = NavBackend(route_delay_s=0.0, on_change=lambda snap: events.append(snap["last_event"]))
    asyncio.run(nav.set_destination("airport"))
    asyncio.run(nav.cancel_navigation())
    assert [e.split(":")[0] for e in events] == ["route", "cancel"]
