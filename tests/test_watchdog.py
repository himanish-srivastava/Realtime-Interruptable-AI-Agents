import asyncio

from holdfast.timeline import ConversationTimeline
from holdfast.watchdog import ReplyWatchdog
from tests.test_gate import Sim


def setup(**kw):
    sim = Sim()
    tl = ConversationTimeline(clock=sim.now)
    nudges = []
    wd = ReplyWatchdog(tl, nudges.append, clock=sim.now, sleep=sim.sleep, **kw)
    return sim, tl, wd, nudges


def test_nudges_once_when_tools_finish_and_agent_stays_silent():
    sim, tl, wd, nudges = setup(after_tools_s=2.0)
    tl.on_agent_state("listening")
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert len(nudges) == 1 and "result" in nudges[0].lower()


def test_no_nudge_when_agent_speaks_after_tools():
    sim, tl, wd, nudges = setup(after_tools_s=2.0)
    sim.at(101.0, lambda: tl.on_agent_state("speaking"))
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert nudges == []


def test_no_nudge_while_user_is_speaking():
    sim, tl, wd, nudges = setup(after_tools_s=2.0)
    sim.at(101.0, lambda: tl.on_user_state("speaking"))
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert nudges == []


def test_nudges_are_capped_per_conversation():
    sim, tl, wd, nudges = setup(after_tools_s=1.0, max_nudges=2)
    for _ in range(4):
        wd.tools_finished()
        asyncio.run(wd.drain())
    assert len(nudges) == 2


def test_no_response_nudge_after_a_real_user_turn_with_no_reaction():
    sim, tl, wd, nudges = setup(no_response_s=5.0)
    tl.on_user_state("speaking")
    sim.t = 102.0
    tl.on_user_state("listening")
    tl.on_transcript("check my order status please", is_final=True)
    wd.user_turn_ended()
    asyncio.run(wd.drain())
    assert len(nudges) == 1


def test_no_response_nudge_skipped_when_a_tool_started():
    sim, tl, wd, nudges = setup(no_response_s=5.0)
    tl.on_transcript("check my order", is_final=True)
    sim.at(101.0, tl.on_tool_activity)
    wd.user_turn_ended()
    asyncio.run(wd.drain())
    assert nudges == []


def test_no_response_nudge_skipped_for_noise_without_transcript():
    sim, tl, wd, nudges = setup(no_response_s=5.0)
    tl.on_agent_state("speaking")
    sim.t = 101.0
    tl.on_agent_state("listening")
    wd.user_turn_ended()  # e.g. VAD fired on background noise, nothing transcribed
    asyncio.run(wd.drain())
    assert nudges == []


def test_disabled_watchdog_never_nudges():
    sim, tl, wd, nudges = setup(after_tools_s=1.0, enabled=False)
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert nudges == []


def test_no_report_nudge_while_the_model_is_still_calling_tools():
    sim, tl, wd, nudges = setup(after_tools_s=2.0)
    sim.at(101.0, tl.on_tool_activity)  # e.g. a re-issued call after a challenge
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert nudges == []


def test_no_report_nudge_while_agent_is_thinking():
    sim, tl, wd, nudges = setup(after_tools_s=2.0)
    sim.at(101.0, lambda: tl.on_agent_state("thinking"))
    wd.tools_finished()
    asyncio.run(wd.drain())
    assert nudges == []
