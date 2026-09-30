import asyncio

from holdfast.gate import CommitGate, GatePolicy, Verdict
from holdfast.timeline import ConversationTimeline


class Sim:
    """Deterministic clock + sleep. Scheduled callbacks fire as time advances."""

    def __init__(self, t=100.0):
        self.t = t
        self.events = []

    def now(self):
        return self.t

    def at(self, when, fn):
        self.events.append((when, fn))
        self.events.sort(key=lambda e: e[0])

    async def sleep(self, dt):
        target = self.t + dt
        while self.events and self.events[0][0] <= target:
            when, fn = self.events.pop(0)
            self.t = max(self.t, when)
            fn()
        self.t = target


def setup(policy=None, *, user_ended_at=None):
    sim = Sim()
    tl = ConversationTimeline(clock=sim.now)
    if user_ended_at is not None:
        # user spoke from 95 until user_ended_at
        sim.t = 95.0
        tl.on_user_state("speaking")
        sim.t = user_ended_at
        tl.on_user_state("listening")
        tl.on_transcript("please do the thing", is_final=True)
        sim.t = 100.0
    gate = CommitGate(tl, policy or GatePolicy(), clock=sim.now, sleep=sim.sleep)
    return sim, tl, gate


def run(coro):
    return asyncio.run(coro)


def never():
    return False


def test_executes_after_read_settle_window_when_user_is_silent():
    sim, _, gate = setup(GatePolicy(settle_read_s=0.4), user_ended_at=99.9)
    d = run(gate.admit("search_flights", {"destination": "Porto"}, mutating=False, interrupted=never))
    assert d.verdict is Verdict.EXECUTE
    assert 100.29 <= sim.t <= 100.36  # waited only until 0.4s of silence


def test_mutating_calls_wait_longer_than_read_calls():
    sim, _, gate = setup(GatePolicy(settle_read_s=0.3, settle_write_s=1.0), user_ended_at=99.9)
    d = run(gate.admit("book_flight", {"passenger_name": "Dana"}, mutating=True, interrupted=never))
    assert d.verdict is Verdict.EXECUTE
    assert sim.t >= 100.9


def test_stale_when_user_resumes_speaking_during_settle_window():
    sim, tl, gate = setup(GatePolicy(settle_write_s=1.0, min_resume_s=0.3), user_ended_at=99.9)

    def resume():
        tl.on_user_state("speaking")

    def stop():
        tl.on_user_state("listening")

    sim.at(100.4, resume)
    sim.at(101.4, stop)
    d = run(gate.admit("modify_autopay", {"bill_type": "utilities"}, mutating=True, interrupted=never))
    assert d.verdict is Verdict.STALE
    assert "kept speaking" in d.reason


def test_short_noise_blip_does_not_make_call_stale():
    sim, tl, gate = setup(GatePolicy(settle_read_s=0.5, min_resume_s=0.3), user_ended_at=99.9)
    sim.at(100.2, lambda: tl.on_user_state("speaking"))
    sim.at(100.3, lambda: tl.on_user_state("listening"))
    d = run(gate.admit("track_order", {"order_id": "A1"}, mutating=False, interrupted=never))
    assert d.verdict is Verdict.EXECUTE


def test_call_issued_mid_utterance_is_stale_once_user_finishes():
    sim, tl, gate = setup(GatePolicy(min_resume_s=0.3))
    tl.on_user_state("speaking")  # user already talking at t=100 when the call arrives
    sim.at(101.5, lambda: tl.on_user_state("listening"))
    d = run(gate.admit("search_flights", {"destination": "Lisbon"}, mutating=False, interrupted=never))
    assert d.verdict is Verdict.STALE


def test_interrupted_speech_handle_makes_call_stale_immediately():
    sim, _, gate = setup(user_ended_at=99.9)
    d = run(gate.admit("search_flights", {"destination": "Porto"}, mutating=False, interrupted=lambda: True))
    assert d.verdict is Verdict.STALE
    assert sim.t == 100.0


def test_repeated_stale_verdicts_are_capped_so_noise_cannot_starve_execution():
    policy = GatePolicy(settle_read_s=0.2, min_resume_s=0.1, max_stale_per_tool=2)
    sim, tl, gate = setup(policy, user_ended_at=99.9)
    verdicts = []
    for _ in range(3):
        start = sim.t
        sim.at(start + 0.05, lambda: tl.on_user_state("speaking"))
        sim.at(start + 0.5, lambda: tl.on_user_state("listening"))
        d = run(gate.admit("track_order", {"order_id": "A1"}, mutating=False, interrupted=never))
        verdicts.append(d.verdict)
    assert verdicts == [Verdict.STALE, Verdict.STALE, Verdict.EXECUTE]


def test_gives_up_waiting_after_max_wait_and_executes():
    sim, tl, gate = setup(GatePolicy(max_wait_s=3.0), user_ended_at=None)
    tl.on_user_state("speaking")  # e.g. VAD latched on background noise before the call
    sim.t = 100.0
    # the call arrived while "speaking"; the segment never closes
    d = run(gate.admit("track_order", {"order_id": "A1"}, mutating=False, interrupted=never))
    assert d.verdict in (Verdict.STALE, Verdict.EXECUTE)
    assert sim.t <= 103.1


def test_challenges_a_value_the_user_appears_to_have_corrected_once():
    sim, tl, gate = setup(GatePolicy(settle_read_s=0.1), user_ended_at=99.9)
    tl.on_transcript("flights to Lisbon, no wait, Porto", is_final=True)
    args = {"destination": "Lisbon", "date": "2026-05-09"}
    d1 = run(gate.admit("search_flights", args, mutating=False, interrupted=never))
    assert d1.verdict is Verdict.CHALLENGE
    assert "Lisbon" in d1.reason
    d2 = run(gate.admit("search_flights", args, mutating=False, interrupted=never))
    assert d2.verdict is Verdict.EXECUTE


def test_repair_check_can_be_disabled():
    sim, tl, gate = setup(GatePolicy(settle_read_s=0.1, repair_check=False), user_ended_at=99.9)
    tl.on_transcript("flights to Lisbon, no wait, Porto", is_final=True)
    d = run(gate.admit("search_flights", {"destination": "Lisbon"}, mutating=False, interrupted=never))
    assert d.verdict is Verdict.EXECUTE


def test_mutating_call_waits_briefly_for_a_pending_transcript():
    sim, tl, gate = setup(GatePolicy(settle_write_s=0.2, transcript_wait_s=1.0))
    sim.t = 99.0
    tl.on_user_state("speaking")
    sim.t = 99.9
    tl.on_user_state("listening")  # no final transcript yet
    sim.t = 100.0
    sim.at(100.5, lambda: tl.on_transcript("set autopay from checking, sorry, savings", is_final=True))
    d = run(gate.admit("modify_autopay", {"source_account": "checking"}, mutating=True, interrupted=never))
    assert d.verdict is Verdict.CHALLENGE
    assert sim.t >= 100.5
