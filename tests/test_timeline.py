import math

from holdfast.timeline import ConversationTimeline


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def make():
    clock = Clock()
    return ConversationTimeline(clock=clock), clock


def test_silence_is_infinite_before_any_speech():
    tl, _ = make()
    assert math.isinf(tl.silence_duration())


def test_silence_duration_counts_from_last_speech_end():
    tl, clock = make()
    tl.on_user_state("speaking")
    clock.t = 102.0
    tl.on_user_state("listening")
    clock.t = 102.75
    assert tl.silence_duration() == 0.75


def test_silence_is_zero_while_user_speaks():
    tl, clock = make()
    tl.on_user_state("speaking")
    clock.t = 105.0
    assert tl.silence_duration() == 0.0


def test_speech_after_clips_segments_to_the_reference_time():
    tl, clock = make()
    tl.on_user_state("speaking")          # 100
    clock.t = 103.0
    tl.on_user_state("listening")         # segment 100..103
    clock.t = 104.0
    tl.on_user_state("speaking")          # 104..(ongoing)
    clock.t = 104.5
    assert tl.speech_after(102.0) == 1.0 + 0.5


def test_repeated_state_events_do_not_open_duplicate_segments():
    tl, clock = make()
    tl.on_user_state("speaking")
    clock.t = 101.0
    tl.on_user_state("speaking")
    clock.t = 102.0
    tl.on_user_state("listening")
    assert tl.speech_after(0.0) == 2.0


def test_away_state_closes_a_segment_like_listening():
    tl, clock = make()
    tl.on_user_state("speaking")
    clock.t = 101.0
    tl.on_user_state("away")
    assert tl.user_speaking is False
    assert tl.speech_after(0.0) == 1.0


def test_user_text_joins_finals_and_latest_interim():
    tl, _ = make()
    tl.on_transcript("book a flight", is_final=True)
    tl.on_transcript("to", is_final=False)
    tl.on_transcript("to Porto", is_final=False)
    assert tl.user_text() == "book a flight to Porto"
    tl.on_transcript("to Porto please", is_final=True)
    assert tl.user_text() == "book a flight to Porto please"


def test_blank_final_transcripts_are_ignored():
    tl, _ = make()
    tl.on_transcript("   ", is_final=True)
    assert tl.user_text() == ""


def test_transcript_pending_until_a_final_arrives_after_speech_end():
    tl, clock = make()
    tl.on_user_state("speaking")
    clock.t = 101.0
    tl.on_user_state("listening")
    assert tl.transcript_pending() is True
    clock.t = 101.4
    tl.on_transcript("hello there", is_final=True)
    assert tl.transcript_pending() is False


def test_agent_speech_and_tool_activity_are_recorded():
    tl, clock = make()
    tl.on_agent_state("speaking")
    clock.t = 101.0
    tl.on_tool_activity()
    assert tl.agent_spoke_after(99.0) is True
    assert tl.agent_spoke_after(100.5) is False
    assert tl.tool_activity_after(100.5) is True
