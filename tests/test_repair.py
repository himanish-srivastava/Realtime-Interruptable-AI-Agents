from holdfast.repair import find_reparanda, tokenize


def test_tokenize_normalizes_case_and_thousands_separators():
    assert tokenize("Send $1,000 to PARIS!") == ["send", "1000", "to", "paris"]


def test_flags_value_immediately_followed_by_correction():
    text = "Find me a flight to Lisbon, no wait, Porto on the ninth"
    hits = find_reparanda(text, {"destination": "Lisbon", "date": "2026-05-09"})
    assert [h.arg for h in hits] == ["destination"]
    assert hits[0].marker == "no wait"


def test_fillers_between_value_and_marker_are_ignored():
    text = "track order KX77, uh, um, actually order KX78"
    hits = find_reparanda(text, {"order_id": "KX77"})
    assert len(hits) == 1


def test_corrected_value_is_not_flagged():
    text = "track order KX77, uh, actually order KX78"
    assert find_reparanda(text, {"order_id": "KX78"}) == []


def test_value_restated_after_marker_is_not_flagged():
    text = "to Lisbon, actually yes Lisbon is right"
    assert find_reparanda(text, {"destination": "Lisbon"}) == []


def test_no_marker_means_no_flag():
    text = "convert 250 dollars to euros please"
    assert find_reparanda(text, {"amount": 250.0, "to_currency": "EUR"}) == []


def test_instead_of_is_comparative_not_a_repair():
    text = "search Oslo instead of Bergen"
    assert find_reparanda(text, {"destination": "Oslo"}) == []


def test_rather_than_is_comparative_not_a_repair():
    text = "use savings rather than checking"
    assert find_reparanda(text, {"source_account": "savings"}) == []


def test_marker_far_from_value_is_not_attributed_to_it():
    text = "flights to Lisbon on the fifth of may for two people actually the sixth"
    assert find_reparanda(text, {"destination": "Lisbon"}) == []


def test_integral_float_matches_integer_in_transcript():
    text = "exchange 1,200 sorry 1,500 dollars"
    hits = find_reparanda(text, {"amount": 1200.0})
    assert [h.arg for h in hits] == ["amount"]


def test_marker_with_nothing_after_it_is_not_a_repair():
    text = "book it for Dana, sorry"
    assert find_reparanda(text, {"passenger_name": "Dana"}) == []


def test_booleans_and_empty_values_are_skipped():
    assert find_reparanda("pets yes, no wait, no pets", {"pets_allowed": True, "x": ""}) == []


def test_marker_is_attributed_only_to_the_nearest_preceding_value():
    text = "commute from APT1 to Harbor, sorry, to Downtown"
    hits = find_reparanda(text, {"origin_address": "APT1", "destination_address": "Harbor"})
    assert [h.arg for h in hits] == ["destination_address"]


def test_corrected_call_is_not_rechallenged_on_an_unrelated_earlier_value():
    # regression: found by scripts/harness_selftest.py
    text = "move my utilities autopay to checking sorry to savings"
    assert find_reparanda(text, {"bill_type": "utilities", "source_account": "savings"}) == []
    hits = find_reparanda(text, {"bill_type": "utilities", "source_account": "checking"})
    assert [h.arg for h in hits] == ["source_account"]


def test_please_between_value_and_marker_is_skipped():
    hits = find_reparanda("book it for Dana Kim please, sorry, Dana Lee", {"passenger_name": "Dana Kim"})
    assert [h.arg for h in hits] == ["passenger_name"]
