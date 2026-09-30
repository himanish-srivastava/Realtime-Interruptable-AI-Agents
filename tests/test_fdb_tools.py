import asyncio
import json

import pytest

from holdfast.fdb_tools import FDB_TOOL_NAMES, fdb_tool_specs
from holdfast.gate import CommitGate, GatePolicy
from holdfast.ledger import ActionLedger
from holdfast.timeline import ConversationTimeline
from holdfast.toolkit import GatedToolkit

FAST = GatePolicy(settle_read_s=0.0, settle_write_s=0.0, transcript_wait_s=0.0, poll_s=0.001)
MUTATING = {"book_flight", "update_identity_doc", "modify_autopay", "update_search_filter", "add_to_cart"}


def kit(tmp_path, registry, schema):
    gate = CommitGate(ConversationTimeline(), FAST)
    return GatedToolkit(fdb_tool_specs(registry, schema=schema), gate, ActionLedger(),
                        room_name="r", call_log_path=tmp_path / "c.log")


def logged(tmp_path):
    return [json.loads(l)["call"] for l in (tmp_path / "c.log").read_text().splitlines()]


@pytest.mark.parametrize("schema", ["template", "extended"])
def test_exposes_exactly_the_twelve_benchmark_tools(fdb_registry, schema):
    specs = fdb_tool_specs(fdb_registry, schema=schema)
    assert sorted(s.name for s in specs) == sorted(FDB_TOOL_NAMES)
    assert sorted(FDB_TOOL_NAMES) == sorted(fdb_registry.FUNCTIONS)
    assert {s.name for s in specs if s.mutating} == MUTATING


def test_template_schema_matches_reference_agent_requiredness(fdb_registry):
    specs = {s.name: s for s in fdb_tool_specs(fdb_registry, schema="template")}
    assert specs["search_apartments"].parameters["required"] == ["city", "bedrooms", "max_price"]
    assert "pets_allowed" not in specs["search_apartments"].parameters["properties"]
    assert "category" not in specs["search_products"].parameters["properties"]


def test_extended_schema_never_forces_invented_values(fdb_registry):
    specs = {s.name: s for s in fdb_tool_specs(fdb_registry, schema="extended")}
    apt = specs["search_apartments"].parameters
    assert apt["required"] == ["city"]
    assert {"bedrooms", "max_price", "pets_allowed"} <= set(apt["properties"])
    assert "category" in specs["search_products"].parameters["properties"]


def test_unknown_schema_mode_is_rejected(fdb_registry):
    with pytest.raises(ValueError):
        fdb_tool_specs(fdb_registry, schema="bogus")


def test_apartment_search_with_only_a_city_runs_on_the_real_mock(tmp_path, fdb_registry):
    k = kit(tmp_path, fdb_registry, "extended")
    out = json.loads(asyncio.run(k.invoke("search_apartments", {"city": "Denver", "pets_allowed": True})))
    assert out["status"] == "success" and out["city"] == "Denver"
    assert logged(tmp_path) == [{**logged(tmp_path)[0], "args": {"city": "Denver", "pets_allowed": True}}]


def test_defaults_are_logged_like_the_reference_agent(tmp_path, fdb_registry):
    k = kit(tmp_path, fdb_registry, "extended")
    asyncio.run(k.invoke("calculate_commute", {"origin_address": "APT1", "destination_address": "Midtown"}))
    asyncio.run(k.invoke("add_to_cart", {"product_id": "PROD1"}))
    args = [c["args"] for c in logged(tmp_path)]
    assert args[0]["mode"] == "driving"
    assert args[1] == {"product_id": "PROD1", "quantity": 1}


def test_every_tool_executes_against_the_real_mock(tmp_path, fdb_registry):
    k = kit(tmp_path, fdb_registry, "extended")
    samples = {
        "search_flights": {"destination": "Oslo", "date": "2026-07-04"},
        "book_flight": {"passenger_name": "Dana Kim"},
        "update_identity_doc": {"doc_type": "passport", "doc_number": "X1234567"},
        "get_card_benefits": {"card_type": "gold"},
        "get_exchange_rate": {"amount": 300, "from_currency": "USD", "to_currency": "JPY"},
        "modify_autopay": {"bill_type": "utilities", "source_account": "savings"},
        "search_apartments": {"city": "Denver", "bedrooms": 2, "max_price": 2500},
        "calculate_commute": {"origin_address": "APT1", "destination_address": "Midtown", "mode": "transit"},
        "update_search_filter": {"filter_name": "max_price", "value": "2200"},
        "track_order": {"order_id": "ZX900"},
        "search_products": {"query": "desk lamp", "max_price": 40},
        "add_to_cart": {"product_id": "PROD1", "quantity": 2},
    }
    for name, a in samples.items():
        out = json.loads(asyncio.run(k.invoke(name, a)))
        assert out["status"] == "success", (name, out)
    assert [c["function"] for c in logged(tmp_path)] == list(samples)
