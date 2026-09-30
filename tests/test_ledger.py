import asyncio

import pytest

from holdfast.ledger import ActionLedger, canonical_key


def test_canonical_key_ignores_case_whitespace_key_order_and_float_form():
    a = canonical_key("get_exchange_rate", {"amount": 500.0, "to_currency": " eur ", "from_currency": "USD"})
    b = canonical_key("get_exchange_rate", {"from_currency": "usd", "to_currency": "EUR", "amount": 500})
    assert a == b


def test_canonical_key_distinguishes_different_values_and_tools():
    assert canonical_key("track_order", {"order_id": "A1"}) != canonical_key("track_order", {"order_id": "A2"})
    assert canonical_key("track_order", {"order_id": "A1"}) != canonical_key("cancel", {"order_id": "A1"})


def test_second_identical_call_returns_cached_result_without_executing():
    ledger = ActionLedger()
    calls = []

    async def fn():
        calls.append(1)
        return {"ok": True}

    async def go():
        r1 = await ledger.run("book_flight", {"passenger_name": "Dana"}, fn)
        r2 = await ledger.run("book_flight", {"passenger_name": " dana "}, fn)
        return r1, r2

    (res1, st1), (res2, st2) = asyncio.run(go())
    assert calls == [1]
    assert st1 == "executed" and st2 == "duplicate"
    assert res1 == res2 == {"ok": True}


def test_concurrent_identical_calls_join_the_inflight_execution():
    ledger = ActionLedger()
    calls = []

    async def fn():
        calls.append(1)
        await asyncio.sleep(0.01)
        return "done"

    async def go():
        return await asyncio.gather(
            ledger.run("add_to_cart", {"product_id": "P1", "quantity": 1}, fn),
            ledger.run("add_to_cart", {"product_id": "P1", "quantity": 1}, fn),
        )

    results = asyncio.run(go())
    assert calls == [1]
    assert sorted(s for _, s in results) == ["executed", "joined"]


def test_failed_execution_is_not_recorded_so_a_retry_runs_again():
    ledger = ActionLedger()
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("backend down")
        return "ok"

    async def go():
        with pytest.raises(RuntimeError):
            await ledger.run("track_order", {"order_id": "A1"}, flaky)
        return await ledger.run("track_order", {"order_id": "A1"}, flaky)

    res, status = asyncio.run(go())
    assert (res, status) == ("ok", "executed")
    assert len(attempts) == 2


def test_slot_tracking_reports_superseded_action():
    ledger = ActionLedger()
    assert ledger.claim_slot("route", "k1") is None
    assert ledger.claim_slot("route", "k2") == "k1"
    assert ledger.slot_owner("route") == "k2"


def test_history_records_every_outcome_in_order():
    ledger = ActionLedger()

    async def fn():
        return 1

    async def go():
        await ledger.run("t", {"a": 1}, fn)
        await ledger.run("t", {"a": 1}, fn)

    asyncio.run(go())
    assert [h["status"] for h in ledger.history] == ["executed", "duplicate"]
