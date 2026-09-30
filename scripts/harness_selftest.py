#!/usr/bin/env python3
"""Offline end-to-end check of the FDB-v3 wiring. No API keys, no LiveKit.

What is REAL here:
  * the benchmark's mock backend (mock_apis.py),
  * the harness's process_single(): telemetry-log parsing, latency
    measurement, result_<provider>.json writing,
  * the harness's scorers evaluate_pass_rate.py and evaluate_tool_calls.py
    (exact-match mode; the official run adds the gpt-4o judge),
  * the Holdfast toolkit, commit gate, ledger and timeline.

What is STOOD IN:
  * the LiveKit transport + realtime model: a scripted "model" issues tool
    calls while the user's speech timeline is replayed in real time,
  * the NeMo ASR model: returns fixed word timings.

The scripted model reproduces three failure modes documented in the FDB-v3
paper: a call issued before the user finished correcting it, a misunderstood
self-correction, and a repeated action. Each scenario runs twice:
  holdfast_selftest - through the Holdfast gate + ledger
  naive_selftest    - every call executes immediately (reference-agent behaviour)

This demonstrates that the control layer and the scoring pipeline fit
together. It does NOT measure how the real model behaves; only a real
benchmark run does that. Scenarios are invented, not taken from FDB-v3.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from holdfast.fdb_tools import fdb_tool_specs  # noqa: E402
from holdfast.gate import CommitGate, Decision, GatePolicy, Verdict  # noqa: E402
from holdfast.ledger import ActionLedger  # noqa: E402
from holdfast.timeline import ConversationTimeline  # noqa: E402
from holdfast.toolkit import GatedToolkit  # noqa: E402

CALL_LOG = Path("/tmp/agent_tool_calls.log")  # hardcoded in the harness
POLICY = GatePolicy(settle_read_s=0.15, settle_write_s=0.3, min_resume_s=0.1,
                    transcript_wait_s=0.3, poll_s=0.01)

SCENARIOS = [
    {
        "id": "selftest_preemptive",
        "title": "Order status with a mid-utterance correction",
        "domain": "ecommerce_support", "difficulty": "medium",
        "disfluency_features": ["SELF_CORRECTION", "FILLER"], "state_rollback_test": True,
        "user": "track order QX51 uh no wait QX15 and find a desk lamp under 40 dollars",
        "speech": (0.0, 2.4),
        "expected_tool_calls": [
            {"function": "track_order", "args": {"order_id": "QX15"}},
            {"function": "search_products", "args": {"query": "desk lamp", "max_price": 40}},
        ],
        "agent_says": "Order QX15 is out for delivery and I found a desk lamp for 30 dollars",
    },
    {
        "id": "selftest_miscomprehension",
        "title": "Autopay source corrected mid-sentence",
        "domain": "finance_billing", "difficulty": "easy",
        "disfluency_features": ["SELF_CORRECTION"], "state_rollback_test": True,
        "user": "move my utilities autopay to checking sorry to savings",
        "speech": (0.0, 2.0),
        "expected_tool_calls": [
            {"function": "modify_autopay", "args": {"bill_type": "utilities", "source_account": "savings"}},
        ],
        "agent_says": "Done your utilities autopay now comes from savings",
    },
    {
        "id": "selftest_duplicate",
        "title": "Add to cart must happen exactly once",
        "domain": "ecommerce_support", "difficulty": "medium",
        "disfluency_features": ["PAUSE"], "state_rollback_test": False,
        "user": "find blue mugs and add two to my cart",
        "speech": (0.0, 1.8),
        "expected_tool_calls": [
            {"function": "search_products", "args": {"query": "blue mugs"}},
            {"function": "add_to_cart", "args": {"product_id": "$RESULT_0.products[0].product_id", "quantity": 2}},
        ],
        "agent_says": "I added two blue mugs to your cart",
    },
]
for s in SCENARIOS:
    s["dialogue"] = [{"user": s["user"], "ai": s["agent_says"]}]
    s["num_expected_calls"] = len(s["expected_tool_calls"])


# ── scripted "model" behaviour per scenario ──────────────────────────────────
async def model_preemptive(kit, at):
    early = asyncio.create_task(_call_at(kit, at, 1.0, "track_order", {"order_id": "QX51"}))  # user still talking
    await _call_at(kit, at, 2.7, "track_order", {"order_id": "QX15"})
    await _call_at(kit, at, 2.8, "search_products", {"query": "desk lamp", "max_price": 40})
    await early


async def model_miscomprehension(kit, at):
    out = json.loads(await _call_at(kit, at, 2.4, "modify_autopay",
                                    {"bill_type": "utilities", "source_account": "checking"}))
    if out.get("status") == "not_executed":  # challenged: the model re-reads and fixes the value
        await kit.invoke("modify_autopay", {"bill_type": "utilities", "source_account": "savings"})


async def model_duplicate(kit, at):
    res = json.loads(await _call_at(kit, at, 2.2, "search_products", {"query": "blue mugs"}))
    pid = res["products"][0]["product_id"]
    await kit.invoke("add_to_cart", {"product_id": pid, "quantity": 2})
    await asyncio.sleep(0.2)
    await kit.invoke("add_to_cart", {"product_id": pid, "quantity": 2})  # e.g. after a nudge


MODELS = {"selftest_preemptive": model_preemptive, "selftest_miscomprehension": model_miscomprehension,
          "selftest_duplicate": model_duplicate}


async def _call_at(kit, at, t, name, args):
    await at(t)
    return await kit.invoke(name, args)


class PassGate:
    """Reference behaviour: no holding, no checks."""

    def __init__(self, timeline):
        self.timeline = timeline

    async def admit(self, *a, **k):
        return Decision(Verdict.EXECUTE)


class NoLedger(ActionLedger):
    async def run(self, name, args, fn):
        return await fn(), "executed"


# ── fake transport + ASR used by the harness's process_single ────────────────
def _write_wav(path, seconds, rate, tone=None):
    n = int(seconds * rate)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) * 30).astype(np.int16)  # faint room noise
    if tone:
        a, b = int(tone[0] * rate), int(tone[1] * rate)
        t = np.arange(b - a) / rate
        x[a:b] = (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(x.tobytes())


def _words(text, start, end):
    ws = text.split()
    step = (end - start) / max(len(ws), 1)
    return [{"word": w, "start": round(start + i * step, 2), "end": round(start + (i + 1) * step - 0.02, 2)}
            for i, w in enumerate(ws)]


class FakeASR:
    def __init__(self, scenario, agent_start):
        self.s, self.agent_start = scenario, agent_start

    def transcribe(self, paths, timestamps=True):
        p = str(paths[0])
        if "input" in Path(p).name:
            words = _words(self.s["user"], *self.s["speech"])
        else:
            words = _words(self.s["agent_says"], self.agent_start, self.agent_start + 2.0)

        class R:
            pass

        r = R()
        r.text = " ".join(w["word"] for w in words)
        r.timestamp = {"word": words}
        return [r]


def make_runner(scenario, mode):
    def fake_inference(input_path, output_path, provider):
        room = f"selftest-{mode}-{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        tl = ConversationTimeline()
        registry = MockAPIRegistry(latency_profile="instant")
        specs = fdb_tool_specs(registry, schema="extended")
        if mode == "holdfast":
            kit = GatedToolkit(specs, CommitGate(tl, POLICY), ActionLedger(), room_name=room, call_log_path=CALL_LOG)
        else:
            kit = GatedToolkit(specs, PassGate(tl), NoLedger(), room_name=room, call_log_path=CALL_LOG)

        async def at(t):
            await asyncio.sleep(max(0.0, t0 + t - time.time()))

        async def user():
            s, e = scenario["speech"]
            await at(s)
            tl.on_user_state("speaking")
            await at(e)
            tl.on_user_state("listening")
            await at(e + 0.1)
            tl.on_transcript(scenario["user"], is_final=True)

        async def main():
            await asyncio.gather(user(), MODELS[scenario["id"]](kit, at))

        asyncio.run(main())
        _write_wav(output_path, 7.0, 24000, tone=(3.4, 5.4))
        return room, t0

    return fake_inference


def score(v3, bench, data_dir, provider, out_dir):
    reports = {}
    for script, key in (("evaluate_pass_rate.py", "pass"), ("evaluate_tool_calls.py", "tools")):
        out = out_dir / f"{provider}_{key}.json"
        r = subprocess.run([sys.executable, str(v3 / script), "--benchmark", str(bench), "--results-dir",
                            str(data_dir), "--provider", provider, "--output", str(out)],
                           cwd=out_dir, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{script} failed:\n{r.stdout}\n{r.stderr}")
        reports[key] = json.loads(out.read_text())
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fdb-v3-dir", default=os.getenv("FDB_V3_DIR", ROOT / "third_party/Full-Duplex-Bench/v3"))
    ap.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = ap.parse_args()
    v3 = Path(args.fdb_v3_dir).resolve()
    sys.path.insert(0, str(v3))
    global MockAPIRegistry
    from mock_apis import MockAPIRegistry  # noqa: F401
    import run_tool_benchmark as rtb

    work = Path(tempfile.mkdtemp(prefix="holdfast_selftest_"))
    data_dir = work / "data"
    bench = work / "selftest_benchmark.json"
    bench.write_text(json.dumps({"benchmark_name": "holdfast-selftest", "scenarios": SCENARIOS}, indent=2))

    summary = {}
    for mode in ("holdfast", "naive"):
        provider = f"{mode}_selftest"
        for i, sc in enumerate(SCENARIOS):
            ex_dir = data_dir / f"{sc['id']}_{i:024x}"
            ex_dir.mkdir(parents=True, exist_ok=True)
            if not (ex_dir / "input.wav").exists():
                _write_wav(ex_dir / "input.wav", 7.0, 48000, tone=(sc["speech"][0], sc["speech"][1]))
            rtb.run_livekit_inference = make_runner(sc, mode)
            res = rtb.process_single("selftest", sc["id"], ex_dir / "input.wav", provider,
                                     {s["id"]: s for s in SCENARIOS}, FakeASR(sc, 3.4), force=True)
            assert res and res["status"] == "completed", res
        summary[mode] = score(v3, bench, data_dir, provider, work)

    print("\nHOLDFAST SELF-TEST (real FDB-v3 pipeline + scorers, scripted model, exact-match judge)")
    print(f"{'scenario':32} {'holdfast':>10} {'naive':>10}")
    by = {m: {r["scenario_id"]: r for r in summary[m]["pass"]["scenario_results"]} for m in summary}
    for sc in SCENARIOS:
        row = [("PASS" if by[m][sc["id"]]["passed"] else "FAIL") for m in ("holdfast", "naive")]
        print(f"{sc['id']:32} {row[0]:>10} {row[1]:>10}")
        if not by["naive"][sc["id"]]["passed"]:
            print(f"{'':34}naive failure: {by['naive'][sc['id']]['failure_reason']}")
    hp, np_ = summary["holdfast"]["pass"]["overall_pass_rate"], summary["naive"]["pass"]["overall_pass_rate"]
    print(f"{'pass rate':32} {hp:>10.0%} {np_:>10.0%}")
    ok = hp == 1.0 and np_ == 0.0
    print("RESULT:", "OK - pipeline wiring and control layer behave as designed" if ok else "UNEXPECTED")
    if args.keep:
        print("data kept in", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
