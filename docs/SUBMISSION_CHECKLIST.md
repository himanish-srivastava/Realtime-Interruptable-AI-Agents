# Submission checklist (Theme 05 participant guide)

| Guide requirement | Where | Status |
|---|---|---|
| LiveKit voice agent, any architecture | `agents/fdb_agent.py` (OpenAI Realtime + Holdfast control layer) | Done, CLI verified |
| Runs on FDB-v3 | `reproduce.sh` drives the unmodified harness at commit `3e799c4` | Wiring verified offline. **Needs one real run with your keys.** |
| Iterate on self-corrections and multi-step chains | commit gate, repair challenge, ledger, prompt, watchdog | Done. Tune on `devset/`. |
| One new use case, end to end | Holdfast Drive: `agents/drive_agent.py console` | Logic verified (`scripts/drive_sim.py`). **Needs a live run on your laptop.** |
| README: architecture (one diagram), exact setup/run, extension marked | `README.md` | Done |
| One-command reproduction + declared provider | `./reproduce.sh`; provider OpenAI Realtime `gpt-realtime-1.5` | Done. Test once on a machine that isn't yours. |
| Your best run's results + logs (scores, seeds, config) | `results/runs/<ts>/` (manifest, resolved config, pip freeze, all reports and logs) | **Run `RUNS=3 ./reproduce.sh` and commit the folder.** The realtime API has no seed (documented). |
| Demo video, 3–5 min | run sheet below | **To record** |
| Slide deck, ≤ 8 slides | outline below | **To make** |
| API keys documented, not included | `.env.example`, README "API keys" | Done |
| Pin seeds and versions | exact pins in `requirements-*.txt`, harness commit, manifest per run | Done (no seed exists for the model) |
| Don't hardcode or tune on test items | fair-play statement in README; original dev set | Done |
| No own servers at eval time; no cross-scenario cache | per-room runtime; only OpenAI + LiveKit | Done |

## Before you submit

1. `cp .env.example .env`, fill in the keys, then `RUNS=3 ./reproduce.sh` on your machine.
2. Read `results/runs/<ts>/summary.md` and `holdfast_audit.log`. Look for `stale` and `challenge`
   verdicts to see the gate working on real audio.
3. Clone the repo onto a second machine (or a fresh cloud VM with a GPU) and run
   `./reproduce.sh` there, exactly as the organizers will.
4. Decide the tool schema (README "Tool schema"). If in doubt, ask the organizers; switching is one line.
5. Record the video and make the slides from the outlines below, using **your real numbers**.

## Slide outline (8 slides)

1. **Problem.** Voice agents act on half-finished sentences. One sentence with a correction ("to
   Lisbon, no wait, Porto") produces a permanent wrong action.
2. **What FDB-v3 measures.** Strict pass rate, and why one extra call fails a scenario. Where
   baselines lose: self-corrections, multi-step chains.
3. **Three facts from the harness source.** Calls are logged at execution; LiveKit waits for
   running tools on barge-in; silence loses the response-quality score.
4. **Architecture.** The README diagram: gate, ledger, watchdog around the realtime model.
5. **The gate in one example.** A timeline of a correction: call held, user resumes, STALE,
   re-issued with the final value. Show an `holdfast_audit.log` excerpt.
6. **Results.** Your `summary.md` against the published GPT-Realtime baseline (same model), with
   breakdowns by disfluency type and difficulty, and mean ± std over runs.
7. **Extension: Holdfast Drive.** Destination change, reroute replacing the old route, no
   duplicate texts. A dashboard screenshot.
8. **Next.** Settle-window sweep on recorded speech, a streaming repair classifier, speculative
   read-only execution, compensating actions.

## Video run sheet (3–5 min, single takes)

- **0:00–0:20.** One sentence of framing: agents act on half-finished sentences; Holdfast doesn't.
- **0:20–2:00, benchmark.** Terminal 1: `python agents/fdb_agent.py dev`. Terminal 2: run one
  self-correction scenario through the harness
  (`python run_tool_benchmark.py --provider holdfast --example <id>` from `third_party/Full-Duplex-Bench/v3`).
  Play the input audio, then show `tail -f /tmp/holdfast_audit.log`, where a `stale` or
  `challenge` line appears, and `/tmp/agent_tool_calls.log`, where only the final call is logged.
- **2:00–4:15, extension.** `python agents/drive_agent.py console`, dashboard visible. Say the
  four lines from the README table, interrupting yourself naturally. Point at the dashboard after
  each: one route, replaced not stacked, one message and then one update.
- **4:15–4:45.** Show `summary.md` numbers and close.
