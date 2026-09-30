#!/usr/bin/env bash
# Holdfast: one-command FDB-v3 reproduction (install -> configure -> evaluate).
#
#   cp .env.example .env   # fill in LiveKit Cloud + OpenAI keys
#   ./reproduce.sh
#
# Declared model provider: OpenAI Realtime API, model gpt-realtime-1.5 (config/holdfast.yaml).
# Judge: the benchmark's own gpt-4o LLM judge (--use-llm), unchanged.
#
# Optional environment variables:
#   RUNS=3                  repeat the full benchmark N times and report mean/std (default 1)
#   FDB_DATA_DIR=/path      use an existing fdb_v3_data_released folder (skips download)
#   DATASET=devset          run our own dev set (devset/data) instead of FDB-v3
#   SKIP_INSTALL=1          reuse existing .venv-agent / .venv-harness
#   PYTHON=python3.10       interpreter used to create the venvs (>= 3.10)
#   HOLDFAST_SET="a.b=v"    config overrides, e.g. "tools.schema=template"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
FDB_REPO=https://github.com/DanielLin94144/Full-Duplex-Bench.git
FDB_COMMIT=3e799c45a045256f47d5f1c9cda90157e2d2ec9e
V3="$ROOT/third_party/Full-Duplex-Bench/v3"
PY="${PYTHON:-python3}"
RUNS="${RUNS:-1}"
DATASET="${DATASET:-fdb}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/runs/$RUN_ID"
AGENT_PID=""

step() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
cleanup() { [ -n "$AGENT_PID" ] && kill "$AGENT_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

step "Preflight"
command -v git >/dev/null || die "git is required"
command -v ffmpeg >/dev/null || die "ffmpeg is required (e.g. apt install ffmpeg)"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python >= 3.10 required; set PYTHON="
[ -f "$ROOT/.env" ] || die "missing .env: cp .env.example .env and fill in the keys"
set -a; . "$ROOT/.env"; set +a
for v in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET OPENAI_API_KEY; do
  [ -n "${!v:-}" ] || die "$v is empty in .env"
done
mkdir -p "$OUT"

step "FDB-v3 harness at pinned commit ${FDB_COMMIT:0:7}"
if [ ! -d "$ROOT/third_party/Full-Duplex-Bench/.git" ]; then
  git clone --quiet "$FDB_REPO" "$ROOT/third_party/Full-Duplex-Bench"
fi
git -C "$ROOT/third_party/Full-Duplex-Bench" fetch --quiet origin "$FDB_COMMIT" 2>/dev/null || true
git -C "$ROOT/third_party/Full-Duplex-Bench" checkout --quiet "$FDB_COMMIT"
# the harness reads credentials from v3/.env.local (cwd-relative)
printf 'LIVEKIT_URL=%s\nLIVEKIT_API_KEY=%s\nLIVEKIT_API_SECRET=%s\nOPENAI_API_KEY=%s\n' \
  "$LIVEKIT_URL" "$LIVEKIT_API_KEY" "$LIVEKIT_API_SECRET" "$OPENAI_API_KEY" > "$V3/.env.local"

if [ -z "${SKIP_INSTALL:-}" ]; then
  step "Installing agent environment (.venv-agent)"
  "$PY" -m venv "$ROOT/.venv-agent"
  "$ROOT/.venv-agent/bin/pip" install -q --upgrade pip
  "$ROOT/.venv-agent/bin/pip" install -q -r requirements-agent.txt
  step "Installing harness environment (.venv-harness, includes NeMo ASR; this takes a while)"
  "$PY" -m venv "$ROOT/.venv-harness"
  "$ROOT/.venv-harness/bin/pip" install -q --upgrade pip
  "$ROOT/.venv-harness/bin/pip" install -q -r requirements-harness.txt
fi
AGENT_PY="$ROOT/.venv-agent/bin/python"
HARNESS_PY="$ROOT/.venv-harness/bin/python"
export FDB_V3_DIR="$V3"

step "Offline checks (unit tests + harness self-test; no API calls)"
"$AGENT_PY" -m pytest -q tests | tail -2
"$AGENT_PY" scripts/harness_selftest.py | sed -n '/HOLDFAST SELF-TEST/,$p'

step "Benchmark data ($DATASET)"
if [ "$DATASET" = "devset" ]; then
  DATA="$ROOT/devset/data"
  BENCH="$ROOT/devset/devset_benchmark.json"
  [ -f "$BENCH" ] && [ -d "$DATA" ] || die "dev set not built; see devset/README.md"
else
  if [ -n "${FDB_DATA_DIR:-}" ]; then DATA="$(cd "$FDB_DATA_DIR" && pwd)"
  else DATA="$("$HARNESS_PY" scripts/fetch_fdb_data.py "$ROOT/third_party/fdb_data" | tail -1)"; fi
  BENCH="$V3/benchmark_data_v2.json"
  # make the harness's default commands (e.g. run_tool_benchmark.py --example ...) find the data too
  [ -e "$V3/fdb_v3_data_released" ] || ln -s "$DATA" "$V3/fdb_v3_data_released"
fi
N_EX=$(find "$DATA" -mindepth 2 -maxdepth 2 -name input.wav | wc -l)
echo "data: $DATA ($N_EX examples)"
[ "$N_EX" -gt 0 ] || die "no input.wav files under $DATA"

# manifest: everything needed to match our logs to a re-run
{
  echo "run_id: $RUN_ID"
  echo "holdfast_commit: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "fdb_commit: $FDB_COMMIT"
  echo "dataset: $DATASET ($N_EX examples)"
  echo "runs: $RUNS"
  echo "holdfast_set: ${HOLDFAST_SET:-}"
  echo "host: $(uname -srm)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/gpu: /'
} > "$OUT/manifest.txt"
"$AGENT_PY" -c 'import json,yaml; from holdfast.config import load_config; print(yaml.safe_dump(load_config(), sort_keys=False))' > "$OUT/config_resolved.yaml"
"$AGENT_PY" -m pip freeze > "$OUT/pip_freeze_agent.txt"
"$HARNESS_PY" -m pip freeze > "$OUT/pip_freeze_harness.txt"

BASE_PROVIDER="$("$AGENT_PY" -c 'from holdfast.config import load_config; print(load_config()["provider_id"])')"
for r in $(seq 1 "$RUNS"); do
  PROVIDER="$BASE_PROVIDER"; [ "$RUNS" -gt 1 ] && PROVIDER="${BASE_PROVIDER}_r$r"
  RDIR="$OUT/run$r"; mkdir -p "$RDIR"
  step "Run $r/$RUNS (provider id: $PROVIDER)"

  # fresh telemetry for this run (the harness matches calls by room name)
  for f in /tmp/agent_tool_calls.log /tmp/agent_heartbeat.log /tmp/holdfast_audit.log; do : > "$f"; done

  "$AGENT_PY" agents/fdb_agent.py start > "$RDIR/agent.log" 2>&1 &
  AGENT_PID=$!
  for _ in $(seq 1 90); do
    grep -q "registered worker" "$RDIR/agent.log" && break
    kill -0 "$AGENT_PID" 2>/dev/null || { tail -30 "$RDIR/agent.log"; die "agent exited during startup"; }
    sleep 1
  done
  grep -q "registered worker" "$RDIR/agent.log" || { tail -30 "$RDIR/agent.log"; die "agent did not register with LiveKit"; }
  echo "agent registered (pid $AGENT_PID)"

  (cd "$V3" && "$HARNESS_PY" run_tool_benchmark_all_released.py --provider "$PROVIDER" --root_dir "$DATA" --force) \
    2>&1 | tee "$RDIR/inference.log" | grep -E "^\[|Summary|Total|Success|Error|Avg" || true
  kill "$AGENT_PID" 2>/dev/null || true; wait "$AGENT_PID" 2>/dev/null || true; AGENT_PID=""

  step "Scoring run $r (benchmark's own scripts, gpt-4o judge)"
  (cd "$V3" && "$HARNESS_PY" evaluate_tool_calls.py --benchmark "$BENCH" --results-dir "$DATA" \
      --provider "$PROVIDER" --output "$RDIR/evaluation_report.json" --use-llm) > "$RDIR/eval_tool_calls.log" 2>&1
  (cd "$V3" && "$HARNESS_PY" evaluate_pass_rate.py --benchmark "$BENCH" --results-dir "$DATA" \
      --provider "$PROVIDER" --output "$RDIR/pass_rate_report.json" --use-llm) > "$RDIR/eval_pass_rate.log" 2>&1
  (cd "$V3" && "$HARNESS_PY" analyze_tool_latency.py --results-dir "$DATA" --provider "$PROVIDER" \
      --output "$RDIR/latency_report.json") > "$RDIR/eval_latency.log" 2>&1 || echo "latency analysis failed (see log)"

  cp /tmp/agent_tool_calls.log "$RDIR/agent_tool_calls.log"
  cp /tmp/holdfast_audit.log "$RDIR/holdfast_audit.log"
  cp /tmp/agent_heartbeat.log "$RDIR/agent_heartbeat.log"
  mkdir -p "$RDIR/results"
  find "$DATA" -name "result_${PROVIDER}.json" -exec sh -c 'd=$(basename "$(dirname "$1")"); cp "$1" "$2/$d.json"' _ {} "$RDIR/results" \;
done

step "Summary"
"$AGENT_PY" scripts/summarize_results.py "$OUT"
echo "All logs, reports and the manifest are in: $OUT"
