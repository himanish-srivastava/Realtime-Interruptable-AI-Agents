#!/usr/bin/env python3
"""Aggregate one or more FDB-v3 runs (run*/ subfolders of a results directory)
into summary.json and summary.md: mean, std, min, max per metric across runs.

Only the organizers' re-run is scored; these numbers document our own runs."""

import json
import statistics
import sys
from pathlib import Path


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _stats(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4), "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}


def _run_metrics(run: Path) -> tuple[dict, dict]:
    m, b = {}, {}
    pr, ev, lat = (_load(run / n) for n in ("pass_rate_report.json", "evaluation_report.json", "latency_report.json"))
    if pr:
        m["pass_rate"] = pr.get("overall_pass_rate")
        for group in ("by_domain", "by_difficulty", "by_disfluency_feature"):
            for k, v in (pr.get(group) or {}).items():
                b[f"{group}.{k}"] = v
    if ev:
        bm = ev.get("by_metric", {})
        m["tool_selection_f1"] = bm.get("tool_selection_acc")
        m["argument_accuracy"] = bm.get("argument_acc")
        m["response_quality"] = bm.get("response_qual")
        m["turn_take_rate"] = ev.get("turn_taking", {}).get("turn_take_rate")
        m["avg_response_latency_s"] = ev.get("latency", {}).get("avg_response_latency_s")
        m["interruption_rate"] = ev.get("latency", {}).get("interruption_rate")
    if lat:
        for k, v in (lat.get("aggregate") or {}).items():
            if isinstance(v, dict):
                for stat in ("mean", "median"):
                    if isinstance(v.get(stat), (int, float)):
                        m[f"latency.{k}.{stat}"] = v[stat]
    return m, b


LABELS = {"pass_rate": "Pass rate (strict)", "tool_selection_f1": "Tool selection F1",
          "argument_accuracy": "Argument accuracy", "response_quality": "Response quality",
          "turn_take_rate": "Turn-take rate", "avg_response_latency_s": "Avg response latency (s)",
          "interruption_rate": "Interruption rate"}


def summarize(results_dir) -> dict:
    root = Path(results_dir)
    runs = sorted(p for p in root.glob("run*") if p.is_dir())
    per = [_run_metrics(r) for r in runs]
    keys_m = sorted({k for m, _ in per for k in m})
    keys_b = sorted({k for _, b in per for k in b})
    out = {
        "runs": len(runs),
        "metrics": {k: _stats([m.get(k) for m, _ in per]) for k in keys_m},
        "breakdown": {k: _stats([b.get(k) for _, b in per]) for k in keys_b},
    }
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Holdfast FDB-v3 results ({len(runs)} run(s))", "", "| Metric | Mean | Std | Min | Max |",
             "|---|---|---|---|---|"]
    for k in keys_m:
        s = out["metrics"][k]
        if s:
            lines.append(f"| {LABELS.get(k, k)} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    lines += ["", "## Pass rate breakdown", "", "| Slice | Mean | Std |", "|---|---|---|"]
    for k in keys_b:
        s = out["breakdown"][k]
        if s:
            lines.append(f"| {k} | {s['mean']:.3f} | {s['std']:.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    s = summarize(sys.argv[1] if len(sys.argv) > 1 else ".")
    print((Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "summary.md").read_text())
