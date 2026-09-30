import json

from scripts.summarize_results import summarize


def write(run, name, obj):
    run.mkdir(parents=True, exist_ok=True)
    (run / name).write_text(json.dumps(obj))


def fake_run(root, i, pass_rate, f1):
    run = root / f"run{i}"
    write(run, "pass_rate_report.json", {"overall_pass_rate": pass_rate, "total_scenarios": 100,
                                         "by_domain": {"housing_location": pass_rate},
                                         "by_difficulty": {"hard": pass_rate / 2},
                                         "by_disfluency_feature": {"SELF_CORRECTION": pass_rate}})
    write(run, "evaluation_report.json", {"by_metric": {"tool_selection_acc": f1, "argument_acc": 0.7,
                                                        "response_qual": 0.8},
                                          "turn_taking": {"turn_take_rate": 1.0},
                                          "latency": {"avg_response_latency_s": 4.0, "interruption_rate": 0.05}})
    write(run, "latency_report.json", {"aggregate": {"tool_call_latency_s": {"mean": 3.1, "median": 3.0}}})


def test_summarize_aggregates_runs_with_mean_and_std(tmp_path):
    fake_run(tmp_path, 1, 0.60, 0.90)
    fake_run(tmp_path, 2, 0.70, 0.92)
    s = summarize(tmp_path)
    assert s["runs"] == 2
    assert abs(s["metrics"]["pass_rate"]["mean"] - 0.65) < 1e-9
    assert s["metrics"]["pass_rate"]["std"] > 0
    assert s["metrics"]["tool_selection_f1"]["mean"] == 0.91
    assert s["metrics"]["latency.tool_call_latency_s.mean"]["mean"] == 3.1
    assert s["breakdown"]["by_domain.housing_location"]["mean"] == 0.65
    assert (tmp_path / "summary.json").exists() and "Pass rate" in (tmp_path / "summary.md").read_text()


def test_summarize_tolerates_a_missing_report(tmp_path):
    fake_run(tmp_path, 1, 0.5, 0.9)
    (tmp_path / "run1" / "latency_report.json").unlink()
    s = summarize(tmp_path)
    assert s["metrics"]["pass_rate"]["mean"] == 0.5
