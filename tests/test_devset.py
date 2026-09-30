import json
import subprocess
import sys
from pathlib import Path

import yaml

from holdfast.fdb_tools import FDB_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def test_devset_scenarios_are_well_formed_and_cover_the_space():
    sc = yaml.safe_load((ROOT / "devset" / "scenarios.yaml").read_text())["scenarios"]
    assert len({s["id"] for s in sc}) == len(sc)
    tools = {c["function"] for s in sc for c in s["expected"]}
    assert tools == set(FDB_TOOL_NAMES)
    feats = {f for s in sc for f in s["disfluency_features"]}
    assert feats == {"FILLER", "PAUSE", "HESITATION", "FALSE_START", "SELF_CORRECTION"}
    assert {s["difficulty"] for s in sc} == {"easy", "medium", "hard"}
    for s in sc:
        assert len(s["expected"]) == {"easy": 1, "medium": 2, "hard": 3}[s["difficulty"]], s["id"]


def test_build_devset_writes_harness_compatible_metadata(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "devset" / "build_devset.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    bench = json.loads((ROOT / "devset" / "devset_benchmark.json").read_text())
    s0 = bench["scenarios"][0]
    for key in ("id", "title", "domain", "difficulty", "expected_tool_calls", "dialogue", "disfluency_features"):
        assert key in s0
    import re
    folders = [p.name for p in (ROOT / "devset" / "data").iterdir()]
    assert all(re.match(r"^(.+)_([0-9a-f]{24})$", f) for f in folders)
