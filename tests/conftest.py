import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _find_fdb_v3():
    for cand in (os.getenv("FDB_V3_DIR"), ROOT / "third_party" / "Full-Duplex-Bench" / "v3"):
        if cand and (Path(cand) / "mock_apis.py").exists():
            return Path(cand)
    return None


@pytest.fixture(scope="session")
def fdb_registry():
    """The benchmark's own MockAPIRegistry (skips if the harness is not checked out)."""
    v3 = _find_fdb_v3()
    if v3 is None:
        pytest.skip("FDB-v3 harness not found; set FDB_V3_DIR or run reproduce.sh once")
    sys.path.insert(0, str(v3))
    from mock_apis import MockAPIRegistry

    return MockAPIRegistry(latency_profile="instant")
