import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="spike_radar_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("MODELS_DIR", f"{_TMP}/models")
os.environ["RUN_COLLECTOR"] = "false"
os.environ["RUN_WORKER"] = "false"
os.environ["RUN_RADAR"] = "false"
os.environ.setdefault("BOOTSTRAP_ITERS", "80")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from app import db  # noqa: E402


@pytest.fixture()
def engine(tmp_path):
    eng = db.make_engine(f"sqlite:///{tmp_path}/t.db")
    db.set_engine(eng)
    yield eng
    eng.dispose()
