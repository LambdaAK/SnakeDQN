import csv
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_slither", ROOT / "scripts" / "train_slither.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_iterations_write_checkpoints_and_metrics(tmp_path):
    run = tmp_path / "run"
    load_script().main(["--run-dir", str(run), "--num-envs", "4", "--max-snakes", "4", "--min-snakes", "3",
                        "--rollout", "8", "--iterations", "2", "--eval-every", "2", "--eval-arenas", "2",
                        "--eval-decisions", "10", "--minibatch", "32", "--hidden", "32", "--device", "cpu"])
    for name in ("config.json", "metrics.csv", "latest.pt", "best.pt", "checkpoints/iter_00002.pt"):
        assert (run / name).exists(), name
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert len(rows) == 2 and rows[-1]["mass_share_vs_hunter"] != ""
