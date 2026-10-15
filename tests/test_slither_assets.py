import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from snake_rl.slither.model import SlitherNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_export_matches_pytorch(tmp_path):
    net = SlitherNet(hidden=32).eval()
    path = tmp_path / "m.onnx"
    load("build_slither_assets").export_onnx(net, path)
    x = np.random.default_rng(0).random((7, 263)).astype(np.float32)
    logits, value = ort.InferenceSession(str(path)).run(None, {"obs": x})
    with torch.no_grad():
        want_logits, want_value = net(torch.from_numpy(x))
    np.testing.assert_allclose(logits, want_logits.numpy(), atol=1e-4)
    np.testing.assert_allclose(value, want_value.numpy(), atol=1e-4)


def test_eval_cli_bot_vs_bot(tmp_path):
    out = tmp_path / "e.json"
    load("eval_slither").main(["bot:forager", "--bots", "wander", "--arenas", "4", "--decisions", "30",
                               "--warmup", "5", "--out", str(out)])
    import json
    data = json.loads(out.read_text())
    assert data["policy"] == "bot:forager" and 0 <= data["results"]["wander"]["mass_share"] <= 1
