"""Export a trained slither run for the browser: web/slither/model.onnx and web/slither/stats.json.

  uv run --group train python scripts/build_slither_assets.py --run-dir runs/slither \
      --final-eval results/slither_eval.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.model import load_slither_policy

ROOT = Path(__file__).resolve().parents[1]


def export_onnx(model, path):
    dummy = torch.zeros(1, model.obs_dim)
    torch.onnx.export(model, (dummy,), str(path), input_names=["obs"], output_names=["logits", "value"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sample = torch.rand(16, model.obs_dim)
    with torch.no_grad():
        want_logits, want_value = model(sample)
    got_logits, got_value = ort.InferenceSession(str(path)).run(None, {"obs": sample.numpy()})
    np.testing.assert_allclose(got_logits, want_logits.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(got_value, want_value.numpy(), rtol=1e-3, atol=1e-3)


def curve(metrics_path):
    points = []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            if row["mass_share_vs_hunter"]:
                points.append({"env_steps": int(row["env_steps"]), "hours": float(row["hours"]),
                               "mass_share_vs_hunter": float(row["mass_share_vs_hunter"]),
                               "mass_share_vs_forager": float(row["mass_share_vs_forager"])})
    return points


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/slither")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--final-eval", help="results JSON from scripts/eval_slither.py")
    p.add_argument("--out-dir", default=str(ROOT / "web" / "slither"))
    args = p.parse_args(argv)

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_slither_policy(run_dir / args.checkpoint)
    export_onnx(model, out_dir / "model.onnx")
    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    final = json.loads(Path(args.final_eval).read_text())["results"] if args.final_eval else None
    stats = {
        "final_eval": final,
        "curve": curve(run_dir / "metrics.csv"),
        "checkpoint": {"iteration": checkpoint["iteration"], "env_steps": checkpoint["env_steps"],
                       "metrics": checkpoint["metrics"]},
        "config": SlitherConfig().to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out_dir / 'model.onnx'} and {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
