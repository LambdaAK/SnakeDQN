"""Export a trained run for the browser demo: web/model.onnx and web/stats.json.

  uv run --group train python scripts/build_web_assets.py --run-dir runs/selfplay
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from snake_rl.agents import load_policy

ROOT = Path(__file__).resolve().parents[1]


def export_onnx(model, path):
    b = model.board_size
    dummy = torch.zeros(1, model.channels, b, b)
    torch.onnx.export(model, (dummy,), str(path), input_names=["obs"],
                      output_names=["logits", "value"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sample = torch.rand(16, model.channels, b, b)
    with torch.no_grad():
        want_logits, want_value = model(sample)
    session = ort.InferenceSession(str(path))
    got_logits, got_value = session.run(None, {"obs": sample.numpy()})
    np.testing.assert_allclose(got_logits, want_logits.numpy(), atol=1e-4)
    np.testing.assert_allclose(got_value, want_value.numpy(), atol=1e-4)


def curve(metrics_path):
    points = []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            if row["win_vs_flood_fill"]:
                points.append({
                    "env_steps": int(row["env_steps"]),
                    "hours": float(row["hours"]),
                    "win_vs_flood_fill": float(row["win_vs_flood_fill"]),
                    "win_vs_greedy": float(row["win_vs_greedy"]),
                })
    return points


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/selfplay")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--final-eval", help="JSON written by scripts/final_eval.py, merged into stats")
    p.add_argument("--out-dir", default=str(ROOT / "web"))
    args = p.parse_args(argv)

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    model = load_policy(run_dir / args.checkpoint)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    export_onnx(model, out_dir / "model.onnx")

    stats = {
        "checkpoint": {"iteration": checkpoint["iteration"], "env_steps": checkpoint["env_steps"],
                       "parameters": sum(p.numel() for p in model.parameters())},
        "curve": curve(run_dir / "metrics.csv"),
    }
    if args.final_eval:
        stats["final_eval"] = json.loads(Path(args.final_eval).read_text())
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out_dir / 'model.onnx'} and {out_dir / 'stats.json'} "
          f"(iteration {checkpoint['iteration']}, {checkpoint['env_steps']:,} env steps)")


if __name__ == "__main__":
    main()
