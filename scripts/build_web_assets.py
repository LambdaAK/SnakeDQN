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
    np.testing.assert_allclose(got_logits, want_logits.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(got_value, want_value.numpy(), rtol=1e-3, atol=1e-3)


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


def training_svg(points, path):
    """Static version of the demo's chart for the README (GitHub can't run the page's JS)."""
    W, H, L, R, T, B = 720, 290, 48, 16, 36, 52
    max_x = points[-1]["env_steps"]
    x = lambda v: L + (W - L - R) * v / max_x  # noqa: E731
    y = lambda v: T + (H - T - B) * (1 - v)  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             'font-family="system-ui, sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" rx="12" fill="#131a2e"/>',
             '<text x="48" y="22" fill="#e6e9f2" font-size="14" font-weight="600">'
             'Win rate during self-play training</text>']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#26304f"/>')
        parts.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" fill="#8d97b5" text-anchor="end">{int(v * 100)}%</text>')
    for k in range(6):
        v = max_x * k / 5
        parts.append(f'<text x="{x(v):.1f}" y="{H - 30}" fill="#8d97b5" text-anchor="middle">{v / 1e6:.0f}M</text>')
    for key, color in (("win_vs_flood_fill", "#f59e0b"), ("win_vs_greedy", "#38bdf8")):
        d = "".join(f'{"L" if i else "M"}{x(p["env_steps"]):.1f},{y(p[key]):.1f}' for i, p in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts.append('<rect x="470" y="13" width="10" height="10" rx="2" fill="#f59e0b"/>'
                 '<text x="485" y="22" fill="#8d97b5">vs flood-fill bot</text>'
                 '<rect x="595" y="13" width="10" height="10" rx="2" fill="#38bdf8"/>'
                 '<text x="610" y="22" fill="#8d97b5">vs greedy bot</text>')
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 10}" fill="#8d97b5" text-anchor="middle">self-play turns</text>')
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/selfplay")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--final-eval", help="JSON written by scripts/final_eval.py, merged into stats")
    p.add_argument("--out-dir", default=str(ROOT / "web"))
    p.add_argument("--model-name", default="model.onnx")
    p.add_argument("--stats-name", default="stats.json")
    p.add_argument("--chart", default=str(ROOT / "docs" / "training.svg"),
                   help="README chart path; empty to skip")
    args = p.parse_args(argv)

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    model = load_policy(run_dir / args.checkpoint)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    export_onnx(model, out_dir / args.model_name)

    stats = {
        "checkpoint": {"iteration": checkpoint["iteration"], "env_steps": checkpoint["env_steps"],
                       "parameters": sum(p.numel() for p in model.parameters())},
        "curve": curve(run_dir / "metrics.csv"),
    }
    if args.final_eval:
        stats["final_eval"] = json.loads(Path(args.final_eval).read_text())
    (out_dir / args.stats_name).write_text(json.dumps(stats, indent=2) + "\n")
    if args.chart and stats["curve"]:
        training_svg(stats["curve"], args.chart)
    print(f"wrote {out_dir / args.model_name} and {out_dir / args.stats_name} "
          f"(iteration {checkpoint['iteration']}, {checkpoint['env_steps']:,} env steps)")


if __name__ == "__main__":
    main()
