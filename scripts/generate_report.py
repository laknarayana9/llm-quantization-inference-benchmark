"""Generate hero charts from committed results (the report step).

Loads results/*.json, computes the self-host-vs-managed cost crossover, and
renders the report charts into report/charts/. Cost inputs are CLI flags so the
real GPU $/hr and managed price are explicit.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inferbench.analysis.cost import self_host_cost_per_1m, crossover_tps  # noqa: E402

CONFIG_COLORS = {"bf16": "#c1440e", "awq": "#1f6f3c", "gptq": "#2b5fa3", "managed": "#888888"}
CONFIG_LABELS = {"bf16": "BF16", "awq": "AWQ-Int4", "gptq": "GPTQ-Int4", "managed": "Managed (Qwen3-30B)"}


def load_cells(results_dir: str) -> dict:
    cells = {}
    for f in glob.glob(f"{results_dir}/*_*.json"):
        if "outputs" in f or "quality" in f:
            continue
        d = json.load(open(f))
        c = d["cell"]
        cells[(c["config"], c["workload"], c["concurrency"])] = d
    return cells


def _series(cells, configs, workload, field):
    out = {}
    for cfg in configs:
        pts = []
        for conc in [1, 5, 20, 50]:
            d = cells.get((cfg, workload, conc))
            if d:
                pts.append((conc, d[field]))
        if pts:
            out[cfg] = pts
    return out


def line_chart(path, series, ylabel, title, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg, pts in series.items():
        xs = [c for c, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", label=CONFIG_LABELS.get(cfg, cfg),
                color=CONFIG_COLORS.get(cfg))
    ax.set_xlabel("concurrency (simultaneous users)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([1, 5, 20, 50])
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cost_crossover_chart(path, gpu_hr, managed_per_1m, configs_tps):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tps = list(range(20, 5000, 20))
    ax.plot(tps, [self_host_cost_per_1m(gpu_hr, t) for t in tps],
            label=fr"self-host \$/1M (L40S \${gpu_hr}/hr)", color="#1f6f3c")
    ax.axhline(managed_per_1m, linestyle="--", color="#888",
               label=fr"managed \${managed_per_1m}/1M out")
    xo = crossover_tps(gpu_hr, managed_per_1m)
    ax.axvline(xo, linestyle=":", color="#c1440e", alpha=0.7)
    ax.annotate(f"crossover ≈ {xo:.0f} tok/s", (xo, managed_per_1m * 2.4),
                fontsize=9, color="#c1440e")
    for name, t in configs_tps.items():
        ax.scatter([t], [self_host_cost_per_1m(gpu_hr, t)], s=40, zorder=5,
                   color=CONFIG_COLORS.get(name), label=f"{CONFIG_LABELS.get(name,name)} @ peak ({t:.0f} tok/s)")
    ax.set_xlabel("sustained output tokens/sec")
    ax.set_ylabel(r"\$ / 1M output tokens")
    ax.set_title("Self-host vs managed cost crossover")
    ax.set_ylim(0, managed_per_1m * 4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def memory_chart(path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fmts = ["BF16", "AWQ-Int4", "GPTQ-Int4"]
    weights = [14.29, 5.29, 5.27]
    bars = ax.bar(fmts, weights, color=["#c1440e", "#1f6f3c", "#2b5fa3"])
    for b, w in zip(bars, weights):
        ax.text(b.get_x() + b.get_width() / 2, w + 0.2, f"{w} GiB", ha="center", fontsize=10)
    ax.set_ylabel("model weights (GiB)")
    ax.set_title("Model weights memory — INT4 is ~2.7× smaller (Qwen2.5-7B)")
    ax.set_ylim(0, 16)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="report/charts")
    ap.add_argument("--gpu-hr", type=float, default=1.06, help="self-host GPU $/hr")
    ap.add_argument("--managed-per-1m", type=float, default=0.30,
                    help="managed $ per 1M output tokens")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.results_dir)
    self_configs = ["bf16", "awq", "gptq"]
    all_configs = ["bf16", "awq", "gptq", "managed"]

    # throughput vs concurrency (chat) and p95 latency vs concurrency (summary)
    line_chart(f"{args.out_dir}/throughput_chat.png",
               _series(cells, all_configs, "chat", "output_tps"),
               "output tokens/sec", "Throughput vs concurrency — chat workload")
    line_chart(f"{args.out_dir}/latency_summary.png",
               _series(cells, all_configs, "summary", "p95_s"),
               "p95 latency (s)", "p95 latency vs concurrency — 8k-context summary")

    # cost crossover, using each self-host config's PEAK measured throughput
    peak = {}
    for cfg in self_configs:
        tps = [cells[(cfg, "chat", c)]["output_tps"] for c in [50] if (cfg, "chat", 50) in cells]
        if tps:
            peak[cfg] = max(tps)
    cost_crossover_chart(f"{args.out_dir}/cost_crossover.png",
                         args.gpu_hr, args.managed_per_1m, peak)

    memory_chart(f"{args.out_dir}/memory_weights.png")

    print(f"Crossover throughput: {crossover_tps(args.gpu_hr, args.managed_per_1m):.0f} tok/s")
    print(f"Charts written to {args.out_dir}/:")
    for p in sorted(glob.glob(f"{args.out_dir}/*.png")):
        print("  ", p)


if __name__ == "__main__":
    main()
