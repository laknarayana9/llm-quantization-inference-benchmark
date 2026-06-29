from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inferbench.analysis.cost import self_host_cost_per_1m  # noqa: E402


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    keep = []
    for label, lat, qual in points:
        dominated = any((o_lat <= lat and o_qual >= qual) and (o_lat < lat or o_qual > qual)
                        for o_label, o_lat, o_qual in points if o_label != label)
        if not dominated:
            keep.append(label)
    return keep


def render_cost_crossover(path: str, gpu_usd_per_hr: float,
                          configs_tps: dict[str, float], managed_usd_per_1m: float) -> None:
    fig, ax = plt.subplots()
    tps_axis = [t for t in range(10, 2000, 10)]
    ax.plot(tps_axis, [self_host_cost_per_1m(gpu_usd_per_hr, t) for t in tps_axis],
            label="self-host $/1M")
    ax.axhline(managed_usd_per_1m, linestyle="--", label="managed $/1M")
    for name, tps in configs_tps.items():
        ax.scatter([tps], [self_host_cost_per_1m(gpu_usd_per_hr, tps)], label=name)
    ax.set_xlabel("sustained output tokens/sec")
    ax.set_ylabel("$ / 1M output tokens")
    ax.set_ylim(0, managed_usd_per_1m * 4)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_latency_vs_concurrency(path: str, series: dict[str, list[tuple[int, float]]]) -> None:
    fig, ax = plt.subplots()
    for name, pts in series.items():
        xs = [c for c, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("p95 latency (s)")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
