from pathlib import Path

from inferbench.analysis.charts import pareto_frontier, render_latency_vs_concurrency


def test_pareto_frontier_drops_dominated():
    # (label, latency, quality): lower latency + higher quality is better
    pts = [("A", 1.0, 0.9), ("B", 2.0, 0.8), ("C", 1.5, 0.95)]
    front = set(pareto_frontier(pts))
    assert "A" in front and "C" in front  # B dominated by A (faster and better)
    assert "B" not in front


def test_render_writes_png(tmp_path: Path):
    out = tmp_path / "lat.png"
    render_latency_vs_concurrency(str(out), {"awq": [(1, 0.5), (5, 0.9)]})
    assert out.exists() and out.stat().st_size > 0
