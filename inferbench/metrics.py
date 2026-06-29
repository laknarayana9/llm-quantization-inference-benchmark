from __future__ import annotations

from inferbench.models import RequestResult, SweepCell, CellResult


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def aggregate(cell: SweepCell, kind: str, results: list[RequestResult],
              wall_clock_s: float, gpu_mem_gb: float | None = None) -> CellResult:
    ok = [r for r in results if r.ok]
    lat = [r.latency_s for r in ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    total_completion = sum(r.completion_tokens for r in ok)
    n = len(results)
    return CellResult(
        cell=cell, kind=kind, n=n,
        p50_s=percentile(lat, 50), p95_s=percentile(lat, 95), p99_s=percentile(lat, 99),
        ttft_p50_s=percentile(ttft, 50), ttft_p95_s=percentile(ttft, 95),
        output_tps=(total_completion / wall_clock_s) if wall_clock_s > 0 else 0.0,
        req_per_s=(len(ok) / wall_clock_s) if wall_clock_s > 0 else 0.0,
        error_rate=(1 - len(ok) / n) if n else 0.0,
        gpu_mem_gb=gpu_mem_gb, raw=results,
    )
