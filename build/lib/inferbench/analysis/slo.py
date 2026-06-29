from __future__ import annotations

from dataclasses import dataclass

from inferbench.models import CellResult


@dataclass(frozen=True)
class SloProfile:
    error_rate_max: float
    ttft_max_s: float | None = None
    p95_max_s: float | None = None
    quality_min: float | None = None


def evaluate(cell_result: CellResult, profile: SloProfile, quality: float | None) -> dict:
    v: list[str] = []
    if profile.p95_max_s is not None and cell_result.p95_s > profile.p95_max_s:
        v.append(f"p95 {cell_result.p95_s:.3f}s > {profile.p95_max_s}s")
    if profile.ttft_max_s is not None and cell_result.ttft_p95_s > profile.ttft_max_s:
        v.append(f"ttft_p95 {cell_result.ttft_p95_s:.3f}s > {profile.ttft_max_s}s")
    if profile.quality_min is not None and (quality is None or quality < profile.quality_min):
        v.append(f"quality {quality} < {profile.quality_min}")
    if cell_result.error_rate > profile.error_rate_max:
        v.append(f"error_rate {cell_result.error_rate:.3f} > {profile.error_rate_max}")
    return {"passes": not v, "violations": v}
