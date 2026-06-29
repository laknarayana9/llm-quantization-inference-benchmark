from __future__ import annotations


def mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def relative_to_baseline(config_scores: dict[str, list[float]], baseline: str) -> dict[str, dict]:
    base = mean(config_scores[baseline])
    out: dict[str, dict] = {}
    for cfg, scores in config_scores.items():
        m = mean(scores)
        out[cfg] = {
            "mean": m,
            "delta_vs_baseline": m - base,
            "retained_pct": (m / base * 100.0) if base else 0.0,
        }
    return out
