"""Orchestrate quality scoring across configs, relative to a baseline.

Given per-config model outputs for a workload, score each config with the
workload's metric (deterministic JSON validity for chat; injected LLM judge for
rag/summary) and express the result relative to the baseline config.
"""
from __future__ import annotations

from inferbench.evals.json_validity import is_valid
from inferbench.evals.grounding import score_grounding
from inferbench.evals.summary import score_summary
from inferbench.evals.aggregate import relative_to_baseline


def score_outputs(workload: str, outputs: dict[str, list[str]], refs: list[dict],
                  judge, baseline: str) -> dict[str, dict]:
    """outputs: config -> list of generated strings (aligned with `refs`).

    refs carry the per-item ground truth: {"schema": ...} for chat,
    {"context","question"} for rag, {"document"} for summary.
    """
    config_scores: dict[str, list[float]] = {}
    for cfg, outs in outputs.items():
        scores: list[float] = []
        for out, ref in zip(outs, refs):
            if workload == "chat":
                scores.append(1.0 if is_valid(out, ref["schema"]) else 0.0)
            elif workload == "rag":
                scores.append(score_grounding(ref["context"], ref["question"], out, judge))
            elif workload == "summary":
                scores.append(score_summary(ref["document"], out, judge))
            else:
                raise ValueError(f"unknown workload {workload}")
        config_scores[cfg] = scores
    return relative_to_baseline(config_scores, baseline=baseline)
