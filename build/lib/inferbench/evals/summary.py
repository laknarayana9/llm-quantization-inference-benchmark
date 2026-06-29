from __future__ import annotations

from inferbench.evals.judge import JudgeFn


def score_summary(document: str, summary: str, judge: JudgeFn) -> float:
    prompt = (
        "Score from 0.0 to 1.0 the quality of the SUMMARY for the DOCUMENT "
        "(coverage, accuracy, conciseness). Reply with only the number.\n\n"
        f"DOCUMENT:\n{document}\n\nSUMMARY:\n{summary}\n\nScore:")
    return judge(prompt)
