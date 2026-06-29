from __future__ import annotations

from inferbench.evals.judge import JudgeFn


def score_grounding(context: str, question: str, answer: str, judge: JudgeFn) -> float:
    prompt = (
        "Score from 0.0 to 1.0 how fully the ANSWER is supported by the CONTEXT "
        "(faithfulness/grounding). Reply with only the number.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\nScore:")
    return judge(prompt)
