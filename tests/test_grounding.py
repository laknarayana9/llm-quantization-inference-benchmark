from inferbench.evals.grounding import score_grounding


def test_grounding_uses_injected_judge():
    captured = {}

    def judge(prompt: str) -> float:
        captured["p"] = prompt
        return 0.8

    s = score_grounding("ctx text", "q?", "ans", judge)
    assert s == 0.8
    assert "ctx text" in captured["p"] and "ans" in captured["p"]
