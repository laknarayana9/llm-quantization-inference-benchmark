from inferbench.evals.summary import score_summary


def test_summary_uses_injected_judge():
    def judge(prompt: str) -> float:
        assert "document" in prompt.lower()
        return 0.6

    assert score_summary("the document", "a summary", judge) == 0.6
