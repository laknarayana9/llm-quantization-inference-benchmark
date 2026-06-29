from inferbench.evals.aggregate import relative_to_baseline


def test_relative_to_baseline():
    scores = {"bf16": [1.0, 1.0], "awq": [0.9, 0.9], "gptq": [0.8, 1.0]}
    out = relative_to_baseline(scores, baseline="bf16")
    assert out["bf16"]["mean"] == 1.0
    assert abs(out["awq"]["delta_vs_baseline"] + 0.1) < 1e-9
    assert abs(out["awq"]["retained_pct"] - 90.0) < 1e-9
