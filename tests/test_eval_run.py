from inferbench.evals.run import score_outputs


def test_score_outputs_chat_uses_json_validity():
    # chat workload: deterministic JSON validity, no judge needed
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    outputs = {"bf16": ['{"name":"A"}', '{"name":"B"}'],
               "awq": ['{"name":"A"}', 'broken']}
    res = score_outputs("chat", outputs, refs=[{"schema": schema}, {"schema": schema}],
                        judge=None, baseline="bf16")
    assert res["bf16"]["mean"] == 1.0
    assert res["awq"]["mean"] == 0.5
    assert res["awq"]["retained_pct"] == 50.0


def test_score_outputs_rag_uses_injected_judge():
    judge = lambda prompt: 0.8  # noqa: E731
    outputs = {"bf16": ["ans1"], "awq": ["ans2"]}
    refs = [{"context": "ctx", "question": "q"}]
    res = score_outputs("rag", outputs, refs=refs, judge=judge, baseline="bf16")
    assert res["bf16"]["mean"] == 0.8 and res["awq"]["retained_pct"] == 100.0
