from inferbench.workloads import WORKLOADS, build_messages, count_tokens


class FakeTok:
    def encode(self, s):
        return s.split()

    def decode(self, toks):
        return " ".join(toks)


def test_workload_specs():
    assert WORKLOADS["chat"].target_in == 200 and WORKLOADS["chat"].target_out == 100
    assert WORKLOADS["summary"].target_in == 8000


def test_build_messages_truncates_to_target():
    tok = FakeTok()
    long_ctx = " ".join(["word"] * 5000)
    item = type("I", (), {"context": long_ctx, "question": "What?", "answer": "x"})()
    msgs = build_messages("rag", item, tok)
    n = count_tokens(msgs[-1]["content"], tok)
    assert 0.5 * 2000 <= n <= 1.1 * 2000  # context truncated toward 2000-token target
    assert msgs[0]["role"] == "system"
