import pytest

from inferbench.datasets_loader import load_structured, load_rag, load_summary, RagItem


def test_structured_is_deterministic_and_has_schema():
    a = load_structured(n=5)
    b = load_structured(n=5)
    assert [x.instruction for x in a] == [x.instruction for x in b]
    assert all(isinstance(x.json_schema, dict) and x.json_schema for x in a)
    assert len(a) == 5


@pytest.mark.network
def test_rag_and_summary_deterministic():
    r1 = load_rag(n=3)
    r2 = load_rag(n=3)
    assert [x.question for x in r1] == [x.question for x in r2]
    s1 = load_summary(n=3)
    s2 = load_summary(n=3)
    assert [x.reference for x in s1] == [x.reference for x in s2]
    assert isinstance(r1[0], RagItem)
