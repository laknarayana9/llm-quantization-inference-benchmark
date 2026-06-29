from inferbench.analysis.gpu import parse_used_mem_mib


def test_parse_used_mem_mib_to_gb():
    csv = "memory.used [MiB]\n15360 MiB\n"
    assert abs(parse_used_mem_mib(csv) - 15.0) < 1e-6   # 15360 MiB -> 15.0 GB


def test_parse_handles_plain_number():
    assert abs(parse_used_mem_mib("6144\n") - 6.0) < 1e-6


def test_parse_empty_returns_zero():
    assert parse_used_mem_mib("") == 0.0
