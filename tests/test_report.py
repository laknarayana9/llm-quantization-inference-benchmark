from inferbench.models import SweepCell, CellResult
from inferbench.report import results_table, build_report


def _c():
    return CellResult(cell=SweepCell(config="awq", workload="chat", concurrency=5),
                      kind="self_host", n=10, p50_s=1, p95_s=1.8, p99_s=2.0,
                      ttft_p50_s=0.2, ttft_p95_s=0.3, output_tps=120, req_per_s=5,
                      error_rate=0.0, gpu_mem_gb=6.0, raw=[])


def test_results_table_has_row():
    md = results_table([_c()])
    assert "awq" in md and "1.80" in md and "|" in md


def test_build_report_sections():
    doc = build_report([_c()], recommendation="Use AWQ for chat.",
                       limitations="Managed is a black box.")
    for h in ["# ", "## Methodology", "## Results", "## Recommendation", "## Limitations"]:
        assert h in doc
