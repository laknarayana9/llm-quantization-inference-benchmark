from inferbench.models import SweepCell, CellResult
from inferbench.analysis.slo import SloProfile, evaluate


def _cell(p95, ttft, err):
    return CellResult(cell=SweepCell(config="awq", workload="chat", concurrency=5),
                      kind="self_host", n=10, p50_s=1, p95_s=p95, p99_s=p95,
                      ttft_p50_s=ttft, ttft_p95_s=ttft, output_tps=50, req_per_s=5,
                      error_rate=err, gpu_mem_gb=6.0, raw=[])


def test_slo_pass_and_violations():
    prof = SloProfile(ttft_max_s=0.5, p95_max_s=2.0, quality_min=0.8, error_rate_max=0.01)
    assert evaluate(_cell(1.5, 0.3, 0.0), prof, quality=0.9)["passes"]
    bad = evaluate(_cell(3.0, 0.9, 0.05), prof, quality=0.5)
    assert not bad["passes"]
    assert len(bad["violations"]) == 4  # p95, ttft, quality, error_rate all violated
