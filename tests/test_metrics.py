import math

from inferbench.models import RequestResult, SweepCell
from inferbench.metrics import percentile, aggregate


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([10], 99) == 10.0


def test_aggregate_computes_tps_and_error_rate():
    rs = [RequestResult(ok=True, latency_s=1.0, prompt_tokens=200, completion_tokens=100, ttft_s=0.1),
          RequestResult(ok=True, latency_s=2.0, prompt_tokens=200, completion_tokens=100, ttft_s=0.2),
          RequestResult(ok=False, latency_s=0.0, prompt_tokens=0, completion_tokens=0, error="boom")]
    cell = SweepCell(config="awq", workload="chat", concurrency=2)
    c = aggregate(cell, "self_host", rs, wall_clock_s=4.0, gpu_mem_gb=6.0)
    assert c.n == 3
    assert math.isclose(c.error_rate, 1 / 3, rel_tol=1e-6)
    # only successful completions count toward throughput: 200 tokens / 4s
    assert math.isclose(c.output_tps, 50.0, rel_tol=1e-6)
    assert c.gpu_mem_gb == 6.0
    assert c.p50_s > 0  # computed over successful latencies only
