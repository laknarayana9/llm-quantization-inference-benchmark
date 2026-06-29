from inferbench.models import EndpointConfig, RequestResult, SweepCell, CellResult


def test_endpoint_config_defaults():
    ep = EndpointConfig(name="awq", base_url="http://x/v1",
                        model="Qwen/Qwen2.5-7B-Instruct-AWQ", kind="self_host")
    assert ep.api_key is None and ep.kind == "self_host"


def test_request_result_roundtrip():
    r = RequestResult(ok=True, ttft_s=0.1, latency_s=1.2, prompt_tokens=200,
                      completion_tokens=100, error=None)
    assert r.model_dump()["completion_tokens"] == 100


def test_cell_result_requires_cell():
    c = CellResult(cell=SweepCell(config="awq", workload="chat", concurrency=5),
                   kind="self_host", n=0, p50_s=0, p95_s=0, p99_s=0,
                   ttft_p50_s=0, ttft_p95_s=0, output_tps=0, req_per_s=0,
                   error_rate=0, gpu_mem_gb=None, raw=[])
    assert c.cell.concurrency == 5
