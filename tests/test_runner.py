import asyncio

from inferbench.models import EndpointConfig, SweepCell, RequestResult
from inferbench.runner import run_cell


async def test_run_cell_discards_warmup_and_respects_concurrency():
    calls = []
    live = 0
    peak = 0

    async def fake_complete(client, ep, messages, max_tokens, **kw):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        calls.append(messages)
        return RequestResult(ok=True, latency_s=0.01, prompt_tokens=10,
                             completion_tokens=5, ttft_s=0.005)

    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    cell = SweepCell(config="t", workload="chat", concurrency=3)
    payloads = [[{"role": "user", "content": str(i)}] for i in range(100)]
    res = await run_cell(fake_complete, ep, cell, payloads, max_tokens=5,
                         warmup=2, n=10, concurrency=3)
    assert res.n == 10                      # warmup excluded from results
    assert len(calls) == 12                 # 2 warmup + 10 measured were sent
    assert peak <= 3                        # concurrency cap respected
    assert res.output_tps > 0
