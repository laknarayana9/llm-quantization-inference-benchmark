import httpx
import respx

from inferbench.models import EndpointConfig
from inferbench.client import complete

SSE = (
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
    'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":200,"completion_tokens":2}}\n\n'
    'data: [DONE]\n\n'
)


@respx.mock
async def test_complete_captures_ttft_and_usage():
    respx.post("http://x/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"}))
    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        r = await complete(c, ep, [{"role": "user", "content": "hi"}], max_tokens=2)
    assert r.ok and r.completion_tokens == 2 and r.prompt_tokens == 200
    assert r.ttft_s is not None and r.latency_s >= r.ttft_s


@respx.mock
async def test_complete_records_http_error():
    respx.post("http://x/v1/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        r = await complete(c, ep, [{"role": "user", "content": "hi"}], max_tokens=2)
    assert not r.ok and r.error is not None
