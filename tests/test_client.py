import json

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
async def test_ignore_eos_only_sent_to_self_host():
    route = respx.post("http://x/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"}))
    msgs = [{"role": "user", "content": "hi"}]

    # managed endpoint: vLLM-only ignore_eos must NOT be sent (managed APIs reject it)
    ep_m = EndpointConfig(name="m", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        await complete(c, ep_m, msgs, max_tokens=2)
    managed_body = json.loads(route.calls.last.request.content)
    assert "ignore_eos" not in managed_body

    # self-host endpoint: ignore_eos IS sent (vLLM honors it for fixed output length)
    ep_s = EndpointConfig(name="s", base_url="http://x/v1", model="m", kind="self_host")
    async with httpx.AsyncClient() as c:
        await complete(c, ep_s, msgs, max_tokens=2)
    self_host_body = json.loads(route.calls.last.request.content)
    assert self_host_body.get("ignore_eos") is True


@respx.mock
async def test_complete_captures_text_when_requested():
    respx.post("http://x/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"}))
    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        r = await complete(c, ep, [{"role": "user", "content": "hi"}], max_tokens=2, capture_text=True)
    assert r.text == "Hello world"


@respx.mock
async def test_complete_omits_text_by_default():
    respx.post("http://x/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"}))
    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        r = await complete(c, ep, [{"role": "user", "content": "hi"}], max_tokens=2)
    assert r.text is None


@respx.mock
async def test_complete_records_http_error():
    respx.post("http://x/v1/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
    ep = EndpointConfig(name="t", base_url="http://x/v1", model="m", kind="managed")
    async with httpx.AsyncClient() as c:
        r = await complete(c, ep, [{"role": "user", "content": "hi"}], max_tokens=2)
    assert not r.ok and r.error is not None
