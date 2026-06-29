from __future__ import annotations

import json
import time

import httpx

from inferbench.models import EndpointConfig, RequestResult


async def complete(client: httpx.AsyncClient, ep: EndpointConfig, messages: list[dict],
                   max_tokens: int, seed: int = 1234, ignore_eos: bool = True,
                   timeout_s: float = 120.0) -> RequestResult:
    url = ep.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
    payload = {
        "model": ep.model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": seed, "stream": True,
        "stream_options": {"include_usage": True},
    }
    # ignore_eos is a vLLM-only extension that forces fixed output length. Managed
    # APIs (e.g. Token Factory) reject unknown params, so only send it to self-host
    # endpoints. Managed output length is therefore not forced (documented limitation).
    if ignore_eos and ep.kind == "self_host":
        payload["ignore_eos"] = True
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers,
                                 timeout=timeout_s) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")[:200]
                return RequestResult(ok=False, latency_s=time.perf_counter() - start,
                                     prompt_tokens=0, completion_tokens=0,
                                     error=f"http {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        completion_tokens += 1
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                    completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
        return RequestResult(ok=True, latency_s=time.perf_counter() - start,
                             prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                             ttft_s=ttft)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        return RequestResult(ok=False, latency_s=time.perf_counter() - start,
                             prompt_tokens=0, completion_tokens=0, error=f"{type(e).__name__}: {e}")
