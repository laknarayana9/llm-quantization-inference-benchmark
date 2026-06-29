import json
from pathlib import Path

from inferbench.models import EndpointConfig, RequestResult
from inferbench.sweep import run_sweep


class FakeTok:
    def encode(self, s):
        return s.split()

    def decode(self, toks):
        return " ".join(toks)


async def fake_complete(client, ep, messages, max_tokens, **kw):
    return RequestResult(ok=True, latency_s=0.05, prompt_tokens=200,
                         completion_tokens=max_tokens, ttft_s=0.01)


def test_build_endpoints_filters_by_kind(monkeypatch):
    from inferbench.sweep import build_endpoints
    monkeypatch.setenv("NEBIUS_API_KEY", "secret")
    sweep = {"endpoints": [
        {"name": "managed", "base_url": "http://x/v1", "model": "m", "kind": "managed",
         "api_key_env": "NEBIUS_API_KEY"},
        {"name": "bf16", "base_url": "http://y/v1", "model": "m", "kind": "self_host"},
    ]}
    managed = build_endpoints(sweep, kind="managed")
    assert len(managed) == 1 and managed[0].api_key == "secret"
    self_host = build_endpoints(sweep, kind="self_host")
    assert len(self_host) == 1 and self_host[0].api_key is None
    assert len(build_endpoints(sweep)) == 2  # no filter -> all


async def test_gpu_mem_fn_applied_only_to_self_host(tmp_path: Path, monkeypatch):
    import inferbench.sweep as S
    monkeypatch.setattr(
        S, "_structured_items",
        lambda n: [type("I", (), {"instruction": "x", "json_schema": {}})()] * n)
    sweep = {"workloads": ["chat"], "concurrency": [1], "warmup": 0, "n": 2,
             "max_tokens": {"chat": 5}}
    eps = [EndpointConfig(name="managed", base_url="http://x/v1", model="m", kind="managed"),
           EndpointConfig(name="bf16", base_url="http://y/v1", model="m", kind="self_host")]
    cells = await run_sweep(sweep, eps, fake_complete, FakeTok(), str(tmp_path),
                            gpu_mem_fn=lambda: 12.5)
    by_kind = {c.kind: c for c in cells}
    assert by_kind["self_host"].gpu_mem_gb == 12.5   # captured for self-host
    assert by_kind["managed"].gpu_mem_gb is None     # NOT captured for managed


async def test_run_sweep_writes_cells(tmp_path: Path, monkeypatch):
    # tiny sweep: 1 workload x 2 concurrency, n=4, warmup=1
    import inferbench.sweep as S
    monkeypatch.setattr(
        S, "_structured_items",
        lambda n: [type("I", (), {"instruction": "x", "json_schema": {}})()] * n)
    sweep = {"workloads": ["chat"], "concurrency": [1, 2], "warmup": 1, "n": 4,
             "max_tokens": {"chat": 5}}
    ep = EndpointConfig(name="managed", base_url="http://x/v1", model="m", kind="managed")
    cells = await run_sweep(sweep, [ep], fake_complete, FakeTok(), str(tmp_path))
    assert len(cells) == 2
    assert (tmp_path / "managed_chat_1.json").exists()
    saved = json.loads((tmp_path / "managed_chat_2.json").read_text())
    assert saved["cell"]["concurrency"] == 2
