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
