# Testing & Results Capture

How this project verifies its own correctness, and how it captures benchmark
results. Two distinct things share the word "test" here:

1. **Unit tests** — prove the toolkit's logic is correct (the methodology is
   enforced, the math is right) without a GPU or network.
2. **Benchmark runs** — produce the actual measured results (latency, throughput,
   quality) as committed JSON artifacts.

This doc covers both.

---

## 1. Running the unit tests

```bash
# fast suite — no network, no GPU (default for development)
.venv/bin/python -m pytest -m "not network" -q

# include the dataset-download tests (hits HuggingFace Hub)
.venv/bin/python -m pytest -q

# a single module
.venv/bin/python -m pytest tests/test_metrics.py -v
```

- **27 tests**, 16 files. All but one run fully offline and deterministically.
- One test is marked `@pytest.mark.network` (the RAG/summary dataset loaders that
  download from HuggingFace) and is **excluded by default** via `-m "not network"`.
- Config lives in `pyproject.toml` (`asyncio_mode = "auto"`, the `network` marker).

### How the suite stays GPU-free and network-free

Every component that touches the outside world is tested through an injected
fake, so the logic is verified in isolation:

| Component | Real dependency | Test double |
|---|---|---|
| `client.complete` | live HTTP / SSE stream | `respx` mock returning a canned SSE body |
| `runner.run_cell` | the client | a `fake_complete` coroutine |
| `sweep.run_sweep` | datasets + tokenizer + client | `monkeypatch` + `FakeTok` + `fake_complete` |
| `workloads.build_messages` | model tokenizer | `FakeTok` (whitespace tokenizer) |
| `evals.grounding/summary` | an LLM judge | an injected `judge(prompt) -> float` |
| `evals.json_validity` | (none — deterministic) | literal output strings |

---

## 2. What the tests actually guarantee

The unit tests are not box-ticking — each one pins a specific methodology or
correctness property that the benchmark's credibility depends on.

| Test file | Guarantees |
|---|---|
| `test_metrics.py` | percentiles interpolate correctly; throughput = completion tokens ÷ wall-clock over **successful** requests only; failed requests count toward `error_rate`, not latency |
| `test_client.py` | TTFT captured at first content token; authoritative token counts read from `usage`; HTTP/timeout errors recorded as `ok=False`; **`ignore_eos` sent only to self-host**, never managed |
| `test_runner.py` | warmup requests are discarded before measurement; the `asyncio.Semaphore` caps in-flight requests at the target concurrency; aggregation runs over the measured batch |
| `test_workloads.py` | prompts are shaped to the per-workload token target (truncated via the model tokenizer); workload specs are (200/100), (2000/300), (8000/500) |
| `test_datasets_loader.py` | loaders are **deterministic** (same `n` → identical items); structured prompts carry a JSON schema |
| `test_json_validity.py` | JSON extraction tolerates ```` ```json ```` fences; validity requires required keys + correct primitive types |
| `test_grounding.py` / `test_summary.py` | scorers build a judge prompt containing the context/answer/document and pass the score through (judge injected) |
| `test_eval_aggregate.py` | quality is expressed **relative to the BF16 baseline** (`delta_vs_baseline`, `retained_pct`) |
| `test_cost.py` | self-host $/1M = GPU $/hr ÷ throughput; the crossover tps is the algebraic inverse (cost equals managed price at that point) |
| `test_slo.py` | SLO evaluation accumulates **all** violations (not first-fail); null thresholds mean "not enforced" |
| `test_charts.py` | the Pareto frontier drops dominated configs; chart renderers write non-empty PNGs |
| `test_report.py` | the results table renders rows (GPU shown as `—` for managed); the report has all required sections |
| `test_sweep.py` | the full pipeline (config → loaders → workloads → runner → serialization) writes one JSON per cell |

To see which methodology rule a test protects, the failure message and the
assertion read as the spec: e.g. `test_run_cell_discards_warmup_and_respects_concurrency`
asserts `len(calls) == 12` (2 warmup + 10 measured sent) but `res.n == 10`.

---

## 3. How benchmark results are captured

Real results come from a **sweep run**, not from the unit tests. The driver
(`inferbench.sweep.run_sweep`) loops `config × workload × concurrency`, and for
each cell writes one JSON file.

### Where they go

```
results/<config>_<workload>_<concurrency>.json
# e.g. results/managed_chat_5.json, results/awq_summary_50.json
```

`results/` is git-ignored by default (raw artifacts shouldn't bloat history);
real result files are committed **deliberately** with `git add -f` so the report
is reproducible from the repo.

### What one cell file contains

Each file is a serialized `CellResult` (`inferbench/models.py`). Realistic shape:

```json
{
  "cell": { "config": "managed", "workload": "chat", "concurrency": 5 },
  "kind": "managed",
  "n": 50,
  "p50_s": 0.41,
  "p95_s": 0.68,
  "p99_s": 0.92,
  "ttft_p50_s": 0.08,
  "ttft_p95_s": 0.14,
  "output_tps": 712.5,
  "req_per_s": 7.1,
  "error_rate": 0.0,
  "gpu_mem_gb": null,
  "raw": [
    { "ok": true, "latency_s": 0.42, "prompt_tokens": 200,
      "completion_tokens": 100, "ttft_s": 0.08, "error": null }
  ]
}
```

Notes:
- **`gpu_mem_gb` is `null` for managed cells** — the managed GPU is not
  observable (the black-box limitation). Self-host cells populate it from
  `nvidia-smi` .
- **`raw`** keeps every per-request record, so any new statistic (a different
  percentile, a histogram) can be recomputed from saved JSON **without re-running**
  the benchmark.
- Aggregate fields (`p50_s`, `output_tps`, …) are computed by
  `inferbench.metrics.aggregate` over the successful requests in `raw`.

### Methodology applied during capture (not just asserted in tests)

- **Warmup discarded:** the first `warmup` requests per cell are sent but not
  recorded (`configs/sweep.yaml: warmup`).
- **Fixed output length:** self-host runs send `ignore_eos` + a fixed
  `max_tokens` so tokens/sec is comparable; managed runs cannot force this
  (documented limitation).
- **N per cell:** `configs/sweep.yaml: n` (default 50) requests measured per cell.
- **Concurrency:** an `asyncio.Semaphore` holds exactly `concurrency` requests in
  flight.
- **Pinned inputs:** datasets load fixed indices; default `seed: 1234`.

---

## 4. Running a real capture

### Managed (no GPU — available now)

```bash
export NEBIUS_API_KEY=...                                              # your Token Factory key
.venv/bin/python scripts/run_managed.py --sweep configs/sweep.yaml --reduced   # smoke: n=10, conc [1,5]
.venv/bin/python scripts/run_managed.py --sweep configs/sweep.yaml             # full sweep
git add -f results/managed_*.json && git commit -m "chore: managed benchmark results"
```

`--reduced` runs a quick smoke (n=10, concurrency [1,5]) to confirm the endpoint
works before committing to the full 3-workload × 4-concurrency sweep.

### Self-host (Self-host — needs a GPU)

Uncomment the self-host endpoints in `configs/sweep.yaml`, start the vLLM servers
(`serve/launch_*.sh`), then run `scripts/run_selfhost.py` (the sweep step), which
also captures GPU memory via `nvidia-smi`. See
[the self-host phase](docs/RUNBOOK_BREV.md).

---

## 5. Reproducibility checklist

A result is only credible if someone else can regenerate it. This repo captures:

- **Pinned datasets** (fixed split + fixed indices, no shuffle) — same prompts
  every run.
- **Fixed seed** (`1234`) and `temperature: 0.0`.
- **Committed raw results** (`raw` per request) so statistics can be recomputed.
- **Environment metadata** (the run records GPU type, CUDA/driver version, vLLM
  version, checkpoint commit hashes alongside the results).
- **Generated report** — the README table is produced from `results/*.json`, never
  hand-typed, so it cannot drift from what was actually measured.
```
