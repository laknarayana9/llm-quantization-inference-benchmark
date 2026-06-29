# Brev GPU Runbook — Self-Host Sweep 

Step-by-step for the ~$20 Brev session. Everything is pre-built and tested, so
this is pure execution. **Goal: minimize GPU hours.** Budget ~3–5 hours.

## 0. Pick the GPU

- **L40S (48GB)** — best fit. BF16-7B (~15GB) + KV cache for 8k×high-concurrency
  fits comfortably; AWQ/GPTQ (~6GB) leave tons of headroom.
- **A10G / A100-40GB** also fine. On a **24GB** card, BF16 at 8k×50 may OOM — which
  is the Task 6 OOM experiment, so that's acceptable (use it intentionally).
- One GPU; no tensor-parallel (7B doesn't need it).

## 1. Set up the box (~15 min)

```bash
git clone https://github.com/laknarayana9/llm-quantization-inference-benchmark.git
cd llm-quantization-inference-benchmark
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install vllm            # pulls torch + CUDA wheels
python -m pytest -m "not network" -q   # sanity: 36 tests pass on the GPU box too
nvidia-smi                  # confirm GPU + driver
```

Pre-download the checkpoints so the first request doesn't stall:

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-7B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct-AWQ",
          "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"]:
    snapshot_download(m)
EOF
```

## 2. Per-format sweep (serve one at a time)

The self-host endpoints are already uncommented in `configs/sweep.yaml`
(bf16:8000, awq:8001, gptq:8002). Since you serve one format at a time, use
`--only <name>` so the sweep targets just the live server (no dead-port errors).

> **Always launch via the `serve/launch_*.sh` scripts** — never `vllm serve …` by
> hand. The scripts pin `--max-model-len 8192`; omitting it makes vLLM provision
> for Qwen's full 32k window, which changes KV-cache sizing, max concurrency, and
> the OOM threshold — your "8k benchmark" would no longer be controlled or
> comparable. See [serve/README.md](../serve/README.md) for the full rationale.

For each format (`bf16`, `awq`, `gptq`):

```bash
# terminal 1 — start the server, wait for "Application startup complete"
bash serve/launch_awq.sh                          # (or _bf16 / _gptq)
curl -s http://localhost:8001/v1/models | head    # verify it's up

# terminal 2 — smoke first (cheap), then full; --only matches the running server
python scripts/run_selfhost.py --only awq --reduced
python scripts/run_selfhost.py --only awq
# Ctrl-C the server in terminal 1 before starting the next format.
```

After all three: you'll have `results/{bf16,awq,gptq}_{chat,rag,summary}_{1,5,20,50}.json`
with real `gpu_mem_gb` captured.

## 3. Quality eval run (low concurrency, natural completions)

`scripts/run_evals.py` is ready. Two steps fit the one-server-at-a-time flow:
generate each config's outputs while its server is up, then score them all at the
end (the judge calls your managed Token Factory endpoint, so keep `.env.local`
present and `NEBIUS_API_KEY` exported).

```bash
# while each server is up (gen reuses the same prompts; ignore_eos off = natural length):
python scripts/run_evals.py gen --workload rag --config bf16 --n 30   # bf16 server up
python scripts/run_evals.py gen --workload rag --config awq  --n 30   # awq server up
python scripts/run_evals.py gen --workload rag --config gptq --n 30   # gptq server up

# then score all three vs the baseline (needs NEBIUS_API_KEY for the judge):
set -a; . ./.env.local; set +a
python scripts/run_evals.py score --workload rag --configs bf16,awq,gptq --baseline bf16
# repeat for --workload chat (deterministic JSON, no judge) and summary.
```

This writes `results/quality_<workload>.json` with each config's mean score and
**% retained vs the BF16 baseline** — your quantization-quality-degradation numbers.

## 4. Failure-mode experiments (Task 6)

- **OOM:** edit `serve/launch_bf16.sh` to `--max-num-seqs 256`, run the 8k summary
  at concurrency 50, capture the OOM, then show the fix. Record in
  `report/failure_modes.md`.
- **Long-prompt TTFT / p99 spikes:** already visible in your sweep data; quote them.

## 5. Generate report (can be done off-GPU after)

Pull the committed results back and run the report step (`scripts/generate_report.py`)
to render the hero charts + README. Plug in the real Brev GPU $/hr and Token
Factory price for the cost crossover.

## Commit results

```bash
git add -f results/{bf16,awq,gptq}_*.json results/quality_*.json
git commit -m "chore: self-host vLLM benchmark results"
git push
```

## Time/cost discipline

- Smoke (`--reduced`) before every full sweep — catch issues at 10 requests, not 50.
- Stop each server the instant its sweep finishes.
- If running low on credit, **skip SGLang v1.1** (it's the optional milestone) and
  make sure you at least have the three full self-host sweeps committed.
