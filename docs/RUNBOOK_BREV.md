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

Collect real outputs and score them. Use a strong judge model (e.g. via the
managed Token Factory you already have a key for — point `api_complete_fn` at a
larger Qwen3 model). This is the eval step's `scripts/run_evals.py` (write it from
`inferbench.evals.run.score_outputs` + `capture_text=True`, `ignore_eos=False`).

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
