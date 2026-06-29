# Self-Hosted vLLM Serving — Launch & Tuning

One launch script per quantization format. Each serves the OpenAI-compatible API
that the benchmark client targets (just like Token Factory), on a distinct port
so the sweep config can address them by `base_url`.

| Script | Checkpoint | Port | Quantization |
|---|---|---|---|
| `launch_bf16.sh` | `Qwen/Qwen2.5-7B-Instruct` | 8000 | none (BF16 baseline) |
| `launch_awq.sh` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 8001 | `awq_marlin` |
| `launch_gptq.sh` | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | 8002 | `gptq_marlin` |

## The tuning knobs (the "deployment shape")

These flags are the levers a field engineer pulls. Each maps to a specific
behavior or failure it controls:

- **`--gpu-memory-utilization 0.90`** — fraction of VRAM vLLM may use for weights
  + KV cache. Higher = bigger KV cache = more concurrency, but too high risks OOM
  from CUDA/activation overhead. Lower this first if you hit OOM.
- **`--max-num-seqs 64`** — max sequences in the running batch (the concurrency
  cap at the engine). Raising it lifts throughput until VRAM/compute saturate;
  lowering it is the primary fix for KV-cache OOM at high concurrency.
- **`--max-model-len 8192`** — context budget. Bounds the KV cache per sequence
  (KV grows with sequence length × batch). Must be ≥ the 8k summary workload's
  input + output.
- **`--enable-chunked-prefill`** — interleaves long-prompt prefill with ongoing
  decode so one 8k-token prefill doesn't stall everyone else's token generation.
  Big win for the long-context (summary) workload's tail latency.
- **`--quantization {awq_marlin,gptq_marlin}`** — selects the optimized Marlin
  Int4 kernels for the pre-quantized weights. (BF16 omits this.)

## Running

On a single GPU, serve **one format at a time** (each wants ~90% of VRAM), run
that config's sweep, stop it, then start the next:

```bash
bash serve/launch_bf16.sh        # terminal 1 — wait for "Application startup complete"
# verify:
curl -s http://localhost:8000/v1/models | head
# in terminal 2, run the sweep for whatever self-host endpoints are up, then Ctrl-C the server.
```

See [the Brev runbook](../docs/RUNBOOK_BREV.md) for the full GPU session flow.

## Failure-mode experiments (the OOM experiment)

To reproduce the KV-cache OOM deliberately, raise `--max-num-seqs` (e.g. 256) and
run the 8k summary workload at concurrency 50 — then show the fix (lower
`--max-num-seqs` / `--gpu-memory-utilization`). Document symptom → cause → fix in
`report/failure_modes.md`.
