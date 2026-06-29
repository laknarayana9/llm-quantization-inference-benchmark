#!/usr/bin/env bash
# AWQ-Int4 — Qwen team's official AWQ checkpoint on vLLM (port 8001).
set -euo pipefail
# See launch_bf16.sh: avoid FlashInfer sampler JIT (needs nvcc the box may lack).
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --port 8001 \
  --quantization awq_marlin \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
