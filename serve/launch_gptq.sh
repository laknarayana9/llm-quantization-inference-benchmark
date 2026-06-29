#!/usr/bin/env bash
# GPTQ-Int4 — Qwen team's official GPTQ checkpoint on vLLM (port 8002).
set -euo pipefail
# See launch_bf16.sh: avoid FlashInfer sampler JIT (needs nvcc the box may lack).
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --port 8002 \
  --quantization gptq_marlin \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
