#!/usr/bin/env bash
# BF16 baseline — full-precision Qwen2.5-7B-Instruct on vLLM (port 8000).
set -euo pipefail
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
