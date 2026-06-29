#!/usr/bin/env bash
# GPTQ-Int4 — Qwen team's official GPTQ checkpoint on vLLM (port 8002).
set -euo pipefail
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --port 8002 \
  --quantization gptq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --enable-chunked-prefill
