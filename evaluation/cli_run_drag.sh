#!/bin/bash

# DRAG Evaluation Command Cheatsheet
# ==================================
# Each block below is an example invocation of eval_drag.py for a specific backend.
# Update SAVE_DIR / model names as needed before running.

BENCHMARK_PATH="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/benchmark.json"
SAVE_DIR="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/results"

# --- vLLM (external server) --------------------------------------------------
# Assumes a vLLM server exposing the model at http://localhost:1053/v1

python eval_drag.py --task inference --backend vllm \
  --model_path 'osunlp/GUI-Drag-7B' \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name "GUI-Drag-7B" \
  --batch_size 4



# Optional: start a local vLLM server (run in a separate terminal)
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
#   --served-model-name qwen25vl \
#   --host 0.0.0.0 \
#   --port 1053 \
#   --max-model-len 16384

# --- Claude Computer-Use Agent -----------------------------------------------
python eval_drag.py \
  --task inference \
  --backend claude \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name claude_cua \
  --batch_size 1 \
  --max_concurrent 1

# Add --without_hint to drop the sentence-level hint used in our paper
# python eval_drag.py --task inference --backend claude \
#   --benchmark "$BENCHMARK_PATH" \
#   --save_dir "$SAVE_DIR" \
#   --model_save_name claude_cua_nohint \
#   --batch_size 1 \
#   --max_concurrent 1 \
#   --without_hint

# --- OpenAI Operator ---------------------------------------------------------
python eval_drag.py \
  --task inference \
  --backend operator \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name operator_preview \
  --batch_size 16 \
  --max_concurrent 16

# Optional no-hint variant for Operator
# python eval_drag.py --task inference --backend operator \
#   --benchmark "$BENCHMARK_PATH" \
#   --save_dir "$SAVE_DIR" \
#   --model_save_name operator_preview_nohint \
#   --batch_size 16 \
#   --max_concurrent 16 \
#   --without_hint

# --- UI TARS (UITAR backend) -------------------------------------------------
# Requires a UITAR-compatible endpoint, e.g. UI-TARS served through vLLM at 1054
python eval_drag.py \
  --task inference \
  --backend uitar \
  --base_url http://localhost:1054/v1 \
  --model_name ui-tars \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name uitar_15
