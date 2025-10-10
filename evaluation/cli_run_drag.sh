#!/bin/bash

# DRAG Evaluation Command Cheatsheet
# ==================================
# Each block below is an example invocation of eval_drag.py for a specific backend.
# Update SAVE_DIR / model names as needed before running.

BENCHMARK_PATH="benchmark.json"
SAVE_DIR="results"

# --- vLLM (external server) --------------------------------------------------
# Assumes a vLLM server exposing the model at http://localhost:1053/v1

python eval_drag.py --task inference --backend vllm \
  --model_path 'osunlp/GUI-Drag-7B' \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name "GUI-Drag-7B" \
  --batch_size 4


# --- Claude Computer-Use Agent -----------------------------------------------
python eval_drag.py \
  --task inference \
  --backend claude \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name claude_cua \
  --batch_size 1 \
  --max_concurrent 1


# --- OpenAI Operator ---------------------------------------------------------
python eval_drag.py \
  --task inference \
  --backend operator \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name operator_preview \
  --batch_size 16 \
  --max_concurrent 16


# --- UI TARS (UITAR backend) -------------------------------------------------

python eval_drag.py \
  --task inference \
  --backend uitar \
  --base_url http://localhost:1054/v1 \
  --model_name ui-tars \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name uitar_15
