#!/bin/bash

# Metric Aggregation Helper
# =========================
# Runs metric_new.py for every model subdirectory and then produces
# a leaderboard summary (SR / B-Dist only).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARK_PATH="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/benchmark.json"
RESULTS_BASE_DIR="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/results"
METRICS_BASE_DIR="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/metric_results"
REPORT_DIR="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/metric_reports"

mkdir -p "$METRICS_BASE_DIR" "$REPORT_DIR"

# 1. Compute per-item metrics for each model result folder
for model_path in "$RESULTS_BASE_DIR"/*; do
  if [ -d "$model_path" ]; then
    model_name=$(basename "$model_path")
    echo "Evaluating metrics for $model_name"
    mkdir -p "$METRICS_BASE_DIR/$model_name"
    python "$SCRIPT_DIR/metric_new.py" \
      --results_dir "$model_path" \
      --benchmark_path "$BENCHMARK_PATH" \
      --output_dir "$METRICS_BASE_DIR/$model_name"
  fi
done

# 2. Aggregate into a leaderboard report (edit INCLUDE_MODELS as needed)
# Update this list to the models you want to appear (order matters)
INCLUDE_MODELS=(
  "GUI-Drag-3B_vllm_thinking_false"
  "GUI-Drag-7B_vllm_thinking_false"
)

python "$SCRIPT_DIR/analysis_leaderboard_new.py" \
  --metrics_dir "$METRICS_BASE_DIR" \
  --benchmark_path "$BENCHMARK_PATH" \
  --include_models "${INCLUDE_MODELS[@]}" \
  --include_dense_text \
  --output_txt "$REPORT_DIR/leaderboard_report.txt" \
  --output_json "$REPORT_DIR/leaderboard_results.json"

echo "Done. Reports written to $REPORT_DIR"
