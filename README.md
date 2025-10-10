# DRAG Evaluation Bundle (to_submis)

This folder contains a clean copy of the evaluation + metric tooling used in the paper. Everything is self-contained so you can run the drag benchmark, regenerate metrics (SR / B-Dist), and produce the leaderboard tables referenced in the manuscript.

## Layout

- `evaluation/eval_drag.py` – unified evaluation driver supporting **vLLM**, **Claude CUA**, **OpenAI Operator**, and **UI-TARS**.
- `evaluation/cli_run_drag.sh` – cheat sheet with the exact commands we ran for each backend (edit paths / model names as needed).
- `metrics/metric_new.py` – per-item metric computation (pixel threshold fixed at 3.0 by default).
- `metrics/analysis_leaderboard_new.py` – aggregates metric outputs and reports **SR** (success rate) and **B-Dist** (mean bbox distance) only.
- `metrics/cli_run_drag_metric_new.sh` – helper script to batch-run the metrics pipeline and export leaderboard reports.
- `metrics/metric_reports/` (created on demand) – target folder for aggregated reports.

## Requirements

- Python 3.10+
- Core deps: `Pillow`, `tqdm`, `matplotlib` (optional for visualization), `openai`, `anthropic` (Bedrock client), `boto3` if you use Bedrock via AWS.
- For vLLM runs: a running vLLM server exposing the model through an OpenAI-compatible endpoint (`http://host:port/v1`).

### Credentials

| Backend | Required environment variables |
| --- | --- |
| Claude CUA | `AWS_REGION`, `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` (for Bedrock access) |
| OpenAI Operator | `OPENAI_API_KEY` |
| UI-TARS | whatever endpoint key the serving stack expects (the script assumes base URL + model name) |

## Example Workflow (GUI-Drag-3B)

### 1. Run the evaluation

```bash
cd /fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/evaluation

BENCHMARK_PATH="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/benchmark.json"
SAVE_DIR="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/results"

python eval_drag.py --task inference --backend vllm \
  --model_path osunlp/GUI-Drag-3B \
  --benchmark "$BENCHMARK_PATH" \
  --save_dir "$SAVE_DIR" \
  --model_save_name GUI-Drag-3B \
  --batch_size 4
```

This populates `to_submis/results/GUI-Drag-3B_vllm_thinking_false/` with one JSON per benchmark item.

### 2. Compute SR / B-Dist metrics and leaderboard rows

```bash
cd /fs/scratch/PAA0201/lzy37ld/OSWorld-G/to_submis/metrics
bash cli_run_drag_metric_new.sh
```

- Per-item metrics are written to `to_submis/metric_results/GUI-Drag-3B_vllm_thinking_false/`.
- Aggregated reports (SR / B-Dist only) appear in `to_submis/metric_reports/leaderboard_report.txt` and `leaderboard_results.json`.

### 3. (Optional) Adjust configuration for additional models

If you evaluate more models, add their result-folder names to `INCLUDE_MODELS` inside `metrics/cli_run_drag_metric_new.sh` (the order in the array is the order used in the leaderboard table).

## Notes

- The evaluation script still supports visualization (`--task viz`) using the same folder layout as inference.
- Threshold defaults are pinned to 3 pixels everywhere, matching the manuscript settings.
- Feel free to copy additional assets (plots, notebooks, etc.) into this folder if you need them for the submission; just keep originals untouched outside `to_submis/`.
