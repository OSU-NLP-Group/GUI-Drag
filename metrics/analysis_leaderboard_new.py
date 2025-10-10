#!/usr/bin/env python3
"""Generate leaderboard reports for metric_new.py outputs."""

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

DEFAULT_THRESHOLDS = [3.0]
BREAKDOWN_FIELDS = {
    "granularity": "Granularity",
    "category": "Category",
    "form": "Form",
    "has_drag_expression": "Drag Expression",
}


def threshold_key(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def parse_file_path(file_path: str) -> Dict[str, Optional[str]]:
    filename = os.path.basename(file_path)
    is_source_file = "_label_" in filename

    result = {
        "source_file": None,
        "page": None,
        "form": None,
        "category": None,
        "granularity": None,
        "is_source_file": is_source_file,
    }

    if not is_source_file:
        return result

    parts = filename.split("_label_")
    if len(parts) < 2:
        return result

    base_part, category_part = parts[0], parts[1]

    import re

    match = re.search(r"_A(\d+)\.(\d+)", base_part)
    if match:
        result["page"] = int(match.group(1))
        result["form"] = int(match.group(2))
        result["source_file"] = re.sub(r"_A\d+\.\d+.*$", "", base_part)

    category_words = category_part.split("_")
    if category_words:
        result["category"] = category_words[0]

    granularity_match = re.search(r"_\d+_([\w-]+)_top\d+", category_part)
    if granularity_match:
        result["granularity"] = granularity_match.group(1)

    return result


def load_benchmark_data(benchmark_path: str) -> Dict[str, Dict]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        item_id = item.get("item_id")
        if item_id:
            mapping[item_id] = item
    return mapping


def load_dense_text_ids(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    raise ValueError("dense_text_ids file should contain a list of item IDs")


def format_bool(value: Optional[bool]) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def init_breakdown_stats(threshold_keys: List[str]) -> Dict:
    return {
        "total_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "bbox_distances": [],
        "start_boundary": 0,
        "end_boundary": 0,
        "zero_counts": {key: 0 for key in threshold_keys},
    }


def finalize_stats(stats: Dict, threshold_keys: List[str]) -> Dict:
    success = stats["success_count"]
    total = stats["total_count"]
    bbox_distances = stats["bbox_distances"]

    mean_bbox = statistics.mean(bbox_distances) if bbox_distances else None
    median_bbox = statistics.median(bbox_distances) if bbox_distances else None
    std_bbox = (
        statistics.stdev(bbox_distances)
        if bbox_distances and len(bbox_distances) > 1
        else 0.0
    )

    zero_ratio_total = {
        key: (stats["zero_counts"][key] / total if total else None)
        for key in threshold_keys
    }
    zero_ratio_success = {
        key: (stats["zero_counts"][key] / success if success else None)
        for key in threshold_keys
    }

    return {
        "total_count": total,
        "success_count": success,
        "failure_count": stats["failure_count"],
        "success_rate": (success / total * 100.0) if total else 0.0,
        "mean_bbox_distance": mean_bbox,
        "median_bbox_distance": median_bbox,
        "std_bbox_distance": std_bbox,
        "start_boundary": stats["start_boundary"],
        "end_boundary": stats["end_boundary"],
        "zero_counts": stats["zero_counts"],
        "zero_ratio_total": zero_ratio_total,
        "zero_ratio_success": zero_ratio_success,
    }


def analyze_model(
    model_dir: str,
    benchmark_data: Dict[str, Dict],
    thresholds: List[float],
    dense_text_ids: Optional[set],
) -> Dict:
    threshold_keys = [threshold_key(t) for t in thresholds]

    records = []
    files = [f for f in os.listdir(model_dir) if f.endswith(".json") and f != "error_log.json"]

    for filename in files:
        path = os.path.join(model_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        item_id = data.get("item_id") or filename[:-5]
        benchmark_item = benchmark_data.get(item_id, {})
        metadata = parse_file_path(benchmark_item.get("grounded_path", ""))

        expression = benchmark_item.get("expression")
        has_drag_expression = None
        if isinstance(expression, str):
            has_drag_expression = "drag" in expression.lower()

        zero_flags = {}
        for key in threshold_keys:
            zero_flags[key] = bool(
                data.get("zero_bbox_flags", {})
                .get("thresholds", {})
                .get(key)
            )

        start_eval = data.get("start_evaluation", {}) or {}
        end_eval = data.get("end_evaluation", {}) or {}

        records.append(
            {
                "item_id": item_id,
                "status": bool(data.get("status")),
                "mean_bbox_distance": data.get("mean_bbox_distance"),
                "zero_flags": zero_flags,
                "start_boundary": bool(start_eval.get("boundary_case")),
                "end_boundary": bool(end_eval.get("boundary_case")),
                "granularity": metadata.get("granularity") or "Unknown",
                "category": metadata.get("category") or "Unknown",
                "form": metadata.get("form") or "Unknown",
                "has_drag_expression": format_bool(has_drag_expression),
            }
        )

    summary = aggregate_records(records, threshold_keys)
    dense_summary = None
    non_dense_summary = None
    if dense_text_ids:
        dense_records = [rec for rec in records if rec["item_id"] in dense_text_ids]
        if dense_records:
            dense_summary = aggregate_records(dense_records, threshold_keys)
        non_dense_records = [rec for rec in records if rec["item_id"] not in dense_text_ids]
        if non_dense_records:
            non_dense_summary = aggregate_records(non_dense_records, threshold_keys)

    summary.update(
        {
            "model_name": os.path.basename(model_dir.rstrip("/")),
            "threshold_keys": threshold_keys,
            "dense_summary": dense_summary,
            "dense_breakdowns": dense_summary.get("breakdowns") if dense_summary else None,
            "non_dense_summary": non_dense_summary,
        }
    )
    return summary


def aggregate_records(records: List[Dict], threshold_keys: List[str]) -> Dict:
    stats = init_breakdown_stats(threshold_keys)
    breakdowns_raw = {
        field: defaultdict(lambda: init_breakdown_stats(threshold_keys))
        for field in BREAKDOWN_FIELDS.keys()
    }

    for record in records:
        stats["total_count"] += 1
        if record["status"]:
            stats["success_count"] += 1
            if record["mean_bbox_distance"] is not None:
                stats["bbox_distances"].append(record["mean_bbox_distance"])
            if record["start_boundary"]:
                stats["start_boundary"] += 1
            if record["end_boundary"]:
                stats["end_boundary"] += 1
            for key in threshold_keys:
                if record["zero_flags"].get(key):
                    stats["zero_counts"][key] += 1
        else:
            stats["failure_count"] += 1

        for field in BREAKDOWN_FIELDS.keys():
            value = record.get(field) or "Unknown"
            bstats = breakdowns_raw[field][value]
            bstats["total_count"] += 1
            if record["status"]:
                bstats["success_count"] += 1
                if record["mean_bbox_distance"] is not None:
                    bstats["bbox_distances"].append(record["mean_bbox_distance"])
                if record["start_boundary"]:
                    bstats["start_boundary"] += 1
                if record["end_boundary"]:
                    bstats["end_boundary"] += 1
                for key in threshold_keys:
                    if record["zero_flags"].get(key):
                        bstats["zero_counts"][key] += 1
            else:
                bstats["failure_count"] += 1

    summary = finalize_stats(stats, threshold_keys)

    breakdowns = {}
    for field, values in breakdowns_raw.items():
        breakdowns[field] = {
            value: finalize_stats(bstats, threshold_keys)
            for value, bstats in values.items()
            if bstats["total_count"] > 0
        }
    summary["breakdowns"] = breakdowns
    return summary



def render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(values: List[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_count_ratio(count: int, ratio_total: Optional[float], ratio_success: Optional[float]) -> str:
    if ratio_success is not None:
        return f"{count} ({ratio_success * 100:.2f}% success)"
    return str(count)


def model_table_rows(models: List[Dict], threshold_keys: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    display_key = threshold_keys[0] if threshold_keys else None
    for rank, model in enumerate(models, start=1):
        row = [
            str(rank),
            model["model_name"],
            str(model["total_count"]),
            str(model["success_count"]),
            f"{model['success_rate']:.2f}%",
            f"{model['mean_bbox_distance']:.2f}" if model["mean_bbox_distance"] is not None else "N/A",
        ]
        if display_key:
            count = model["zero_counts"].get(display_key, 0)
            ratio_success = model["zero_ratio_success"].get(display_key)
            row.append(format_count_ratio(count, None, ratio_success))
        rows.append(row)
    return rows


def prepare_model_summary(summary: Dict, threshold_keys: List[str]) -> Dict:
    sr_key = threshold_keys[0] if threshold_keys else None
    sr_count = summary["zero_counts"].get(sr_key, 0) if sr_key else 0
    sr_ratio = summary["zero_ratio_total"].get(sr_key) if sr_key else None
    sr_percent = (sr_ratio * 100.0) if sr_ratio is not None else 0.0
    return {
        "model_name": summary["model_name"],
        "total_count": summary["total_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "success_rate": summary["success_rate"],
        "sr_count": sr_count,
        "sr_percent": sr_percent,
        "mean_bbox_distance": summary["mean_bbox_distance"],
        "median_bbox_distance": summary["median_bbox_distance"],
        "std_bbox_distance": summary["std_bbox_distance"],
        "start_boundary": summary["start_boundary"],
        "end_boundary": summary["end_boundary"],
        "zero_counts": summary["zero_counts"],
        "zero_ratio_total": summary["zero_ratio_total"],
        "zero_ratio_success": summary["zero_ratio_success"],
    }


def generate_report(
    model_summaries: List[Dict],
    threshold_keys: List[str],
    output_path: Optional[str],
    non_dense_summaries: Optional[List[Dict]] = None,
    dense_summaries: Optional[List[Dict]] = None,
    breakdown_leaderboards: Optional[Dict[str, Dict[str, List[Dict]]]] = None,
    include_table: bool = True,
    title: str = "DRAG Evaluation Leaderboard (SR / B-Dist)",
    display_threshold_keys: Optional[List[str]] = None,
) -> str:
    headers = [
        "Rank",
        "Model Name",
        "Total",
        "Success",
        "Rate%",
        "B-Dist",
        "SR (<=3px)",
    ]

    lines: List[str] = []

    def append_section(section_title: str, summaries: List[Dict], leading_blank: bool = True) -> None:
        if not summaries:
            return
        if leading_blank:
            lines.append("")
        lines.append(section_title)
        lines.append("=" * 100)
        total_items = sum(m["total_count"] for m in summaries)
        total_success = sum(m["success_count"] for m in summaries)
        total_failure = sum(m["failure_count"] for m in summaries)
        total_rate = (total_success / total_items * 100.0) if total_items else 0.0
        lines.extend(
            [
                f"Total items: {total_items}",
                f"Successful evaluations: {total_success}",
                f"Failed evaluations: {total_failure}",
                f"Success rate: {total_rate:.2f}%",
                "",
                render_table(headers, model_table_rows(summaries, threshold_keys)),
            ]
        )
        lines.append("")

    if include_table:
        append_section(title, model_summaries, leading_blank=False)
        append_section("NON-DENSE TEXT (EXCLUDING DENSE IDS)", non_dense_summaries)
        append_section("DENSE TEXT ONLY LEADERBOARD", dense_summaries)
    else:
        lines.extend([title, "=" * 100, ""])

    if breakdown_leaderboards:
        for field, leaderboards in breakdown_leaderboards.items():
            if not leaderboards:
                continue
            display_name = BREAKDOWN_FIELDS.get(field, field)
            lines.append(f"{display_name.upper()} BREAKDOWN")
            lines.append("=" * 100)
            for value in sorted(leaderboards.keys(), key=lambda v: str(v)):
                summaries = leaderboards[value]
                lines.append(f"{value}")
                rows = model_table_rows(summaries, threshold_keys)
                lines.append(render_table(headers, rows))
                lines.append("")

    report = "\n".join(lines).strip() + "\n"
    if output_path and include_table:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report


def collect_breakdown_leaderboards(models: List[Dict], threshold_keys: List[str]) -> Dict[str, Dict[str, List[Dict]]]:
    result: Dict[str, Dict[str, List[Dict]]] = {
        field: defaultdict(list) for field in BREAKDOWN_FIELDS.keys()
    }
    for model in models:
        breakdowns = model.get("breakdowns", {})
        for field, values in breakdowns.items():
            if field not in result:
                continue
            for value, stats in values.items():
                summary = prepare_model_summary(stats | {"model_name": model["model_name"]}, threshold_keys)
                result[field][value].append(summary)
    final_result: Dict[str, Dict[str, List[Dict]]] = {}
    for field, value_dict in result.items():
        final_result[field] = {}
        for value, summaries in value_dict.items():
            if not summaries:
                continue
            summaries.sort(key=lambda s: s["success_rate"], reverse=True)
            final_result[field][value] = summaries
    return final_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metric_new.py outputs with detailed reporting.")
    parser.add_argument("--metrics_dir", required=True, help="Path to metric results directory")
    parser.add_argument("--benchmark_path", required=True, help="Path to benchmark.json")
    parser.add_argument("--output_txt", help="Path to write leaderboard report")
    parser.add_argument("--output_json", help="Path to write structured JSON output")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        help="Override pixel thresholds (default: 3.0)",
    )
    parser.add_argument(
        "--display_thresholds",
        nargs="*",
        type=float,
        help="(Deprecated) kept for backwards compatibility; SR/B-Dist table ignores this option",
    )
    parser.add_argument(
        "--include_models",
        nargs="*",
        help="Order-sensitive list of model directory names to include",
    )
    parser.add_argument("--include_dense_text", action="store_true", help="Add dense text only analysis")
    parser.add_argument(
        "--dense_text_ids_path",
        default="/fs/scratch/PAA0201/lzy37ld/OSWorld-G/annotation_dense_text/dense_text_ids.json",
        help="Path to dense text id list",
    )
    args = parser.parse_args()

    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    threshold_keys = [threshold_key(t) for t in thresholds]

    if args.display_thresholds:
        display_threshold_keys = []
        for value in args.display_thresholds:
            key = threshold_key(value)
            if key not in display_threshold_keys:
                display_threshold_keys.append(key)
    else:
        display_threshold_keys = list(threshold_keys)

    benchmark_data = load_benchmark_data(args.benchmark_path)

    include_models = args.include_models if args.include_models else None

    all_model_dirs = [
        os.path.join(args.metrics_dir, name)
        for name in os.listdir(args.metrics_dir)
        if os.path.isdir(os.path.join(args.metrics_dir, name))
    ]
    all_model_dirs.sort()

    missing_models: List[str] = []
    if include_models:
        selected_dirs = []
        for name in include_models:
            path = os.path.join(args.metrics_dir, name)
            if os.path.isdir(path):
                selected_dirs.append(path)
            else:
                missing_models.append(name)
    else:
        selected_dirs = all_model_dirs

    dense_text_ids = None
    if args.include_dense_text:
        dense_text_ids = load_dense_text_ids(args.dense_text_ids_path)

    model_summaries = []
    dense_summaries = []
    non_dense_summaries = []
    breakdown_source = []
    dense_breakdown_source = []

    order_map = {name: idx for idx, name in enumerate(include_models)} if include_models else None

    for model_dir in selected_dirs:
        summary = analyze_model(model_dir, benchmark_data, thresholds, dense_text_ids)
        if order_map is not None:
            summary["_order_index"] = order_map.get(summary["model_name"], float("inf"))
        model_summaries.append(summary)
        breakdown_source.append(summary)
        if summary.get("dense_summary"):
            dense_summary = summary["dense_summary"].copy()
            dense_summary["model_name"] = summary["model_name"]
            dense_summaries.append(dense_summary)
            dense_breakdown_source.append(summary)
        if summary.get("non_dense_summary"):
            non_dense_summary = summary["non_dense_summary"].copy()
            non_dense_summary["model_name"] = summary["model_name"]
            non_dense_summaries.append(non_dense_summary)

    def sort_summaries(summaries: List[Dict]) -> None:
        if order_map is not None:
            summaries.sort(key=lambda m: m.get("_order_index", float("inf")))
        else:
            summaries.sort(
                key=lambda m: (
                    float("inf") if m["mean_bbox_distance"] is None else m["mean_bbox_distance"],
                    m["model_name"],
                )
            )

    sort_summaries(model_summaries)
    sort_summaries(dense_summaries)
    sort_summaries(non_dense_summaries)

    for summary in model_summaries:
        summary.pop("_order_index", None)
    for summary in dense_summaries:
        summary.pop("_order_index", None)
    for summary in non_dense_summaries:
        summary.pop("_order_index", None)

    leaderboard_rows = [prepare_model_summary(summary, threshold_keys) for summary in model_summaries]
    dense_rows = [prepare_model_summary(summary, threshold_keys) for summary in dense_summaries]
    non_dense_rows = [prepare_model_summary(summary, threshold_keys) for summary in non_dense_summaries]

    breakdown_leaderboards = collect_breakdown_leaderboards(model_summaries, threshold_keys)
    if dense_summaries:
        dense_breakdown_leaderboards = collect_breakdown_leaderboards(
            [
                {
                    "model_name": summary["model_name"],
                    "breakdowns": summary.get("dense_breakdowns", {}),
                }
                for summary in dense_breakdown_source
            ],
            threshold_keys,
        )
    else:
        dense_breakdown_leaderboards = None

    report = generate_report(
        leaderboard_rows,
        threshold_keys,
        args.output_txt,
        non_dense_rows if non_dense_rows else None,
        dense_rows if dense_rows else None,
        breakdown_leaderboards,
        include_table=True,
        display_threshold_keys=display_threshold_keys,
    )

    has_dense_breakdown = False
    if dense_breakdown_leaderboards:
        has_dense_breakdown = any(dense_breakdown_leaderboards[field] for field in dense_breakdown_leaderboards)

    if has_dense_breakdown:
        # Append dense breakdown tables to same report file if requested
        dense_section = generate_report(
            dense_rows,
            threshold_keys,
            None,
            non_dense_summaries=None,
            dense_summaries=None,
            breakdown_leaderboards=dense_breakdown_leaderboards,
            include_table=False,
            title="DENSE TEXT BREAKDOWN",
            display_threshold_keys=display_threshold_keys,
        )
        if args.output_txt:
            with open(args.output_txt, "a", encoding="utf-8") as f:
                f.write("\n" + dense_section)
        report += "\n" + dense_section

    if missing_models:
        print("Warning: missing metric directories:", ", ".join(missing_models))

    if args.output_json:
        output_payload = {
            "thresholds": threshold_keys,
            "display_thresholds": display_threshold_keys,
            "models": model_summaries,
            "dense_models": dense_summaries,
            "non_dense_models": non_dense_summaries,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

    if not args.output_txt:
        print(report)


if __name__ == "__main__":
    main()
