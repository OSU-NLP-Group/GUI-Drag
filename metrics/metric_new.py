import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

Point = Tuple[float, float]
BBox = List[float]
OcrResult = Dict[str, Dict[str, object]]

DEFAULT_THRESHOLDS = [3.0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def threshold_key(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")

def point_in_bbox(point: Point, bbox: BBox) -> bool:
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def distance_point_to_bbox(point: Point, bbox: BBox) -> float:
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    if point_in_bbox(point, bbox):
        return 0.0
    dx = max(0.0, max(x_min - x, x - x_max))
    dy = max(0.0, max(y_min - y, y - y_max))
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Layout construction (copied/trimmed from original metric implementation)
# ---------------------------------------------------------------------------

def detect_lines_simple(bbox_infos: List[Dict], y_line_threshold: float) -> Dict[int, List[Dict]]:
    sorted_bboxes = sorted(bbox_infos, key=lambda x: x["y_center"])
    lines: Dict[int, List[Dict]] = {}
    current_line = 0
    for i, bbox in enumerate(sorted_bboxes):
        if i == 0:
            lines[current_line] = [bbox]
            bbox["line"] = current_line
        else:
            current_centers = [b["y_center"] for b in lines[current_line]]
            avg_center = sum(current_centers) / len(current_centers)
            if abs(bbox["y_center"] - avg_center) <= y_line_threshold:
                lines[current_line].append(bbox)
                bbox["line"] = current_line
            else:
                current_line += 1
                lines[current_line] = [bbox]
                bbox["line"] = current_line
    for line_bboxes in lines.values():
        line_bboxes.sort(key=lambda info: info["x_min"])
    return lines


def segment_lines_simple(lines: Dict[int, List[Dict]], x_gap_threshold: float) -> Dict[int, List[List[Dict]]]:
    segments: Dict[int, List[List[Dict]]] = {}
    for line_num, line_bboxes in lines.items():
        current_segment: List[Dict] = []
        grouped: List[List[Dict]] = []
        for i, bbox in enumerate(line_bboxes):
            if i == 0:
                current_segment.append(bbox)
                continue
            prev_bbox = line_bboxes[i - 1]
            if bbox["x_min"] - prev_bbox["x_max"] > x_gap_threshold:
                grouped.append(current_segment)
                current_segment = [bbox]
            else:
                current_segment.append(bbox)
        if current_segment:
            grouped.append(current_segment)
        segments[line_num] = grouped
    return segments


def calculate_segment_bounds_simple(segment: List[Dict]) -> Optional[Dict]:
    if not segment:
        return None
    return {
        "left_x": min(box["x_min"] for box in segment),
        "right_x": max(box["x_max"] for box in segment),
        "top_y_avg": sum(box["y_min"] for box in segment) / len(segment),
        "bottom_y_avg": sum(box["y_max"] for box in segment) / len(segment),
        "line_num": segment[0]["line"],
        "bboxes": segment,
    }


def should_merge_segments_simple(seg1: Dict, seg2: Dict, x_align_threshold: float, y_gap_threshold: float) -> bool:
    if abs(seg1["left_x"] - seg2["left_x"]) > x_align_threshold:
        return False
    if seg2["line_num"] == seg1["line_num"]:
        return False
    if seg2["line_num"] > seg1["line_num"]:
        y_gap = abs(seg1["bottom_y_avg"] - seg2["top_y_avg"])
    else:
        y_gap = abs(seg2["bottom_y_avg"] - seg1["top_y_avg"])
    return y_gap <= y_gap_threshold


def build_text_spans_simple(line_segments: Dict[int, List[List[Dict]]], x_align_threshold: float, y_gap_threshold: float) -> Dict[int, List[Dict]]:
    all_segments: List[Dict] = []
    for segments in line_segments.values():
        for segment in segments:
            bounds = calculate_segment_bounds_simple(segment)
            if bounds:
                all_segments.append(bounds)
    segment_groups: List[List[Dict]] = []
    used = set()
    for i, seg1 in enumerate(all_segments):
        if i in used:
            continue
        group = [seg1]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, seg2 in enumerate(all_segments):
                if j in used:
                    continue
                for existing in group:
                    if should_merge_segments_simple(existing, seg2, x_align_threshold, y_gap_threshold):
                        group.append(seg2)
                        used.add(j)
                        changed = True
                        break
        segment_groups.append(group)
    spans: Dict[int, List[Dict]] = {}
    for span_id, segments in enumerate(segment_groups):
        span_bboxes: List[Dict] = []
        for segment in segments:
            span_bboxes.extend(segment["bboxes"])
        spans[span_id] = span_bboxes
    return spans


def build_layout(
    ocr_results: OcrResult,
    y_line_threshold: float,
    x_gap_threshold: float,
    x_align_threshold: float,
    y_gap_threshold: float,
):
    bbox_infos: List[Dict] = []
    for idx, data in ocr_results.items():
        coord = data["coordinate"]
        bbox_infos.append(
            {
                "idx": idx,
                "text": data.get("text", ""),
                "bbox": coord,
                "x_min": coord[0],
                "y_min": coord[1],
                "x_max": coord[2],
                "y_max": coord[3],
                "x_center": (coord[0] + coord[2]) / 2,
                "y_center": (coord[1] + coord[3]) / 2,
            }
        )
    lines = detect_lines_simple(bbox_infos, y_line_threshold)
    line_segments = segment_lines_simple(lines, x_gap_threshold)
    text_spans = build_text_spans_simple(line_segments, x_align_threshold, y_gap_threshold)

    ordered_bboxes: List[Dict] = []
    span_line_bboxes: Dict[Tuple[int, int], List[Dict]] = {}
    for span_id in sorted(text_spans.keys()):
        span_bboxes = text_spans[span_id]
        lines_in_span: Dict[int, List[Dict]] = defaultdict(list)
        for bbox in span_bboxes:
            bbox["span"] = span_id
            lines_in_span[bbox["line"]].append(bbox)
        for line_num in sorted(lines_in_span.keys()):
            line_list = sorted(lines_in_span[line_num], key=lambda info: info["x_min"])
            ordered_bboxes.extend(line_list)
            span_line_bboxes[(span_id, line_num)] = line_list
    bbox_lookup = {info["idx"]: info for info in bbox_infos}
    return {
        "bbox_infos": bbox_infos,
        "ordered_bboxes": ordered_bboxes,
        "span_line_bboxes": span_line_bboxes,
        "bbox_lookup": bbox_lookup,
    }


# ---------------------------------------------------------------------------
# Coordinate assignment helpers
# ---------------------------------------------------------------------------

def find_closest_bbox_with_line_priority(point: Point, bbox_infos: List[Dict]) -> Optional[Dict]:
    x, y = point
    lines_with_bboxes: Dict[int, List[Dict]] = defaultdict(list)
    for bbox_info in bbox_infos:
        line_num = bbox_info.get("line")
        lines_with_bboxes[line_num].append(bbox_info)

    best_line = None
    min_y_distance = float("inf")
    for line_num, line_bboxes in lines_with_bboxes.items():
        avg_y_min = sum(b["y_min"] for b in line_bboxes) / len(line_bboxes)
        avg_y_max = sum(b["y_max"] for b in line_bboxes) / len(line_bboxes)
        if avg_y_min <= y <= avg_y_max:
            y_distance = 0.0
        else:
            y_distance = min(abs(y - avg_y_min), abs(y - avg_y_max))
        if y_distance < min_y_distance:
            min_y_distance = y_distance
            best_line = line_num

    if best_line is not None:
        line_bboxes = lines_with_bboxes[best_line]
        min_x_distance = float("inf")
        closest: Optional[Dict] = None
        for bbox_info in line_bboxes:
            x_min, _, x_max, _ = bbox_info["bbox"]
            if x_min <= x <= x_max:
                x_distance = 0.0
            else:
                x_distance = min(abs(x - x_min), abs(x - x_max))
            if x_distance < min_x_distance:
                min_x_distance = x_distance
                closest = bbox_info
        if closest is not None:
            return {
                "method": "line_priority",
                "assigned_bbox": closest,
                "distance_to_bbox": distance_point_to_bbox(point, closest["bbox"]),
            }

    # Fallback to global closest
    min_distance = float("inf")
    closest_global: Optional[Dict] = None
    for bbox_info in bbox_infos:
        distance = distance_point_to_bbox(point, bbox_info["bbox"])
        if distance < min_distance:
            min_distance = distance
            closest_global = bbox_info
    if closest_global is None:
        return None
    return {
        "method": "global_closest",
        "assigned_bbox": closest_global,
        "distance_to_bbox": min_distance,
    }


def assign_coordinate(point: Point, layout: Dict) -> Dict:
    bbox_infos = layout["bbox_infos"]
    bbox_lookup = layout["bbox_lookup"]

    for bbox_id, bbox_info in bbox_lookup.items():
        if point_in_bbox(point, bbox_info["bbox"]):
            return {
                "coordinate": point,
                "method": "inside",
                "assigned_bbox": bbox_info,
                "distance_to_bbox": 0.0,
            }

    fallback = find_closest_bbox_with_line_priority(point, bbox_infos)
    if fallback is None:
        return {
            "coordinate": point,
            "method": "unassigned",
            "assigned_bbox": None,
            "distance_to_bbox": None,
        }
    fallback.update({"coordinate": point})
    return fallback


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def compute_edge_midpoints(bbox: BBox) -> Tuple[Point, Point]:
    x_min, y_min, x_max, y_max = bbox
    y_mid = (y_min + y_max) / 2
    return (x_min, y_mid), (x_max, y_mid)


def evaluate_role(
    coord: Point,
    role: str,
    assignment: Dict,
    gt_info: Dict,
    span_line_bboxes: Dict[Tuple[int, int], List[Dict]],
    thresholds: List[float],
) -> Dict:
    result = {
        "coordinate": coord,
        "role": role,
        "assigned_bbox_id": None,
        "assigned_line": None,
        "assigned_span": None,
        "assignment_method": assignment.get("method") if assignment else None,
        "boundary_case": False,
        "within_gt_bbox": False,
        "distance_to_target": None,
        "threshold_flags": {threshold_key(th): False for th in thresholds},
        "reason": None,
    }

    if not assignment or not assignment.get("assigned_bbox"):
        result["reason"] = "no_assignment"
        return result

    assigned_bbox = assignment["assigned_bbox"]
    result["assigned_bbox_id"] = assigned_bbox["idx"]
    result["assigned_line"] = assigned_bbox.get("line")
    result["assigned_span"] = assigned_bbox.get("span")

    if gt_info is None:
        result["reason"] = "missing_ground_truth"
        return result

    gt_span = gt_info.get("span")
    gt_line = gt_info.get("line")
    line_key = (gt_span, gt_line)

    if result["assigned_span"] != gt_span:
        result["reason"] = "span_mismatch"
        return result
    if result["assigned_line"] != gt_line:
        result["reason"] = "line_mismatch"
        return result
    if line_key not in span_line_bboxes:
        result["reason"] = "missing_span_line"
        return result

    line_bboxes = span_line_bboxes[line_key]
    line_x_min = min(b["x_min"] for b in line_bboxes)
    line_x_max = max(b["x_max"] for b in line_bboxes)
    line_y_min = min(b["y_min"] for b in line_bboxes)
    line_y_max = max(b["y_max"] for b in line_bboxes)
    line_height = max(1.0, line_y_max - line_y_min)
    vertical_margin = max(5.0, line_height * 0.1)
    within_vertical = (line_y_min - vertical_margin) <= coord[1] <= (line_y_max + vertical_margin)

    if role == "start" and within_vertical and coord[0] <= line_x_min:
        result["boundary_case"] = True
        for th in thresholds:
            result["threshold_flags"][threshold_key(th)] = True
        return result
    if role == "end" and within_vertical and coord[0] >= line_x_max:
        result["boundary_case"] = True
        for th in thresholds:
            result["threshold_flags"][threshold_key(th)] = True
        return result

    gt_bbox = gt_info.get("coordinate")
    if gt_bbox is None:
        result["reason"] = "missing_gt_coordinate"
        return result

    inside_gt = point_in_bbox(coord, gt_bbox)
    result["within_gt_bbox"] = inside_gt
    if not inside_gt:
        result["reason"] = "outside_gt_bbox"
        return result

    left_mid, right_mid = compute_edge_midpoints(gt_bbox)
    target_point = left_mid if role == "start" else right_mid
    distance = math.dist(coord, target_point)
    result["distance_to_target"] = distance

    passed_any = False
    for th in thresholds:
        key = threshold_key(th)
        if distance <= th:
            result["threshold_flags"][key] = True
            passed_any = True
    if not passed_any:
        result["reason"] = "distance_above_threshold"
    return result


def calculate_bbox_distances(ordered_bboxes: List[Dict], predicted_ids: List[Optional[str]], ground_truth_ids: List[str]) -> List[int]:
    order_index = {info["idx"]: idx for idx, info in enumerate(ordered_bboxes)}
    distances: List[int] = []
    if len(ground_truth_ids) == 1:
        gt_pos = order_index.get(ground_truth_ids[0], -1)
        for pid in predicted_ids:
            pred_pos = order_index.get(pid, -1)
            if pred_pos == -1 or gt_pos == -1:
                distances.append(-1)
            else:
                distances.append(abs(pred_pos - gt_pos))
    elif len(ground_truth_ids) == 2:
        for pid, gid in zip(predicted_ids, ground_truth_ids):
            pred_pos = order_index.get(pid, -1)
            gt_pos = order_index.get(gid, -1)
            if pred_pos == -1 or gt_pos == -1:
                distances.append(-1)
            else:
                distances.append(abs(pred_pos - gt_pos))
    return distances


def evaluate_prediction(
    predicted_coords: List[Point],
    ground_truth_bbox_ids: List[str],
    ocr_results: OcrResult,
    thresholds: Optional[List[float]] = None,
    y_line_threshold: float = 10.0,
    x_gap_threshold: float = 20.0,
    x_align_threshold: float = 40.0,
    y_gap_threshold: float = 200.0,
) -> Dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS

    layout = build_layout(
        ocr_results,
        y_line_threshold=y_line_threshold,
        x_gap_threshold=x_gap_threshold,
        x_align_threshold=x_align_threshold,
        y_gap_threshold=y_gap_threshold,
    )

    bbox_lookup = layout["bbox_lookup"]
    ordered_bboxes = layout["ordered_bboxes"]
    span_line_bboxes = layout["span_line_bboxes"]

    ground_truth_infos: List[Dict] = []
    for gt_id in ground_truth_bbox_ids:
        info = bbox_lookup.get(gt_id)
        if not info:
            continue
        gt_entry = {
            "bbox_id": gt_id,
            "coordinate": info["bbox"],
            "text": info.get("text", ""),
            "x_center": info["x_center"],
            "y_center": info["y_center"],
            "line": info.get("line"),
            "span": info.get("span"),
        }
        ground_truth_infos.append(gt_entry)

    if not ground_truth_infos:
        raise ValueError("No valid ground truth bboxes found in OCR results.")

    ground_truth_infos.sort(key=lambda item: (item["x_center"], item["y_center"]))
    sorted_ground_truth_ids = [info["bbox_id"] for info in ground_truth_infos]

    ordered_predicted = sorted(predicted_coords, key=lambda p: (p[0], p[1]))
    assignments: List[Dict] = []
    for coord in ordered_predicted:
        assignments.append(assign_coordinate(coord, layout))

    while len(assignments) < 2:
        assignments.append({"coordinate": None, "method": "missing", "assigned_bbox": None, "distance_to_bbox": None})

    start_gt = ground_truth_infos[0]
    end_gt = ground_truth_infos[-1] if len(ground_truth_infos) > 1 else ground_truth_infos[0]

    start_eval = evaluate_role(
        ordered_predicted[0],
        "start",
        assignments[0],
        start_gt,
        span_line_bboxes,
        thresholds,
    )
    end_eval = evaluate_role(
        ordered_predicted[1],
        "end",
        assignments[1],
        end_gt,
        span_line_bboxes,
        thresholds,
    )

    zero_bbox_flags = {}
    for th in thresholds:
        key = threshold_key(th)
        zero_bbox_flags[key] = (
            start_eval["threshold_flags"].get(key, False)
            and end_eval["threshold_flags"].get(key, False)
        )

    predicted_ids = [assignment.get("assigned_bbox", {}).get("idx") for assignment in assignments[:2]]
    bbox_distances = calculate_bbox_distances(ordered_bboxes, predicted_ids, sorted_ground_truth_ids)
    bbox_distances = [dist for dist in bbox_distances if dist is not None]

    geometric_distances = []
    for eval_record in (start_eval, end_eval):
        geometric_distances.append(eval_record.get("distance_to_target"))
    valid_geom = [d for d in geometric_distances if d is not None]
    mean_bbox_distance = (
        sum(d for d in bbox_distances if d >= 0) / len([d for d in bbox_distances if d >= 0])
        if any(d >= 0 for d in bbox_distances)
        else None
    )
    mean_geometric_distance = (
        sum(valid_geom) / len(valid_geom) if valid_geom else None
    )

    return {
        "ordered_predicted_coords": ordered_predicted,
        "sorted_ground_truth_bbox_ids": sorted_ground_truth_ids,
        "assignments": assignments,
        "start_evaluation": start_eval,
        "end_evaluation": end_eval,
        "zero_bbox_flags": {"thresholds": zero_bbox_flags},
        "bbox_distances": bbox_distances,
        "geometric_distances": geometric_distances,
        "mean_bbox_distance": mean_bbox_distance,
        "mean_geometric_distance": mean_geometric_distance,
    }


# ---------------------------------------------------------------------------
# Batch evaluation entry points
# ---------------------------------------------------------------------------

def evaluate_all_items(
    results_dir: str,
    benchmark_path: str,
    output_dir: str,
    thresholds: Optional[List[float]] = None,
    verbose: bool = False,
    y_line_threshold: float = 10.0,
    x_gap_threshold: float = 20.0,
    x_align_threshold: float = 40.0,
    y_gap_threshold: float = 200.0,
):
    os.makedirs(output_dir, exist_ok=True)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    benchmark_dict = {item["item_id"]: item for item in benchmark_data}

    success_count = 0
    error_count = 0
    error_log: List[Dict] = []

    for item_id, benchmark_item in tqdm(benchmark_dict.items(), desc="Evaluating"):
        results_path = os.path.join(results_dir, f"{item_id}.json")
        if not os.path.exists(results_path):
            error_log.append({"item_id": item_id, "error": "result_file_missing"})
            error_count += 1
            continue
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                result_json = json.load(f)
        except Exception as exc:  # pragma: no cover - I/O guard
            error_log.append({"item_id": item_id, "error": f"result_json_error: {exc}"})
            error_count += 1
            continue

        processed = result_json.get("processed_results")
        if not processed:
            evaluation_result = {
                "item_id": item_id,
                "status": False,
                "error": "processed_results empty",
                "bbox_distances": [],
                "geometric_distances": [],
                "mean_bbox_distance": None,
                "mean_geometric_distance": None,
                "predicted_coords": None,
                "ground_truth_bbox_ids": benchmark_item.get("ids_of_the_bboxes", []),
                "zero_bbox_flags": {"thresholds": {}},
                "start_evaluation": {},
                "end_evaluation": {},
            }
            with open(os.path.join(output_dir, f"{item_id}.json"), "w", encoding="utf-8") as f_out:
                json.dump(evaluation_result, f_out, ensure_ascii=False, indent=2)
            error_log.append({"item_id": item_id, "error": "processed_results empty"})
            error_count += 1
            continue

        first_entry = processed[0]
        if not isinstance(first_entry, dict):
            evaluation_result = {
                "item_id": item_id,
                "status": False,
                "error": "invalid_processed_entry_type",
                "bbox_distances": [],
                "geometric_distances": [],
                "mean_bbox_distance": None,
                "mean_geometric_distance": None,
                "predicted_coords": first_entry,
                "ground_truth_bbox_ids": benchmark_item.get("ids_of_the_bboxes", []),
                "zero_bbox_flags": {"thresholds": {}},
                "start_evaluation": {},
                "end_evaluation": {},
            }
            with open(os.path.join(output_dir, f"{item_id}.json"), "w", encoding="utf-8") as f_out:
                json.dump(evaluation_result, f_out, ensure_ascii=False, indent=2)
            error_log.append({"item_id": item_id, "error": "invalid_processed_entry_type"})
            error_count += 1
            continue

        predicted_coords = first_entry.get("coordinates")
        if not predicted_coords or len(predicted_coords) < 2:
            evaluation_result = {
                "item_id": item_id,
                "status": False,
                "error": "invalid_predicted_coordinates",
                "bbox_distances": [],
                "geometric_distances": [],
                "mean_bbox_distance": None,
                "mean_geometric_distance": None,
                "predicted_coords": predicted_coords,
                "ground_truth_bbox_ids": benchmark_item.get("ids_of_the_bboxes", []),
                "zero_bbox_flags": {"thresholds": {}},
                "start_evaluation": {},
                "end_evaluation": {},
            }
            with open(os.path.join(output_dir, f"{item_id}.json"), "w", encoding="utf-8") as f_out:
                json.dump(evaluation_result, f_out, ensure_ascii=False, indent=2)
            error_log.append({"item_id": item_id, "error": "invalid_predicted_coordinates"})
            error_count += 1
            continue

        ocr_path = benchmark_item.get("parsed_path")
        if not ocr_path or not os.path.exists(ocr_path):
            error_log.append({"item_id": item_id, "error": "ocr_missing"})
            error_count += 1
            continue
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_results = json.load(f)

        ground_truth_ids = benchmark_item.get("ids_of_the_bboxes", [])
        try:
            evaluation = evaluate_prediction(
                predicted_coords=predicted_coords,
                ground_truth_bbox_ids=ground_truth_ids,
                ocr_results=ocr_results,
                thresholds=thresholds,
                y_line_threshold=y_line_threshold,
                x_gap_threshold=x_gap_threshold,
                x_align_threshold=x_align_threshold,
                y_gap_threshold=y_gap_threshold,
            )
        except Exception as exc:
            error_log.append({"item_id": item_id, "error": f"evaluation_failed: {exc}"})
            error_count += 1
            continue

        evaluation_result = {
            "item_id": item_id,
            "status": True,
            "bbox_distances": evaluation["bbox_distances"],
            "geometric_distances": evaluation["geometric_distances"],
            "mean_bbox_distance": evaluation["mean_bbox_distance"],
            "mean_geometric_distance": evaluation["mean_geometric_distance"],
            "predicted_coords": predicted_coords,
            "ground_truth_bbox_ids": evaluation["sorted_ground_truth_bbox_ids"],
            "zero_bbox_flags": evaluation["zero_bbox_flags"],
            "thresholds": [float(th) for th in (thresholds or DEFAULT_THRESHOLDS)],
            "start_evaluation": evaluation["start_evaluation"],
            "end_evaluation": evaluation["end_evaluation"],
            "assignments": evaluation["assignments"],
        }
        with open(os.path.join(output_dir, f"{item_id}.json"), "w", encoding="utf-8") as f_out:
            json.dump(evaluation_result, f_out, ensure_ascii=False, indent=2)
        success_count += 1

    if error_log:
        with open(os.path.join(output_dir, "error_log.json"), "w", encoding="utf-8") as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)

    print("\nEvaluation completed")
    print(f"Success: {success_count}")
    print(f"Failed : {error_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate drag metrics with revised zero-bbox logic.")
    parser.add_argument("--results_dir", required=True, help="Directory with model results (per item JSON).")
    parser.add_argument("--benchmark_path", required=True, help="Path to benchmark.json")
    parser.add_argument("--output_dir", required=True, help="Directory to write metric outputs")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        help="Custom pixel thresholds for zero-bbox evaluation (default: 3.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--y_line_threshold", type=float, default=10.0)
    parser.add_argument("--x_gap_threshold", type=float, default=20.0)
    parser.add_argument("--x_align_threshold", type=float, default=40.0)
    parser.add_argument("--y_gap_threshold", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = args.thresholds if args.thresholds else None
    evaluate_all_items(
        results_dir=args.results_dir,
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
        thresholds=thresholds,
        verbose=args.verbose,
        y_line_threshold=args.y_line_threshold,
        x_gap_threshold=args.x_gap_threshold,
        x_align_threshold=args.x_align_threshold,
        y_gap_threshold=args.y_gap_threshold,
    )


if __name__ == "__main__":
    main()
