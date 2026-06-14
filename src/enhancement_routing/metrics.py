from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, read_json, require_columns, write_csv, write_json


def detections_to_coco_json(detections: pd.DataFrame, path: str | Path) -> list[dict[str, Any]]:
    valid = detections[(detections["category_id"] >= 0) & (~detections["is_reject_action"].astype(bool))]
    rows = []
    for r in valid.itertuples(index=False):
        rows.append(
            {
                "image_id": int(r.image_id),
                "category_id": int(r.category_id),
                "bbox": [float(r.x), float(r.y), float(r.w), float(r.h)],
                "score": float(r.score),
            }
        )
    write_json(path, rows)
    return rows


def evaluate_coco_group(
    detections: pd.DataFrame,
    annotations_path: str | Path,
    output_json: str | Path,
) -> dict[str, float]:
    """Evaluate one condition/action group with pycocotools."""
    detections_to_coco_json(detections, output_json)
    if detections[(detections["category_id"] >= 0) & (~detections["is_reject_action"].astype(bool))].empty:
        return {
            "mAP50_95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "recall": 0.0,
            "mean_confidence": 0.0,
            "detections": 0,
        }
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("Install pycocotools to compute COCO metrics.") from exc

    coco_gt = COCO(str(annotations_path))
    coco_dt = coco_gt.loadRes(str(output_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(detections["image_id"].dropna().astype(int).unique().tolist())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    valid = detections[(detections["category_id"] >= 0) & (~detections["is_reject_action"].astype(bool))]
    return {
        "mAP50_95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "recall": float(stats[8]),
        "mean_confidence": float(valid["score"].mean()) if not valid.empty else 0.0,
        "detections": int(len(valid)),
    }


def evaluate_preprocessing_results(
    detections: pd.DataFrame,
    annotations_path: str | Path,
    tables_dir: str | Path,
) -> pd.DataFrame:
    require_columns(detections, ["image_id", "condition", "action", "category_id", "score"], "detections")
    tables_dir = ensure_dir(tables_dir)
    temp_dir = ensure_dir(tables_dir / "coco_eval_json")
    rows = []
    for (condition, action), group in detections.groupby(["condition", "action"]):
        output_json = temp_dir / f"predictions_{condition}_{action}.json"
        metrics = evaluate_coco_group(group, annotations_path, output_json)
        runtime_ms = float(group.groupby("image_id")["runtime_ms"].max().mean()) if "runtime_ms" in group else 0.0
        rows.append({"condition": condition, "action": action, **metrics, "runtime_ms_per_image": runtime_ms})
    out = pd.DataFrame(rows).sort_values(["condition", "action"])
    write_csv(tables_dir / "preprocessing_results.csv", out)
    return out


def _xywh_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def compute_per_image_scores(
    detections: pd.DataFrame,
    annotations_path: str | Path,
    iou_threshold: float = 0.50,
) -> pd.DataFrame:
    coco = read_json(annotations_path)
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt_by_image[int(ann["image_id"])].append(ann)

    rows = []
    grouped = detections.groupby(["image_id", "condition", "action"], dropna=False)
    for (image_id, condition, action), preds in grouped:
        image_id = int(image_id)
        gts = gt_by_image.get(image_id, [])
        valid_preds = preds[(preds["category_id"] >= 0) & (~preds["is_reject_action"].astype(bool))]
        valid_preds = valid_preds.sort_values("score", ascending=False)
        matched_gt: set[int] = set()
        true_pos = 0
        for pred in valid_preds.itertuples(index=False):
            pbox = np.array([pred.x, pred.y, pred.w, pred.h], dtype=float)
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gts):
                if idx in matched_gt or int(gt["category_id"]) != int(pred.category_id):
                    continue
                iou = _xywh_iou(pbox, np.array(gt["bbox"], dtype=float))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt.add(best_idx)
                true_pos += 1
        gt_count = len(gts)
        recall50 = true_pos / gt_count if gt_count else 0.0
        mean_conf = float(valid_preds["score"].mean()) if not valid_preds.empty else 0.0
        rows.append(
            {
                "image_id": image_id,
                "condition": condition,
                "action": action,
                "gt_count": gt_count,
                "true_positives_iou50": true_pos,
                "per_image_recall50": recall50,
                "mean_confidence": mean_conf,
                "detections": len(valid_preds),
                "runtime_ms": float(preds["runtime_ms"].max()) if "runtime_ms" in preds else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_oracle_actions(per_image_scores: pd.DataFrame, primary_score: str = "per_image_recall50") -> pd.DataFrame:
    candidates = per_image_scores[per_image_scores["action"] != "reject"].copy()
    candidates = candidates.sort_values(
        ["image_id", "condition", primary_score, "mean_confidence", "runtime_ms"],
        ascending=[True, True, False, False, True],
    )
    oracle = candidates.groupby(["image_id", "condition"], as_index=False).head(1)
    oracle = oracle.rename(columns={"action": "oracle_action", primary_score: "oracle_score"})
    return oracle[["image_id", "condition", "oracle_action", "oracle_score", "mean_confidence", "runtime_ms"]]


def build_policy_comparison(
    per_image_scores: pd.DataFrame,
    oracle_actions: pd.DataFrame,
    router_predictions: pd.DataFrame,
    actions: list[str],
    primary_score: str = "per_image_recall50",
) -> pd.DataFrame:
    base = per_image_scores[["image_id", "condition", "action", primary_score, "runtime_ms"]].copy()
    rows = []

    none_scores = base[base["action"] == "none"][["image_id", "condition", primary_score]].rename(columns={primary_score: "none_score"})
    oracle_scores = oracle_actions[["image_id", "condition", "oracle_action", "oracle_score"]]

    def add_policy(policy: str, frame: pd.DataFrame, action_col: str) -> None:
        merged = frame.merge(base, left_on=["image_id", "condition", action_col], right_on=["image_id", "condition", "action"], how="left")
        merged = merged.merge(none_scores, on=["image_id", "condition"], how="left")
        merged = merged.merge(oracle_scores, on=["image_id", "condition"], how="left")
        score = merged[primary_score].fillna(0.0)
        none_score = merged["none_score"].fillna(0.0)
        oracle_score = merged["oracle_score"].fillna(0.0)
        non_none = merged[action_col] != "none"
        preprocessed = non_none & (merged[action_col] != "reject")
        harmful = preprocessed & (score < none_score)
        rows.append(
            {
                "policy": policy,
                "mean_score": float(score.mean()),
                "mean_none_score": float(none_score.mean()),
                "policy_gain": float(score.mean() - none_score.mean()),
                "oracle_gap": float(oracle_score.mean() - score.mean()),
                "harmful_enhancement_rate": float(harmful.sum() / max(preprocessed.sum(), 1)),
                "reject_rate": float((merged[action_col] == "reject").mean()),
                "routing_accuracy": float((merged[action_col] == merged["oracle_action"]).mean()),
                "runtime_ms_per_image": float(merged["runtime_ms"].fillna(0.0).mean()),
                "n": int(len(merged)),
            }
        )

    keys = per_image_scores[["image_id", "condition"]].drop_duplicates()
    for action in actions:
        fixed = keys.copy()
        fixed["chosen_action"] = action
        add_policy(f"always_{action}", fixed, "chosen_action")

    oracle_frame = oracle_actions[["image_id", "condition", "oracle_action"]].rename(columns={"oracle_action": "chosen_action"})
    add_policy("oracle", oracle_frame, "chosen_action")

    if "rule_action" in router_predictions.columns:
        rule = router_predictions[["image_id", "condition", "rule_action"]].rename(columns={"rule_action": "chosen_action"})
        add_policy("rule_based_router", rule, "chosen_action")
    if "rf_action" in router_predictions.columns:
        rf = router_predictions[["image_id", "condition", "rf_action"]].rename(columns={"rf_action": "chosen_action"})
        add_policy("random_forest_router", rf, "chosen_action")

    return pd.DataFrame(rows).sort_values("mean_score", ascending=False)


def export_runtime_overhead(detections: pd.DataFrame, tables_dir: str | Path) -> pd.DataFrame:
    grouped = detections.groupby(["condition", "action", "image_id"], as_index=False)["runtime_ms"].max()
    out = (
        grouped.groupby(["condition", "action"], as_index=False)
        .agg(runtime_ms_mean=("runtime_ms", "mean"), runtime_ms_median=("runtime_ms", "median"), runtime_ms_std=("runtime_ms", "std"))
        .fillna(0.0)
    )
    write_csv(Path(tables_dir) / "runtime_overhead.csv", out)
    return out
