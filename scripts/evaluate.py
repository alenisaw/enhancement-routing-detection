from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from enhancement_routing.methods import rule_based_router, train_random_forest_router
from enhancement_routing.metrics import (
    build_policy_comparison,
    compute_oracle_actions,
    compute_per_image_scores,
    evaluate_preprocessing_results,
    export_runtime_overhead,
)
from enhancement_routing.utils import ensure_dir, load_config, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preprocessing policies and routers.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    tables_dir = ensure_dir(paths["tables_dir"])
    models_dir = ensure_dir(paths.get("models_dir", "results/models"))

    detections_path = tables_dir / "detections.csv"
    features_path = tables_dir / "quality_features.csv"
    if not detections_path.exists():
        raise FileNotFoundError(f"Missing {detections_path}. Run scripts/run_experiments.py first.")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run scripts/run_experiments.py first.")

    detections = pd.read_csv(detections_path)
    features = pd.read_csv(features_path)

    preprocessing_results = evaluate_preprocessing_results(
        detections=detections,
        annotations_path=paths["subset_annotations"],
        tables_dir=tables_dir,
    )
    print(f"Saved preprocessing metrics: {tables_dir / 'preprocessing_results.csv'}")

    per_image_scores = compute_per_image_scores(
        detections=detections,
        annotations_path=paths["subset_annotations"],
        iou_threshold=float(config["evaluation"].get("iou_threshold", 0.50)),
    )
    write_csv(tables_dir / "per_image_scores.csv", per_image_scores)

    primary_score = config["evaluation"].get("primary_score", "per_image_recall50")
    oracle_actions = compute_oracle_actions(per_image_scores, primary_score=primary_score)
    write_csv(tables_dir / "oracle_actions.csv", oracle_actions)

    router = features[["image_id", "condition"]].copy()
    feature_cols = [c for c in features.columns if c not in {"image_id", "file_name", "condition", "image_path"}]
    router["rule_action"] = [
        rule_based_router(row[feature_cols].to_dict(), reject_score=float(config["evaluation"].get("reject_score", 0.15)))
        for _, row in features.iterrows()
    ]

    router_training = features.merge(oracle_actions[["image_id", "condition", "oracle_action"]], on=["image_id", "condition"], how="inner")
    rf_enabled = bool(config["evaluation"].get("router", {}).get("use_random_forest", True))
    min_rf_samples = 4
    if rf_enabled and len(router_training) >= min_rf_samples:
        rf_pred, test_acc = train_random_forest_router(
            router_training,
            feature_columns=feature_cols,
            config=config,
            model_path=models_dir / "random_forest_router.joblib",
        )
        router = router.merge(rf_pred, on=["image_id", "condition"], how="left")
        print(f"Random Forest router test accuracy: {test_acc:.3f}")
    elif rf_enabled:
        print(f"Skipping Random Forest router: only {len(router_training)} training samples (need >= {min_rf_samples}).")

    write_csv(tables_dir / "router_predictions.csv", router)

    actions = list(config["preprocessing"].get("actions", ["none", "gamma", "clahe", "retinex"]))
    policy_comparison = build_policy_comparison(
        per_image_scores=per_image_scores,
        oracle_actions=oracle_actions,
        router_predictions=router,
        actions=actions,
        primary_score=primary_score,
    )
    write_csv(tables_dir / "policy_comparison.csv", policy_comparison)
    export_runtime_overhead(detections, tables_dir)

    print("Saved evaluation tables:")
    for name in [
        "oracle_actions.csv",
        "router_predictions.csv",
        "policy_comparison.csv",
        "runtime_overhead.csv",
    ]:
        print(f"- {tables_dir / name}")


if __name__ == "__main__":
    main()
