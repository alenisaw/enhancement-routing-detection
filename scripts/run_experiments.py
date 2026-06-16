from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from enhancement_routing.methods import build_feature_table, run_yolo_inference
from enhancement_routing.utils import ensure_dir, load_config, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object detector under preprocessing actions.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Limit manifest rows for smoke tests.")
    parser.add_argument("--skip-detector", action="store_true", help="Only compute feature table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    tables_dir = ensure_dir(paths["tables_dir"])

    manifest_path = Path(paths["manifest_csv"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run scripts/prepare_data.py first.")
    manifest = pd.read_csv(manifest_path)
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    feature_csv = tables_dir / "quality_features.csv"
    feature_table = build_feature_table(manifest, checkpoint_path=feature_csv)
    write_csv(feature_csv, feature_table)
    print(f"Saved no-reference features: {tables_dir / 'quality_features.csv'}")

    if args.skip_detector:
        return

    actions = list(config["preprocessing"].get("actions", ["none", "gamma", "clahe", "retinex"]))
    if "reject" not in actions:
        actions_with_reject = actions + ["reject"]
    else:
        actions_with_reject = actions

    detections = run_yolo_inference(manifest, config, actions=actions_with_reject, limit=None)
    write_csv(tables_dir / "detections.csv", detections)
    print(f"Saved detections: {tables_dir / 'detections.csv'}")


if __name__ == "__main__":
    main()
