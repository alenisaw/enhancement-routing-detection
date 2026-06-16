from __future__ import annotations

import argparse
from pathlib import Path

from enhancement_routing.data import prepare_dataset
from enhancement_routing.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COCO subset and degraded image variants.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to experiment YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset subset size for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    images_dir = Path(config["paths"]["coco_images_dir"])
    annotations_path = Path(config["paths"]["coco_annotations"])
    if not images_dir.exists():
        raise FileNotFoundError(
            f"COCO images directory not found: {images_dir}\n"
            "Download COCO val2017 images and place them at the path above."
        )
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"COCO annotations file not found: {annotations_path}\n"
            "Download instances_val2017.json and place it at the path above."
        )

    manifest = prepare_dataset(config, limit=args.limit)
    print(f"Prepared {manifest['image_id'].nunique()} original images and {len(manifest)} image variants.")
    print(f"Manifest: {Path(config['paths']['manifest_csv']).resolve()}")


if __name__ == "__main__":
    main()
