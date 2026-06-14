from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .methods import apply_degradation, read_rgb, save_rgb
from .utils import ensure_dir, normalize_path, read_json, set_seed, write_csv, write_json


def _valid_image_ids(coco: dict[str, Any], images_dir: Path, require_annotations: bool) -> list[int]:
    annotated_ids = {ann["image_id"] for ann in coco.get("annotations", [])}
    out: list[int] = []
    for image in coco.get("images", []):
        image_id = int(image["id"])
        if require_annotations and image_id not in annotated_ids:
            continue
        if (images_dir / image["file_name"]).exists():
            out.append(image_id)
    return out


def select_coco_subset(
    annotations_path: str | Path,
    images_dir: str | Path,
    subset_size: int,
    seed: int,
    require_annotations: bool = True,
) -> dict[str, Any]:
    """Create a deterministic COCO annotation subset.

    The subset is selected by original image_id. This prevents leakage when several
    degraded copies are generated from the same image.
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    coco = read_json(annotations_path)
    set_seed(seed)

    valid_ids = _valid_image_ids(coco, images_dir, require_annotations)
    if len(valid_ids) < subset_size:
        subset_ids = sorted(valid_ids)
    else:
        import random

        subset_ids = sorted(random.sample(valid_ids, subset_size))

    subset_set = set(subset_ids)
    images = [img for img in coco["images"] if int(img["id"]) in subset_set]
    annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in subset_set]

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }


def generate_condition_images(
    subset: dict[str, Any],
    images_dir: str | Path,
    work_dir: str | Path,
    conditions: list[dict[str, Any]],
    copy_clean_images: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    images_dir = Path(images_dir)
    work_dir = Path(work_dir)
    generated_root = ensure_dir(work_dir / "images")
    rows: list[dict[str, Any]] = []

    for image in tqdm(subset["images"], desc="Generating degraded variants"):
        image_id = int(image["id"])
        file_name = image["file_name"]
        original_path = images_dir / file_name
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix.lower() or ".jpg"

        for condition in conditions:
            condition_name = condition["name"]
            out_dir = ensure_dir(generated_root / condition_name)
            out_name = f"{stem}_{condition_name}{suffix}"
            out_path = out_dir / out_name

            if overwrite or not out_path.exists():
                if condition.get("type") == "clean" and copy_clean_images:
                    shutil.copy2(original_path, out_path)
                else:
                    img = read_rgb(original_path)
                    degraded = apply_degradation(img, condition)
                    save_rgb(out_path, degraded)

            rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "condition": condition_name,
                    "condition_type": condition.get("type", condition_name),
                    "image_path": normalize_path(out_path),
                    "original_path": normalize_path(original_path),
                    "width": image.get("width"),
                    "height": image.get("height"),
                }
            )

    return pd.DataFrame(rows)


def prepare_dataset(config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    seed = int(config["project"].get("seed", 42))
    dataset_cfg = config["dataset"]
    paths = config["paths"]

    subset_size = int(limit or dataset_cfg["subset_size"])
    subset = select_coco_subset(
        annotations_path=paths["coco_annotations"],
        images_dir=paths["coco_images_dir"],
        subset_size=subset_size,
        seed=seed,
        require_annotations=bool(dataset_cfg.get("require_annotations", True)),
    )

    subset_annotations_path = Path(paths["subset_annotations"])
    write_json(subset_annotations_path, subset)

    manifest = generate_condition_images(
        subset=subset,
        images_dir=paths["coco_images_dir"],
        work_dir=paths["work_dir"],
        conditions=config["conditions"],
        copy_clean_images=bool(dataset_cfg.get("copy_clean_images", True)),
        overwrite=bool(config.get("runtime", {}).get("overwrite", False)),
    )
    write_csv(paths["manifest_csv"], manifest)

    table_dir = ensure_dir(paths["tables_dir"])
    summary = pd.DataFrame(
        [
            {
                "subset_size_original_images": len(subset["images"]),
                "generated_image_variants": len(manifest),
                "conditions": manifest["condition"].nunique(),
                "annotations": len(subset.get("annotations", [])),
                "seed": seed,
            }
        ]
    )
    write_csv(table_dir / "dataset_summary.csv", summary)
    return manifest
