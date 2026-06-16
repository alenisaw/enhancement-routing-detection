from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .utils import ensure_dir

# Ultralytics COCO80 class indices to COCO category IDs.
YOLO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(path), bgr)


def _adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_degradation(image: np.ndarray, condition: dict[str, Any]) -> np.ndarray:
    kind = condition.get("type", condition.get("name", "clean"))
    out = image.astype(np.float32)

    if kind == "clean":
        return image.copy()
    if kind == "low_light":
        out = out * float(condition.get("factor", 0.4))
        return _adjust_gamma(np.clip(out, 0, 255).astype(np.uint8), float(condition.get("gamma", 1.6)))
    if kind == "low_light_noise":
        out = out * float(condition.get("factor", 0.4))
        out = _adjust_gamma(np.clip(out, 0, 255).astype(np.uint8), float(condition.get("gamma", 1.6))).astype(np.float32)
        noise = np.random.normal(0, float(condition.get("noise_sigma", 12.0)), out.shape)
        return np.clip(out + noise, 0, 255).astype(np.uint8)
    if kind == "low_light_jpeg":
        out = out * float(condition.get("factor", 0.4))
        out = _adjust_gamma(np.clip(out, 0, 255).astype(np.uint8), float(condition.get("gamma", 1.6)))
        return _jpeg_roundtrip(out, int(condition.get("jpeg_quality", 25)))
    if kind == "gaussian_blur":
        k = int(condition.get("kernel_size", 7))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(image, (k, k), float(condition.get("sigma", 1.8)))
    if kind == "contrast_reduction":
        alpha = float(condition.get("alpha", 0.55))
        beta = float(condition.get("beta", 22))
        return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported degradation type: {kind}")


def preprocess_none(image: np.ndarray, _: dict[str, Any] | None = None) -> np.ndarray:
    return image.copy()


def preprocess_gamma(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    params = params or {}
    return _adjust_gamma(image, float(params.get("gamma", 0.62)))


def preprocess_clahe(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    params = params or {}
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    grid = tuple(params.get("tile_grid_size", [8, 8]))
    clahe = cv2.createCLAHE(clipLimit=float(params.get("clip_limit", 2.0)), tileGridSize=grid)
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_retinex(image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    """Simple multi-scale Retinex-like enhancement.

    This is intentionally classical and deterministic. It is not a learned enhancer.
    """
    params = params or {}
    sigma_list = params.get("sigma_list", [15, 80, 250])
    img = image.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), float(sigma)) + 1.0
        retinex += np.log(img) - np.log(blur)
    retinex /= max(len(sigma_list), 1)

    out = np.zeros_like(retinex)
    for c in range(3):
        channel = retinex[:, :, c]
        lo, hi = np.percentile(channel, [1, 99])
        if math.isclose(float(hi), float(lo)):
            out[:, :, c] = 0
        else:
            out[:, :, c] = (channel - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_preprocessing(image: np.ndarray, action: str, config: dict[str, Any]) -> np.ndarray:
    preprocessing_cfg = config.get("preprocessing", {})
    if action == "none":
        return preprocess_none(image)
    if action == "gamma":
        return preprocess_gamma(image, preprocessing_cfg.get("gamma", {}))
    if action == "clahe":
        return preprocess_clahe(image, preprocessing_cfg.get("clahe", {}))
    if action == "retinex":
        return preprocess_retinex(image, preprocessing_cfg.get("retinex", {}))
    if action == "zero_dce":
        raise NotImplementedError("Zero-DCE wrapper is optional. Configure an external command before enabling it.")
    if action == "reject":
        return image.copy()
    raise ValueError(f"Unsupported preprocessing action: {action}")


def extract_no_reference_features(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mean_intensity = float(np.mean(gray_f))
    median_intensity = float(np.median(gray_f))
    dark_pixel_ratio = float(np.mean(gray_f < 50))
    rms_contrast = float(np.std(gray_f))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    cdf = np.cumsum(hist)
    p05 = int(np.searchsorted(cdf, 0.05))
    p95 = int(np.searchsorted(cdf, 0.95))
    histogram_spread = float(p95 - p05)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    tenengrad = float(np.mean(grad_mag**2))
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
    gradient_magnitude = float(np.mean(grad_mag))

    blur = cv2.GaussianBlur(gray_f, (3, 3), 0)
    residual = gray_f - blur
    noise_estimate = float(np.std(residual))
    local_var = cv2.blur(gray_f**2, (5, 5)) - cv2.blur(gray_f, (5, 5)) ** 2
    local_variance_estimate = float(np.mean(np.maximum(local_var, 0)))

    blockiness = _blockiness_proxy(gray_f)
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    saturation_std = float(np.std(hsv[:, :, 1]))

    return {
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "dark_pixel_ratio": dark_pixel_ratio,
        "rms_contrast": rms_contrast,
        "histogram_spread": histogram_spread,
        "laplacian_variance": laplacian_variance,
        "tenengrad": tenengrad,
        "edge_density": edge_density,
        "gradient_magnitude": gradient_magnitude,
        "noise_estimate": noise_estimate,
        "local_variance_estimate": local_variance_estimate,
        "blockiness_proxy": blockiness,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
    }


def _blockiness_proxy(gray_f: np.ndarray) -> float:
    h, w = gray_f.shape
    if h < 16 or w < 16:
        return 0.0
    vertical_boundaries = gray_f[:, 8::8]
    vertical_prev = gray_f[:, 7::8][:, : vertical_boundaries.shape[1]]
    horizontal_boundaries = gray_f[8::8, :]
    horizontal_prev = gray_f[7::8, :][: horizontal_boundaries.shape[0], :]
    vb = np.mean(np.abs(vertical_boundaries - vertical_prev)) if vertical_boundaries.size else 0.0
    hb = np.mean(np.abs(horizontal_boundaries - horizontal_prev)) if horizontal_boundaries.size else 0.0
    return float((vb + hb) / 2.0)


def rule_based_router(features: dict[str, float], reject_score: float = 0.15) -> str:
    """Interpretable routing policy based only on no-reference image features."""
    mean_i = features["mean_intensity"]
    dark = features["dark_pixel_ratio"]
    blur = features["laplacian_variance"]
    noise = features["noise_estimate"]
    contrast = features["rms_contrast"]
    block = features["blockiness_proxy"]

    if blur < 35:
        return "reject"
    if dark > 0.62 and noise > 13:
        return "reject"
    if block > 18 and mean_i < 80:
        return "gamma"
    if mean_i < 70 and dark > 0.35 and noise < 13:
        return "gamma"
    if contrast < 35 and blur >= 35:
        return "clahe"
    if mean_i < 95 and contrast < 55 and noise < 16:
        return "retinex"
    return "none"


def _extract_row(row_data: tuple) -> dict[str, float | int | str]:
    image_id, file_name, condition, image_path = row_data
    image = read_rgb(image_path)
    features = extract_no_reference_features(image)
    return {
        "image_id": int(image_id),
        "file_name": file_name,
        "condition": condition,
        "image_path": image_path,
        **features,
    }


def build_feature_table(
    manifest: pd.DataFrame,
    n_jobs: int = -1,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 500,
) -> pd.DataFrame:
    saved_rows: list[dict] = []
    done_ids: set[int] = set()

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        saved_df = pd.read_csv(checkpoint_path)
        done_ids = set(saved_df["image_id"].tolist())
        saved_rows = saved_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} images already done, {len(manifest) - len(done_ids)} remaining.")

    pending = [
        (int(row.image_id), row.file_name, row.condition, row.image_path)
        for row in manifest.itertuples(index=False)
        if int(row.image_id) not in done_ids
    ]

    if not pending:
        return pd.DataFrame(saved_rows)

    new_rows: list[dict] = []
    results = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
        joblib.delayed(_extract_row)(t) for t in pending
    )
    for i, row in enumerate(tqdm(results, total=len(pending), desc="Extracting features"), 1):
        new_rows.append(row)
        if checkpoint_path is not None and i % checkpoint_interval == 0:
            pd.DataFrame(saved_rows + new_rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(saved_rows + new_rows)


def run_yolo_inference(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    actions: list[str],
    limit: int | None = None,
) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics to run detector experiments.") from exc

    detector_cfg = config["detector"]
    device = detector_cfg.get("device", "auto")
    if device == "auto":
        device = None
    model = YOLO(detector_cfg.get("weights", "yolov8n.pt"))

    if limit is not None:
        manifest = manifest.head(limit)

    rows: list[dict[str, Any]] = []
    conf = float(detector_cfg.get("confidence", 0.001))
    iou = float(detector_cfg.get("iou", 0.7))
    imgsz = int(detector_cfg.get("image_size", 640))
    max_det = int(detector_cfg.get("max_det", 300))

    for item in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Detector inference"):
        base_image = read_rgb(item.image_path)
        for action in actions:
            if action == "reject":
                rows.append(
                    {
                        "image_id": int(item.image_id),
                        "file_name": item.file_name,
                        "condition": item.condition,
                        "action": action,
                        "category_id": -1,
                        "score": 0.0,
                        "x": 0.0,
                        "y": 0.0,
                        "w": 0.0,
                        "h": 0.0,
                        "runtime_ms": 0.0,
                        "is_reject_action": True,
                    }
                )
                continue

            start = time.perf_counter()
            processed = apply_preprocessing(base_image, action, config)
            # Ultralytics accepts numpy arrays in OpenCV/BGR channel order.
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            predict_kwargs = {
                "source": processed_bgr,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = device
            pred = model.predict(**predict_kwargs)[0]
            runtime_ms = (time.perf_counter() - start) * 1000.0

            if pred.boxes is None or len(pred.boxes) == 0:
                rows.append(
                    {
                        "image_id": int(item.image_id),
                        "file_name": item.file_name,
                        "condition": item.condition,
                        "action": action,
                        "category_id": -1,
                        "score": 0.0,
                        "x": 0.0,
                        "y": 0.0,
                        "w": 0.0,
                        "h": 0.0,
                        "runtime_ms": runtime_ms,
                        "is_reject_action": False,
                    }
                )
                continue

            boxes_xyxy = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes_xyxy, scores, classes, strict=False):
                x1, y1, x2, y2 = box.tolist()
                category_id = YOLO80_TO_COCO91[cls] if 0 <= cls < len(YOLO80_TO_COCO91) else int(cls)
                rows.append(
                    {
                        "image_id": int(item.image_id),
                        "file_name": item.file_name,
                        "condition": item.condition,
                        "action": action,
                        "category_id": int(category_id),
                        "score": float(score),
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                        "runtime_ms": runtime_ms,
                        "is_reject_action": False,
                    }
                )
    return pd.DataFrame(rows)


def train_random_forest_router(
    router_data: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    model_path: str | Path,
) -> tuple[pd.DataFrame, float]:
    router_cfg = config["evaluation"]["router"]
    groups = router_data["image_id"].astype(str) + "_" + router_data["condition"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=float(router_cfg.get("train_fraction", 0.75)),
        random_state=int(router_cfg.get("random_state", 42)),
    )
    train_idx, test_idx = next(splitter.split(router_data, groups=groups))
    clf = RandomForestClassifier(
        n_estimators=int(router_cfg.get("n_estimators", 200)),
        random_state=int(router_cfg.get("random_state", 42)),
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(router_data.iloc[train_idx][feature_columns], router_data.iloc[train_idx]["oracle_action"])
    pred = clf.predict(router_data[feature_columns])
    test_pred = clf.predict(router_data.iloc[test_idx][feature_columns])
    test_acc = float(accuracy_score(router_data.iloc[test_idx]["oracle_action"], test_pred))
    out = router_data[["image_id", "condition"]].copy()
    out["rf_action"] = pred
    ensure_dir(Path(model_path).parent)
    joblib.dump({"model": clf, "feature_columns": feature_columns, "test_accuracy": test_acc}, model_path)
    return out, test_acc
