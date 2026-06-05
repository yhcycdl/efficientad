"""Evaluate anomaly detection and threshold strategies on MVTec AD."""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    MODEL_CHOICES,
    build_engine,
    build_model,
    build_predict_dataset,
    normalize_model_name,
    path_value,
    resolve_checkpoint,
    scalar_value,
    tensor_to_numpy,
)
from data_config import (
    DATA_ROOT,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PERCENTILE,
    DEFAULT_SMOOTH_SIGMA,
    OUTPUTS_ROOT,
    categories_from_arg,
)
from postprocess import apply_threshold_strategy, best_f1_threshold, f1_binary, percentile_threshold, prepare_map, refine_mask, squeeze_map
from visualize import save_prediction_visuals


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class EvalSample:
    image_path: Path
    anomaly_map: np.ndarray
    score: float
    gt_label: int
    gt_mask: np.ndarray
    inference_ms: float


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_mask(mask_path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask_path is None or not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    mask = Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def gt_from_mvtec_path(image_path: Path, category_dir: Path, map_shape: tuple[int, int]) -> tuple[int, np.ndarray]:
    image_path = image_path.resolve()
    category_dir = category_dir.resolve()
    test_dir = category_dir / "test"
    try:
        rel = image_path.relative_to(test_dir)
    except ValueError:
        parts = image_path.parts
        try:
            category_idx = parts.index(category_dir.name)
            if parts[category_idx + 1] != "test":
                return 0, np.zeros(map_shape, dtype=np.uint8)
            rel = Path(*parts[category_idx + 2 :])
        except (ValueError, IndexError):
            return 0, np.zeros(map_shape, dtype=np.uint8)
    defect_type = rel.parts[0]
    if defect_type == "good":
        return 0, np.zeros(map_shape, dtype=np.uint8)
    mask_path = category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    return 1, resize_mask(mask_path, map_shape)


def predict_folder(
    model_name: str,
    checkpoint: Path,
    folder: Path,
    image_size: int,
    output_dir: Path,
) -> list[tuple[Path, np.ndarray, float, float]]:
    images = iter_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found under {folder}")

    dataset = build_predict_dataset(folder, image_size=image_size)
    model = build_model(model_name)
    engine = build_engine(default_root_dir=output_dir)
    start = time.perf_counter()
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=str(checkpoint))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if predictions is None:
        raise RuntimeError("Anomalib returned no predictions.")
    per_image_ms = elapsed_ms / max(len(predictions), 1)

    rows = []
    for idx, prediction in enumerate(predictions):
        image_path = path_value(getattr(prediction, "image_path", None))
        if image_path is None:
            image_path = images[min(idx, len(images) - 1)]
        anomaly_map = tensor_to_numpy(getattr(prediction, "anomaly_map", None))
        if anomaly_map is None:
            anomaly_map = tensor_to_numpy(getattr(prediction, "pred_mask", None))
        if anomaly_map is None:
            raise RuntimeError("Prediction does not contain anomaly_map or pred_mask.")
        score = scalar_value(getattr(prediction, "pred_score", None))
        if score is None:
            score = float(np.max(anomaly_map))
        rows.append((Path(image_path), anomaly_map, float(score), per_image_ms))
    return rows


def resize_float_map(anomaly_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    arr = squeeze_map(anomaly_map)
    try:
        import torch
        import torch.nn.functional as F

        tensor = torch.from_numpy(arr).float()[None, None]
        resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        return resized[0, 0].numpy().astype(np.float32)
    except Exception:
        # Fallback for environments without torch at import time. This path is only
        # for visualization-scale resizing, so preserving relative contrast is enough.
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx - mn < 1e-8:
            return np.zeros((h, w), dtype=np.float32) + mn
        normalized = (arr - mn) / (mx - mn)
        pil = Image.fromarray((normalized * 255).clip(0, 255).astype(np.uint8), mode="L")
        pil = pil.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(pil).astype(np.float32) / 255.0 * (mx - mn) + mn


def predict_folder_fused(
    model_name: str,
    checkpoint: Path,
    folder: Path,
    fusion_scales: list[int],
    output_dir: Path,
    smooth_sigma: float,
    per_image_normalize: bool,
) -> list[tuple[Path, np.ndarray, float, float]]:
    scale_predictions = [
        predict_folder(
            model_name=model_name,
            checkpoint=checkpoint,
            folder=folder,
            image_size=scale,
            output_dir=output_dir / f"predict_scale_{scale}",
        )
        for scale in fusion_scales
    ]
    if not scale_predictions:
        return []

    fused_rows: list[tuple[Path, np.ndarray, float, float]] = []
    for idx, base_prediction in enumerate(scale_predictions[0]):
        image_path = base_prediction[0]
        maps: list[np.ndarray] = []
        scores: list[float] = []
        inference_ms = 0.0
        for predictions in scale_predictions:
            pred_image_path, anomaly_map, score, pred_ms = predictions[idx]
            image_path = pred_image_path
            processed = prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=per_image_normalize)
            maps.append(processed)
            scores.append(float(score))
            inference_ms += float(pred_ms)

        base_shape = maps[0].shape
        resized_maps = [resize_float_map(item, base_shape) if item.shape != base_shape else item for item in maps]
        fused_map = np.mean(np.stack(resized_maps, axis=0), axis=0).astype(np.float32)
        fused_rows.append((image_path, fused_map, float(np.mean(scores)), inference_ms))
    return fused_rows


def collect_eval_samples(
    model_name: str,
    checkpoint: Path,
    data_root: Path,
    category: str,
    image_size: int,
    output_dir: Path,
    smooth_sigma: float,
    fusion_scales: list[int],
    map_normalization: str,
) -> list[EvalSample]:
    category_dir = (data_root / category).resolve()
    test_dir = category_dir / "test"
    per_image_normalize = map_normalization == "per_image"
    if len(fusion_scales) > 1:
        predictions = predict_folder_fused(
            model_name,
            checkpoint,
            test_dir,
            fusion_scales,
            output_dir,
            smooth_sigma,
            per_image_normalize=per_image_normalize,
        )
    else:
        predictions = predict_folder(model_name, checkpoint, test_dir, image_size, output_dir)
    samples: list[EvalSample] = []
    for image_path, anomaly_map, score, inference_ms in predictions:
        processed = (
            anomaly_map
            if len(fusion_scales) > 1
            else prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=per_image_normalize)
        )
        gt_label, gt_mask = gt_from_mvtec_path(image_path, category_dir, processed.shape)
        samples.append(
            EvalSample(
                image_path=image_path,
                anomaly_map=processed,
                score=score,
                gt_label=gt_label,
                gt_mask=gt_mask,
                inference_ms=inference_ms,
            )
        )
    return samples


def collect_train_normal_maps(
    model_name: str,
    checkpoint: Path,
    data_root: Path,
    category: str,
    image_size: int,
    output_dir: Path,
    smooth_sigma: float,
    fusion_scales: list[int],
    map_normalization: str,
) -> list[np.ndarray]:
    train_good = data_root / category / "train" / "good"
    per_image_normalize = map_normalization == "per_image"
    if len(fusion_scales) > 1:
        predictions = predict_folder_fused(
            model_name,
            checkpoint,
            train_good,
            fusion_scales,
            output_dir / "train_normal",
            smooth_sigma,
            per_image_normalize=per_image_normalize,
        )
        return [anomaly_map for _, anomaly_map, _, _ in predictions]
    predictions = predict_folder(model_name, checkpoint, train_good, image_size, output_dir / "train_normal")
    return [
        prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=per_image_normalize)
        for _, anomaly_map, _, _ in predictions
    ]


def split_threshold_val(samples: list[EvalSample], ratio: float, seed: int) -> tuple[list[EvalSample], list[EvalSample]]:
    if ratio <= 0:
        return [], samples
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(samples) * ratio)))
    val_set = set(indices[:val_count])
    val = [sample for idx, sample in enumerate(samples) if idx in val_set]
    final = [sample for idx, sample in enumerate(samples) if idx not in val_set]
    return val, final


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(y_score, dtype=np.float64)
    i = 0
    n = len(y_score)
    while i < n:
        j = i + 1
        while j < n and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos = y_true == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def compute_auc_metrics(samples: list[EvalSample]) -> dict[str, float]:
    image_true = np.array([s.gt_label for s in samples], dtype=np.uint8)
    image_score = np.array([s.score for s in samples], dtype=np.float32)
    pixel_true = np.concatenate([s.gt_mask.reshape(-1) for s in samples]).astype(np.uint8)
    pixel_score = np.concatenate([s.anomaly_map.reshape(-1) for s in samples]).astype(np.float32)
    return {
        "image_auroc": safe_auc(image_true, image_score),
        "pixel_auroc": safe_auc(pixel_true, pixel_score),
        "mean_inference_ms": float(np.mean([s.inference_ms for s in samples])) if samples else float("nan"),
    }


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return []
    try:
        from scipy import ndimage as ndi

        labels, num_labels = ndi.label(mask)
        return [labels == idx for idx in range(1, num_labels + 1)]
    except Exception:
        visited = np.zeros_like(mask, dtype=bool)
        components: list[np.ndarray] = []
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if visited[y, x] or not mask[y, x]:
                    continue
                component_pixels: list[tuple[int, int]] = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    component_pixels.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                component = np.zeros_like(mask, dtype=bool)
                yy, xx = zip(*component_pixels)
                component[np.array(yy), np.array(xx)] = True
                components.append(component)
        return components


def compute_aupro(samples: list[EvalSample], max_fpr: float = 0.3, num_thresholds: int = 200) -> float:
    """Compute normalized AU-PRO for segmentation maps up to a false-positive-rate cap."""
    anomaly_maps = [np.asarray(sample.anomaly_map, dtype=np.float32) for sample in samples]
    gt_masks = [(np.asarray(sample.gt_mask) > 0) for sample in samples]
    components_by_sample = [connected_components(mask) for mask in gt_masks]
    num_regions = sum(len(components) for components in components_by_sample)
    if not anomaly_maps or num_regions == 0:
        return float("nan")

    normal_pixel_count = float(sum((~mask).sum() for mask in gt_masks))
    if normal_pixel_count <= 0:
        return float("nan")

    all_scores = np.concatenate([anomaly_map.reshape(-1) for anomaly_map in anomaly_maps])
    lo = float(np.nanmin(all_scores))
    hi = float(np.nanmax(all_scores))
    if hi <= lo:
        return 0.0

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for threshold in np.linspace(hi, lo, num_thresholds):
        false_positive_count = 0.0
        region_overlaps: list[float] = []
        for anomaly_map, gt_mask, components in zip(anomaly_maps, gt_masks, components_by_sample):
            prediction = anomaly_map >= threshold
            false_positive_count += float(np.logical_and(prediction, ~gt_mask).sum())
            for component in components:
                area = float(component.sum())
                if area > 0:
                    region_overlaps.append(float(np.logical_and(prediction, component).sum()) / area)
        fpr = false_positive_count / normal_pixel_count
        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0
        points.append((fpr, pro))

    points.append((1.0, 1.0))
    points = sorted(points, key=lambda item: item[0])
    fprs = np.array([point[0] for point in points], dtype=np.float64)
    pros = np.array([point[1] for point in points], dtype=np.float64)

    unique_fprs: list[float] = []
    unique_pros: list[float] = []
    for fpr in np.unique(fprs):
        unique_fprs.append(float(fpr))
        unique_pros.append(float(np.max(pros[fprs == fpr])))
    fprs = np.array(unique_fprs, dtype=np.float64)
    pros = np.array(unique_pros, dtype=np.float64)

    if fprs[0] > 0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, 0.0)
    if fprs[-1] < max_fpr:
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pros[-1])

    pro_at_max_fpr = float(np.interp(max_fpr, fprs, pros))
    keep = fprs <= max_fpr
    capped_fprs = np.append(fprs[keep], max_fpr)
    capped_pros = np.append(pros[keep], pro_at_max_fpr)
    order = np.argsort(capped_fprs)
    x = capped_fprs[order]
    y = capped_pros[order]
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(y, x))
    else:
        area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / max_fpr


def evaluate_threshold_strategy(
    samples: list[EvalSample],
    strategy: str,
    fixed_threshold_value: float,
    global_threshold: float | None,
    mask_postprocess: str,
    min_area: int,
    open_size: int,
    close_size: int,
) -> tuple[dict[str, float], list[dict]]:
    y_true_pixels: list[np.ndarray] = []
    y_pred_pixels: list[np.ndarray] = []
    y_true_images: list[int] = []
    y_pred_images: list[int] = []
    per_image_rows: list[dict] = []

    for sample in samples:
        mask, threshold = apply_threshold_strategy(
            sample.anomaly_map,
            strategy=strategy,
            fixed_value=fixed_threshold_value,
            global_threshold=global_threshold,
        )
        if mask_postprocess != "none":
            mask = refine_mask(mask, min_area=min_area, open_size=open_size, close_size=close_size)
        image_pred = int(mask.max() > 0)
        y_true_pixels.append(sample.gt_mask.reshape(-1))
        y_pred_pixels.append(mask.reshape(-1))
        y_true_images.append(sample.gt_label)
        y_pred_images.append(image_pred)
        per_image_rows.append(
            {
                "image_path": str(sample.image_path),
                "strategy": strategy,
                "mask_postprocess": mask_postprocess,
                "threshold": threshold,
                "score": sample.score,
                "gt_label": sample.gt_label,
                "pred_label": image_pred,
                "pixel_f1": f1_binary(sample.gt_mask.reshape(-1), mask.reshape(-1)),
            }
        )

    pixel_true = np.concatenate(y_true_pixels).astype(np.uint8)
    pixel_pred = np.concatenate(y_pred_pixels).astype(np.uint8)
    metrics = {
        "pixel_f1": float(f1_binary(pixel_true, pixel_pred)),
        "image_f1_from_mask": float(f1_binary(y_true_images, y_pred_images)),
    }
    return metrics, per_image_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_visual_examples(
    samples: list[EvalSample],
    output_dir: Path,
    strategy: str,
    fixed_threshold_value: float,
    global_threshold: float | None,
    limit: int,
) -> None:
    visual_dir = output_dir / "visuals"
    selected = samples[:limit]
    for idx, sample in enumerate(selected):
        prefix = f"{idx:03d}_{sample.image_path.parent.name}_{sample.image_path.stem}_{strategy}"
        save_prediction_visuals(
            image_path=sample.image_path,
            anomaly_map=sample.anomaly_map,
            output_dir=visual_dir,
            prefix=prefix,
            smooth_sigma=0,
            threshold_strategy=strategy,
            fixed_threshold=fixed_threshold_value,
            global_threshold=global_threshold,
            score=sample.score,
        )


def evaluate_category(args: argparse.Namespace, category: str) -> tuple[list[dict], list[dict]]:
    model_slug = normalize_model_name(args.model)
    checkpoint = resolve_checkpoint(args.ckpt) if args.ckpt else None
    if checkpoint is None:
        from common import find_latest_checkpoint

        checkpoint = find_latest_checkpoint(args.results_root, model_slug, category)

    output_dir = args.output_dir / model_slug / category
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] model={model_slug} category={category} ckpt={checkpoint}")

    samples = collect_eval_samples(
        model_name=model_slug,
        checkpoint=checkpoint,
        data_root=args.data_root,
        category=category,
        image_size=args.image_size,
        output_dir=output_dir,
        smooth_sigma=args.smooth_sigma,
        fusion_scales=args.fusion_scales,
        map_normalization=args.map_normalization,
    )
    threshold_val, final_test = split_threshold_val(samples, args.threshold_val_ratio, args.seed)
    eval_samples = final_test
    auc_metrics = compute_auc_metrics(eval_samples)
    auc_metrics["segmentation_aupro"] = compute_aupro(
        eval_samples,
        max_fpr=args.aupro_max_fpr,
        num_thresholds=args.aupro_steps,
    )

    train_normal_maps: list[np.ndarray] | None = None
    best_f1_global: float | None = None
    if "percentile" in args.threshold_strategies:
        train_normal_maps = collect_train_normal_maps(
            model_name=model_slug,
            checkpoint=checkpoint,
            data_root=args.data_root,
            category=category,
            image_size=args.image_size,
            output_dir=output_dir,
            smooth_sigma=args.smooth_sigma,
            fusion_scales=args.fusion_scales,
            map_normalization=args.map_normalization,
        )
    if "best_f1" in args.threshold_strategies:
        if not threshold_val:
            raise ValueError("--threshold-strategies best_f1 requires --threshold-val-ratio > 0")
        best_f1_global = best_f1_threshold(
            [s.anomaly_map for s in threshold_val],
            [s.gt_mask for s in threshold_val],
        )

    metric_rows: list[dict] = []
    per_image_rows: list[dict] = []
    for strategy in args.threshold_strategies:
        global_threshold = None
        if strategy == "percentile":
            assert train_normal_maps is not None
            global_threshold = percentile_threshold(train_normal_maps, args.percentile)
        elif strategy == "best_f1":
            global_threshold = best_f1_global

        strategy_metrics, strategy_rows = evaluate_threshold_strategy(
            samples=eval_samples,
            strategy=strategy,
            fixed_threshold_value=args.fixed_threshold,
            global_threshold=global_threshold,
            mask_postprocess=args.mask_postprocess,
            min_area=args.min_area,
            open_size=args.open_size,
            close_size=args.close_size,
        )
        row = {
            "model": model_slug,
            "category": category,
            "checkpoint": str(checkpoint),
            "strategy": strategy,
            "fusion_scales": "+".join(str(item) for item in args.fusion_scales),
            "mask_postprocess": args.mask_postprocess,
            "min_area": args.min_area,
            "open_size": args.open_size,
            "close_size": args.close_size,
            "global_threshold": global_threshold if global_threshold is not None else "",
            "num_threshold_val": len(threshold_val),
            "num_final_test": len(eval_samples),
            "aupro_max_fpr": args.aupro_max_fpr,
            **auc_metrics,
            **strategy_metrics,
        }
        metric_rows.append(row)
        per_image_rows.extend(strategy_rows)

        if args.save_visuals:
            save_visual_examples(
                samples=eval_samples,
                output_dir=output_dir,
                strategy=strategy,
                fixed_threshold_value=args.fixed_threshold,
                global_threshold=global_threshold,
                limit=args.visual_limit,
            )

    metrics_path = output_dir / "metrics.csv"
    per_image_path = output_dir / "per_image.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(per_image_path, per_image_rows)
    print(f"[eval] metrics={metrics_path}")
    print(f"[eval] per_image={per_image_path}")
    return metric_rows, per_image_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MVTec AD results and threshold strategies.")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--category", required=True, choices=["bottle", "hazelnut", "metal_nut", "all"])
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint. If omitted, latest is searched.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "eval")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument(
        "--map-normalization",
        choices=["model", "per_image"],
        default="model",
        help=(
            "model keeps the anomaly-map scale returned by Anomalib/EfficientAD; "
            "per_image reproduces the older per-image min-max normalization."
        ),
    )
    parser.add_argument("--fusion-scales", nargs="+", type=int, default=[DEFAULT_IMAGE_SIZE])
    parser.add_argument("--threshold-strategies", nargs="+", default=["fixed", "otsu", "percentile"], choices=["fixed", "otsu", "percentile", "best_f1"])
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--mask-postprocess", choices=["none", "morph_cc"], default="none")
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--open-size", type=int, default=0)
    parser.add_argument("--close-size", type=int, default=5)
    parser.add_argument("--threshold-val-ratio", type=float, default=0.0)
    parser.add_argument("--aupro-max-fpr", type=float, default=0.3)
    parser.add_argument("--aupro-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_metric_rows: list[dict] = []
    all_per_image_rows: list[dict] = []
    for category in categories_from_arg(args.category):
        metrics, per_image = evaluate_category(args, category)
        all_metric_rows.extend(metrics)
        all_per_image_rows.extend(per_image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_all.csv", all_metric_rows)
    write_csv(args.output_dir / "per_image_all.csv", all_per_image_rows)
    print(f"[eval] aggregate_metrics={args.output_dir / 'metrics_all.csv'}")


if __name__ == "__main__":
    main()
