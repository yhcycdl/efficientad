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

from common import build_engine, build_model, build_predict_dataset, path_value, resolve_checkpoint, scalar_value, tensor_to_numpy
from data_config import (
    DATA_ROOT,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PERCENTILE,
    DEFAULT_SMOOTH_SIGMA,
    OUTPUTS_ROOT,
    categories_from_arg,
)
from postprocess import apply_threshold_strategy, best_f1_threshold, f1_binary, percentile_threshold, prepare_map
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


def collect_eval_samples(
    model_name: str,
    checkpoint: Path,
    data_root: Path,
    category: str,
    image_size: int,
    output_dir: Path,
    smooth_sigma: float,
) -> list[EvalSample]:
    category_dir = (data_root / category).resolve()
    test_dir = category_dir / "test"
    predictions = predict_folder(model_name, checkpoint, test_dir, image_size, output_dir)
    samples: list[EvalSample] = []
    for image_path, anomaly_map, score, inference_ms in predictions:
        processed = prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=True)
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
) -> list[np.ndarray]:
    train_good = data_root / category / "train" / "good"
    predictions = predict_folder(model_name, checkpoint, train_good, image_size, output_dir / "train_normal")
    return [prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=True) for _, anomaly_map, _, _ in predictions]


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


def evaluate_threshold_strategy(
    samples: list[EvalSample],
    strategy: str,
    fixed_threshold_value: float,
    global_threshold: float | None,
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
        image_pred = int(mask.max() > 0)
        y_true_pixels.append(sample.gt_mask.reshape(-1))
        y_pred_pixels.append(mask.reshape(-1))
        y_true_images.append(sample.gt_label)
        y_pred_images.append(image_pred)
        per_image_rows.append(
            {
                "image_path": str(sample.image_path),
                "strategy": strategy,
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
    checkpoint = resolve_checkpoint(args.ckpt) if args.ckpt else None
    if checkpoint is None:
        from common import find_latest_checkpoint

        checkpoint = find_latest_checkpoint(args.results_root, args.model, category)

    output_dir = args.output_dir / args.model.lower() / category
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] model={args.model} category={category} ckpt={checkpoint}")

    samples = collect_eval_samples(
        model_name=args.model,
        checkpoint=checkpoint,
        data_root=args.data_root,
        category=category,
        image_size=args.image_size,
        output_dir=output_dir,
        smooth_sigma=args.smooth_sigma,
    )
    threshold_val, final_test = split_threshold_val(samples, args.threshold_val_ratio, args.seed)
    eval_samples = final_test
    auc_metrics = compute_auc_metrics(eval_samples)

    train_normal_maps: list[np.ndarray] | None = None
    best_f1_global: float | None = None
    if "percentile" in args.threshold_strategies:
        train_normal_maps = collect_train_normal_maps(
            model_name=args.model,
            checkpoint=checkpoint,
            data_root=args.data_root,
            category=category,
            image_size=args.image_size,
            output_dir=output_dir,
            smooth_sigma=args.smooth_sigma,
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
        )
        row = {
            "model": args.model,
            "category": category,
            "checkpoint": str(checkpoint),
            "strategy": strategy,
            "global_threshold": global_threshold if global_threshold is not None else "",
            "num_threshold_val": len(threshold_val),
            "num_final_test": len(eval_samples),
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
    parser.add_argument("--model", required=True, choices=["patchcore", "efficientad", "PatchCore", "EfficientAD"])
    parser.add_argument("--category", required=True, choices=["bottle", "hazelnut", "metal_nut", "all"])
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint. If omitted, latest is searched.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "eval")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--threshold-strategies", nargs="+", default=["fixed", "otsu", "percentile"], choices=["fixed", "otsu", "percentile", "best_f1"])
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--threshold-val-ratio", type=float, default=0.0)
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
