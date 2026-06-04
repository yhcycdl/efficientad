"""Run single-image or folder inference and save anomaly visualizations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import build_engine, build_model, build_predict_dataset, path_value, resolve_checkpoint, scalar_value, tensor_to_numpy
from data_config import DEFAULT_FIXED_THRESHOLD, DEFAULT_IMAGE_SIZE, DEFAULT_SMOOTH_SIGMA, OUTPUTS_ROOT
from postprocess import prepare_map
from visualize import save_prediction_visuals


def prediction_to_record(prediction, fallback_index: int) -> dict:
    image_path = path_value(getattr(prediction, "image_path", None))
    if image_path is None:
        image_path = Path(f"prediction_{fallback_index:04d}.png")

    anomaly_map = tensor_to_numpy(getattr(prediction, "anomaly_map", None))
    if anomaly_map is None:
        anomaly_map = tensor_to_numpy(getattr(prediction, "pred_mask", None))
    if anomaly_map is None:
        raise RuntimeError("Prediction does not contain anomaly_map or pred_mask.")

    score = scalar_value(getattr(prediction, "pred_score", None))
    if score is None:
        score = float(np.max(anomaly_map))

    label = scalar_value(getattr(prediction, "pred_label", None))
    return {
        "image_path": image_path,
        "anomaly_map": anomaly_map,
        "score": float(score),
        "pred_label": label,
    }


def run_inference(
    model_name: str,
    checkpoint: Path,
    input_path: Path,
    output_dir: Path,
    image_size: int,
    smooth_sigma: float,
    threshold_strategy: str,
    fixed_threshold: float,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_predict_dataset(input_path, image_size=image_size)
    model = build_model(model_name)
    engine = build_engine(default_root_dir=output_dir)
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=str(checkpoint))
    if predictions is None:
        raise RuntimeError("Anomalib returned no predictions.")

    rows: list[dict] = []
    for idx, prediction in enumerate(predictions):
        record = prediction_to_record(prediction, idx)
        image_path: Path = record["image_path"]
        stem = f"{idx:04d}_{image_path.stem}"
        processed = prepare_map(record["anomaly_map"], smooth_sigma=smooth_sigma, normalize=True)
        map_path = output_dir / f"{stem}_anomaly_map.npz"
        np.savez_compressed(map_path, anomaly_map=processed, raw_anomaly_map=record["anomaly_map"])

        visuals = save_prediction_visuals(
            image_path=image_path,
            anomaly_map=record["anomaly_map"],
            output_dir=output_dir,
            prefix=stem,
            smooth_sigma=smooth_sigma,
            threshold_strategy=threshold_strategy,
            fixed_threshold=fixed_threshold,
            score=record["score"],
        )
        rows.append(
            {
                "image_path": str(image_path),
                "score": record["score"],
                "pred_label": record["pred_label"],
                "anomaly_map": str(map_path),
                **{name: str(path) for name, path in visuals.items()},
            }
        )
        print(f"[infer] {image_path} score={record['score']:.6f} panel={visuals['panel']}")

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[infer] saved={csv_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Anomalib inference and save visual outputs.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["padim", "stfpm", "patchcore", "efficientad", "PaDiM", "STFPM", "PatchCore", "EfficientAD"],
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Image path or directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "infer")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--threshold-strategy", choices=["fixed", "otsu"], default="otsu")
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        model_name=args.model,
        checkpoint=resolve_checkpoint(args.ckpt),
        input_path=args.input,
        output_dir=args.output_dir,
        image_size=args.image_size,
        smooth_sigma=args.smooth_sigma,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
    )


if __name__ == "__main__":
    main()
