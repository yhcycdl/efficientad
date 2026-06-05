"""Diagnose anomaly-map scale and visualization for one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_engine, build_model, build_predict_dataset, path_value, resolve_checkpoint, scalar_value, tensor_to_numpy
from postprocess import prepare_map, squeeze_map
from visualize import colorize_heatmap, load_image, make_overlay, resize_map_to_image, save_prediction_visuals


def describe(name: str, anomaly_map: np.ndarray) -> None:
    arr = squeeze_map(anomaly_map)
    percentiles = np.percentile(arr, [0, 1, 5, 50, 90, 95, 99, 99.5, 100])
    print(
        f"[{name}] shape={arr.shape} "
        f"min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g} std={arr.std():.6g}"
    )
    print(
        f"[{name}] p0={percentiles[0]:.6g} p1={percentiles[1]:.6g} p5={percentiles[2]:.6g} "
        f"p50={percentiles[3]:.6g} p90={percentiles[4]:.6g} p95={percentiles[5]:.6g} "
        f"p99={percentiles[6]:.6g} p99.5={percentiles[7]:.6g} p100={percentiles[8]:.6g}"
    )


def save_heatmap_variant(image_path: Path, anomaly_map: np.ndarray, output_dir: Path, name: str) -> None:
    image = load_image(image_path)
    display_map = prepare_map(anomaly_map, smooth_sigma=0, normalize=True)
    resized = resize_map_to_image(display_map, image)
    heatmap = colorize_heatmap(resized)
    overlay = make_overlay(image, heatmap)
    heatmap.save(output_dir / f"{name}_heatmap.png")
    overlay.save(output_dir / f"{name}_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect anomaly-map statistics and visualizations for one image.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/map_diagnostics"), type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--smooth-sigma", default=4.0, type=float)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_predict_dataset(args.image, image_size=args.image_size)
    model = build_model(args.model)
    engine = build_engine(default_root_dir=args.output_dir / "engine")
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=str(resolve_checkpoint(args.ckpt)))
    if not predictions:
        raise RuntimeError("No prediction was returned.")

    prediction = predictions[0]
    image_path = path_value(getattr(prediction, "image_path", None)) or args.image
    anomaly_map = tensor_to_numpy(getattr(prediction, "anomaly_map", None))
    pred_mask = tensor_to_numpy(getattr(prediction, "pred_mask", None))
    score = scalar_value(getattr(prediction, "pred_score", None))
    print(f"[prediction] image={image_path} score={score}")
    print(f"[prediction] attrs={sorted(name for name in dir(prediction) if not name.startswith('_'))}")

    if anomaly_map is None:
        raise RuntimeError("Prediction does not contain anomaly_map.")

    describe("raw_anomaly_map", anomaly_map)
    per_image = prepare_map(anomaly_map, smooth_sigma=0, normalize=True)
    describe("per_image_minmax", per_image)
    smoothed = prepare_map(anomaly_map, smooth_sigma=args.smooth_sigma, normalize=True)
    describe(f"smoothed_minmax_sigma_{args.smooth_sigma:g}", smoothed)
    if pred_mask is not None:
        describe("pred_mask", pred_mask)

    np.savez_compressed(
        args.output_dir / "maps.npz",
        raw_anomaly_map=squeeze_map(anomaly_map),
        per_image_minmax=per_image,
        smoothed_minmax=smoothed,
        pred_mask=squeeze_map(pred_mask) if pred_mask is not None else np.array([]),
    )
    save_heatmap_variant(Path(image_path), anomaly_map, args.output_dir, "raw_display_minmax")
    save_heatmap_variant(Path(image_path), smoothed, args.output_dir, f"smoothed_sigma_{args.smooth_sigma:g}")
    save_prediction_visuals(
        image_path=Path(image_path),
        anomaly_map=anomaly_map,
        output_dir=args.output_dir,
        prefix="panel_raw",
        smooth_sigma=0,
        threshold_strategy="otsu",
        score=score,
    )
    save_prediction_visuals(
        image_path=Path(image_path),
        anomaly_map=anomaly_map,
        output_dir=args.output_dir,
        prefix=f"panel_smoothed_sigma_{args.smooth_sigma:g}",
        smooth_sigma=args.smooth_sigma,
        threshold_strategy="otsu",
        score=score,
    )
    print(f"[diagnose] saved={args.output_dir}")


if __name__ == "__main__":
    main()
