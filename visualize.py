"""Visualization helpers for heatmaps, overlays, masks, and demo panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from postprocess import apply_threshold_strategy, prepare_map


def _resampling(name: str):
    return getattr(Image, "Resampling", Image).__dict__[name]


def load_image(path: Path | str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_map_to_image(anomaly_map: np.ndarray, image: Image.Image, nearest: bool = False) -> np.ndarray:
    mode = _resampling("NEAREST" if nearest else "BILINEAR")
    arr = np.asarray(anomaly_map, dtype=np.float32)
    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")
    pil = pil.resize(image.size, mode)
    return np.asarray(pil).astype(np.float32) / 255.0


def colorize_heatmap(anomaly_map: np.ndarray) -> Image.Image:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps

        colored = colormaps["jet"](arr)[:, :, :3]
    except Exception:
        colored = np.stack(
            [
                np.clip(1.5 - np.abs(4.0 * arr - 3.0), 0.0, 1.0),
                np.clip(1.5 - np.abs(4.0 * arr - 2.0), 0.0, 1.0),
                np.clip(1.5 - np.abs(4.0 * arr - 1.0), 0.0, 1.0),
            ],
            axis=-1,
        )
    return Image.fromarray((colored * 255).astype(np.uint8), mode="RGB")


def make_overlay(image: Image.Image, heatmap: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha=alpha)


def mask_image(mask: np.ndarray) -> Image.Image:
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def make_side_by_side(
    image: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    mask: Image.Image,
    score: float | None = None,
    threshold: float | None = None,
) -> Image.Image:
    width, height = image.size
    header_h = 34
    panel = Image.new("RGB", (width * 4, height + header_h), "white")
    draw = ImageDraw.Draw(panel)
    labels = ["image", "heatmap", "overlay", "mask"]
    for i, (label, img) in enumerate(zip(labels, [image, heatmap, overlay, mask.convert("RGB")])):
        x = i * width
        panel.paste(img.convert("RGB"), (x, header_h))
        draw.text((x + 8, 8), label, fill=(20, 20, 20), font=ImageFont.load_default())
    summary = []
    if score is not None:
        summary.append(f"score={score:.4f}")
    if threshold is not None:
        summary.append(f"thr={threshold:.4f}")
    if summary:
        draw.text((width * 2 + 8, 8), " ".join(summary), fill=(20, 20, 20), font=ImageFont.load_default())
    return panel


def save_prediction_visuals(
    image_path: Path | str,
    anomaly_map: np.ndarray,
    output_dir: Path | str,
    prefix: str | None = None,
    smooth_sigma: float = 4.0,
    threshold_strategy: str = "otsu",
    fixed_threshold: float = 0.5,
    global_threshold: float | None = None,
    score: float | None = None,
) -> dict[str, Path]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or image_path.stem

    image = load_image(image_path)
    processed_map = prepare_map(anomaly_map, smooth_sigma=smooth_sigma, normalize=True)
    resized_map = resize_map_to_image(processed_map, image)
    mask, threshold = apply_threshold_strategy(
        resized_map,
        threshold_strategy,
        fixed_value=fixed_threshold,
        global_threshold=global_threshold,
    )

    heatmap = colorize_heatmap(resized_map)
    overlay = make_overlay(image, heatmap)
    mask_pil = mask_image(mask)
    panel = make_side_by_side(image, heatmap, overlay, mask_pil, score=score, threshold=threshold)

    paths = {
        "heatmap": output_dir / f"{prefix}_heatmap.png",
        "overlay": output_dir / f"{prefix}_overlay.png",
        "mask": output_dir / f"{prefix}_mask.png",
        "panel": output_dir / f"{prefix}_panel.png",
    }
    heatmap.save(paths["heatmap"])
    overlay.save(paths["overlay"])
    mask_pil.save(paths["mask"])
    panel.save(paths["panel"])
    return paths


def _load_map(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "anomaly_map" in data:
            return data["anomaly_map"]
        return data[data.files[0]]
    raise ValueError(f"Unsupported map file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create heatmap, overlay, mask, and panel visualizations.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path, help=".npy or .npz anomaly map")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--smooth-sigma", type=float, default=4.0)
    parser.add_argument("--threshold-strategy", choices=["fixed", "otsu"], default="otsu")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    args = parser.parse_args()

    paths = save_prediction_visuals(
        image_path=args.image,
        anomaly_map=_load_map(args.map),
        output_dir=args.output_dir,
        prefix=args.prefix,
        smooth_sigma=args.smooth_sigma,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
