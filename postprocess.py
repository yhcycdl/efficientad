"""Anomaly-map postprocessing and threshold strategies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    strategy: str
    threshold: float | None
    description: str


def squeeze_map(anomaly_map) -> np.ndarray:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D anomaly map after squeeze, got shape {arr.shape}")
    return arr


def min_max_normalize(anomaly_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = squeeze_map(anomaly_map)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn + eps)).astype(np.float32)


def gaussian_smooth(anomaly_map: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    arr = squeeze_map(anomaly_map)
    if sigma <= 0:
        return arr.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except Exception:
        radius = max(1, int(round(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, tmp)
        return smoothed.astype(np.float32)


def prepare_map(anomaly_map: np.ndarray, smooth_sigma: float = 0.0, normalize: bool = True) -> np.ndarray:
    arr = gaussian_smooth(anomaly_map, smooth_sigma)
    if normalize:
        arr = min_max_normalize(arr)
    return arr.astype(np.float32)


def otsu_threshold(anomaly_map: np.ndarray, bins: int = 256) -> float:
    arr = squeeze_map(anomaly_map)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx <= mn:
        return mx

    hist, edges = np.histogram(arr.ravel(), bins=bins, range=(mn, mx))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    score = np.zeros_like(denom)
    valid = denom > 1e-12
    score[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(centers[int(np.argmax(score))])


def percentile_threshold(anomaly_maps: list[np.ndarray], percentile: float = 99.0) -> float:
    if not anomaly_maps:
        raise ValueError("percentile threshold needs at least one anomaly map")
    values = np.concatenate([squeeze_map(m).reshape(-1) for m in anomaly_maps])
    return float(np.percentile(values, percentile))


def f1_binary(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_pred = np.asarray(y_pred).astype(bool).reshape(-1)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    denom = 2.0 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / denom


def fixed_threshold(threshold: float = 0.5) -> float:
    return float(threshold)


def best_f1_threshold(
    anomaly_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    num_steps: int = 256,
) -> float:
    if len(anomaly_maps) != len(gt_masks):
        raise ValueError("anomaly_maps and gt_masks must have the same length")
    if not anomaly_maps:
        raise ValueError("best-f1 threshold needs validation samples")

    y_score = np.concatenate([squeeze_map(m).reshape(-1) for m in anomaly_maps])
    y_true = np.concatenate([(np.asarray(mask) > 0).reshape(-1) for mask in gt_masks]).astype(np.uint8)
    if len(np.unique(y_true)) < 2:
        raise ValueError("best-f1 threshold needs both normal and anomalous pixels")

    lo = float(np.nanmin(y_score))
    hi = float(np.nanmax(y_score))
    if hi <= lo:
        return hi

    best_thr = lo
    best_f1 = -1.0
    for thr in np.linspace(lo, hi, num_steps):
        pred = y_score >= thr
        f1 = f1_binary(y_true, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


def binary_mask(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    return (squeeze_map(anomaly_map) >= float(threshold)).astype(np.uint8)


def apply_threshold_strategy(
    anomaly_map: np.ndarray,
    strategy: str,
    fixed_value: float = 0.5,
    global_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    strategy = strategy.lower()
    if strategy == "fixed":
        threshold = fixed_threshold(fixed_value)
    elif strategy == "otsu":
        threshold = otsu_threshold(anomaly_map)
    elif strategy in {"percentile", "best_f1"}:
        if global_threshold is None:
            raise ValueError(f"{strategy} needs a global_threshold")
        threshold = float(global_threshold)
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    return binary_mask(anomaly_map, threshold), threshold


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
    parser = argparse.ArgumentParser(description="Postprocess a saved anomaly map.")
    parser.add_argument("--map", required=True, type=Path, help=".npy or .npz anomaly map")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy mask path")
    parser.add_argument("--strategy", choices=["fixed", "otsu"], default="otsu")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--smooth-sigma", type=float, default=4.0)
    args = parser.parse_args()

    anomaly_map = prepare_map(_load_map(args.map), smooth_sigma=args.smooth_sigma)
    mask, threshold = apply_threshold_strategy(
        anomaly_map,
        args.strategy,
        fixed_value=args.fixed_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask)
    print(f"saved={args.output} strategy={args.strategy} threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
