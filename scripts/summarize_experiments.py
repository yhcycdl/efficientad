"""Create compact CSV summaries for report tables."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_model, normalize_model_name


EXPERIMENT_LABELS = {
    "eval_cflow": "CFlow",
    "eval": "EfficientAD",
    "eval_draem": "DRAEM",
    "eval_efficientad_fusion_morph": "EfficientAD+Ours",
    "eval_efficientad_m": "EfficientAD-M",
    "eval_efficientad_m_fusion_morph": "EfficientAD-M+Ours",
    "eval_efficientad_m_v2": "EfficientAD-M (fixed eval)",
    "eval_efficientad_m_v2_split": "EfficientAD-M (fixed eval, split)",
    "eval_efficientad_m_fusion_morph_v2": "EfficientAD-M+Ours (fixed eval)",
    "eval_efficientad_m_pad": "EfficientAD-M+Pad",
    "eval_efficientad_m_pad_fusion_morph": "EfficientAD-M+Pad+Ours",
    "eval_fastflow": "FastFlow",
    "eval_patchcore": "PatchCore",
    "eval_padim": "PaDiM",
    "eval_reverse_distillation": "ReverseDistillation",
    "eval_stfpm": "STFPM",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_field(rows: list[dict], field: str, precision: int = 6) -> str:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in {None, ""}:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            values.append(parsed)
    if not values:
        return ""
    return f"{sum(values) / len(values):.{precision}f}"


def variant_label(experiment: str, model: str) -> str:
    return EXPERIMENT_LABELS.get(experiment, model)


def best_rows(metrics: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in metrics:
        key = (row.get("experiment", ""), row.get("model", ""), row.get("category", ""))
        grouped.setdefault(key, []).append(row)
    output = []
    for (experiment, model, category), rows in sorted(grouped.items()):
        best_pixel = max(rows, key=lambda item: to_float(item.get("pixel_f1")))
        fixed = next((item for item in rows if item.get("strategy") == "fixed"), rows[0])
        output.append(
            {
                "experiment": experiment,
                "variant": variant_label(experiment, model),
                "model": model,
                "category": category,
                "image_auroc": fixed.get("image_auroc", ""),
                "pixel_auroc": fixed.get("pixel_auroc", ""),
                "segmentation_aupro": fixed.get("segmentation_aupro", ""),
                "mean_inference_ms": fixed.get("mean_inference_ms", ""),
                "fixed_pixel_f1": fixed.get("pixel_f1", ""),
                "best_strategy": best_pixel.get("strategy", ""),
                "best_pixel_f1": best_pixel.get("pixel_f1", ""),
                "image_f1_from_mask": best_pixel.get("image_f1_from_mask", ""),
                "fusion_scales": best_pixel.get("fusion_scales", ""),
                "mask_postprocess": best_pixel.get("mask_postprocess", ""),
            }
        )
    return output


def aggregate_rows(compact_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in compact_rows:
        key = (row.get("experiment", ""), row.get("variant", ""), row.get("model", ""))
        grouped.setdefault(key, []).append(row)

    output = []
    for (experiment, variant, model), rows in sorted(grouped.items()):
        output.append(
            {
                "experiment": experiment,
                "variant": variant,
                "model": model,
                "num_categories": len(rows),
                "mean_image_auroc": mean_field(rows, "image_auroc"),
                "mean_pixel_auroc": mean_field(rows, "pixel_auroc"),
                "mean_segmentation_aupro": mean_field(rows, "segmentation_aupro"),
                "mean_fixed_pixel_f1": mean_field(rows, "fixed_pixel_f1"),
                "mean_best_pixel_f1": mean_field(rows, "best_pixel_f1"),
                "mean_inference_ms": mean_field(rows, "mean_inference_ms", precision=3),
            }
        )
    return output


def model_profile_rows(models: list[str], results_root: Path) -> list[dict]:
    rows = []
    for model_name in models:
        slug = normalize_model_name(model_name)
        param_count = 0
        trainable_count = 0
        profile_error = ""
        try:
            model = build_model(slug)
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    count = int(param.numel())
                    param_count += count
                    if param.requires_grad:
                        trainable_count += count
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"
        ckpts = sorted((results_root / slug).rglob("latest.ckpt"))
        ckpt_size_mb = sum(path.stat().st_size for path in ckpts if path.exists()) / 1024**2
        rows.append(
            {
                "model": slug,
                "parameters_m": f"{param_count / 1e6:.3f}",
                "trainable_parameters_m": f"{trainable_count / 1e6:.3f}",
                "latest_ckpt_count": len(ckpts),
                "latest_ckpt_total_mb": f"{ckpt_size_mb:.2f}",
                "profile_error": profile_error,
            }
        )
    return rows


def final_comparison_rows(aggregate: list[dict], profiles: list[dict]) -> list[dict]:
    profile_by_model = {row["model"]: row for row in profiles}
    output = []
    for row in aggregate:
        profile = profile_by_model.get(row["model"], {})
        output.append(
            {
                **row,
                "parameters_m": profile.get("parameters_m", ""),
                "trainable_parameters_m": profile.get("trainable_parameters_m", ""),
                "latest_ckpt_total_mb": profile.get("latest_ckpt_total_mb", ""),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics and model profiles.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_summary"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--metrics", nargs="+", default=[
        "outputs/eval/metrics_all.csv",
        "outputs/eval_cflow/metrics_all.csv",
        "outputs/eval_draem/metrics_all.csv",
        "outputs/eval_efficientad_fusion_morph/metrics_all.csv",
        "outputs/eval_efficientad_m/metrics_all.csv",
        "outputs/eval_efficientad_m_fusion_morph/metrics_all.csv",
        "outputs/eval_efficientad_m_v2/metrics_all.csv",
        "outputs/eval_efficientad_m_v2_split/metrics_all.csv",
        "outputs/eval_efficientad_m_fusion_morph_v2/metrics_all.csv",
        "outputs/eval_efficientad_m_pad/metrics_all.csv",
        "outputs/eval_efficientad_m_pad_fusion_morph/metrics_all.csv",
        "outputs/eval_fastflow/metrics_all.csv",
        "outputs/eval_patchcore/metrics_all.csv",
        "outputs/eval_padim/metrics_all.csv",
        "outputs/eval_reverse_distillation/metrics_all.csv",
        "outputs/eval_stfpm/metrics_all.csv",
    ])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "cflow",
            "draem",
            "fastflow",
            "padim",
            "stfpm",
            "efficientad",
            "efficientad_m",
            "efficientad_m_pad",
            "patchcore",
            "reverse_distillation",
        ],
    )
    args = parser.parse_args()

    all_rows = []
    for metric_path in args.metrics:
        path = Path(metric_path)
        tag = path.parent.name
        for row in read_csv(path):
            row = dict(row)
            row["experiment"] = tag
            all_rows.append(row)

    compact_rows = best_rows(all_rows)
    aggregate = aggregate_rows(compact_rows)
    profile_rows = model_profile_rows(args.models, args.results_root)
    final_comparison = final_comparison_rows(aggregate, profile_rows)

    write_csv(args.output_dir / "all_metrics.csv", all_rows)
    write_csv(args.output_dir / "compact_metrics.csv", compact_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "model_profiles.csv", profile_rows)
    write_csv(args.output_dir / "final_comparison.csv", final_comparison)
    print(f"[summary] all={args.output_dir / 'all_metrics.csv'} rows={len(all_rows)}")
    print(f"[summary] compact={args.output_dir / 'compact_metrics.csv'} rows={len(compact_rows)}")
    print(f"[summary] aggregate={args.output_dir / 'aggregate_metrics.csv'} rows={len(aggregate)}")
    print(f"[summary] profiles={args.output_dir / 'model_profiles.csv'} rows={len(profile_rows)}")
    print(f"[summary] final={args.output_dir / 'final_comparison.csv'} rows={len(final_comparison)}")


if __name__ == "__main__":
    main()
