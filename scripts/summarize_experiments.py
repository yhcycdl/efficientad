"""Create compact CSV summaries for report tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_model, normalize_model_name


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


def best_rows(metrics: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in metrics:
        grouped.setdefault((row.get("model", ""), row.get("category", "")), []).append(row)
    output = []
    for (model, category), rows in sorted(grouped.items()):
        best_pixel = max(rows, key=lambda item: float(item.get("pixel_f1") or 0.0))
        fixed = next((item for item in rows if item.get("strategy") == "fixed"), rows[0])
        output.append(
            {
                "model": model,
                "category": category,
                "image_auroc": fixed.get("image_auroc", ""),
                "pixel_auroc": fixed.get("pixel_auroc", ""),
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


def model_profile_rows(models: list[str], results_root: Path) -> list[dict]:
    rows = []
    for model_name in models:
        slug = normalize_model_name(model_name)
        model = build_model(slug)
        param_count = 0
        trainable_count = 0
        if hasattr(model, "parameters"):
            for param in model.parameters():
                count = int(param.numel())
                param_count += count
                if param.requires_grad:
                    trainable_count += count
        ckpts = sorted((results_root / slug).rglob("latest.ckpt"))
        ckpt_size_mb = sum(path.stat().st_size for path in ckpts if path.exists()) / 1024**2
        rows.append(
            {
                "model": slug,
                "parameters_m": f"{param_count / 1e6:.3f}",
                "trainable_parameters_m": f"{trainable_count / 1e6:.3f}",
                "latest_ckpt_count": len(ckpts),
                "latest_ckpt_total_mb": f"{ckpt_size_mb:.2f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics and model profiles.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_summary"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--metrics", nargs="+", default=[
        "outputs/eval/metrics_all.csv",
        "outputs/eval_efficientad_fusion_morph/metrics_all.csv",
        "outputs/eval_patchcore/metrics_all.csv",
        "outputs/eval_padim/metrics_all.csv",
        "outputs/eval_stfpm/metrics_all.csv",
    ])
    parser.add_argument("--models", nargs="+", default=["padim", "stfpm", "efficientad", "patchcore"])
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
    profile_rows = model_profile_rows(args.models, args.results_root)

    write_csv(args.output_dir / "all_metrics.csv", all_rows)
    write_csv(args.output_dir / "compact_metrics.csv", compact_rows)
    write_csv(args.output_dir / "model_profiles.csv", profile_rows)
    print(f"[summary] all={args.output_dir / 'all_metrics.csv'} rows={len(all_rows)}")
    print(f"[summary] compact={args.output_dir / 'compact_metrics.csv'} rows={len(compact_rows)}")
    print(f"[summary] profiles={args.output_dir / 'model_profiles.csv'} rows={len(profile_rows)}")


if __name__ == "__main__":
    main()
