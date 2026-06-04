"""Merge per-category evaluation CSV files after parallel evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge outputs/eval/<model>/<category> CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--model", default="efficientad")
    parser.add_argument("--categories", default="bottle,hazelnut,metal_nut")
    args = parser.parse_args()

    metrics_rows: list[dict] = []
    per_image_rows: list[dict] = []
    for category in [item.strip() for item in args.categories.split(",") if item.strip()]:
        category_dir = args.output_dir / args.model / category
        metrics_rows.extend(read_csv(category_dir / "metrics.csv"))
        per_image_rows.extend(read_csv(category_dir / "per_image.csv"))

    write_csv(args.output_dir / "metrics_all.csv", metrics_rows)
    write_csv(args.output_dir / "per_image_all.csv", per_image_rows)
    print(f"[merge] metrics={args.output_dir / 'metrics_all.csv'} rows={len(metrics_rows)}")
    print(f"[merge] per_image={args.output_dir / 'per_image_all.csv'} rows={len(per_image_rows)}")


if __name__ == "__main__":
    main()
