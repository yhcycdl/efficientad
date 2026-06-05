"""Train anomaly detection models on the selected MVTec AD categories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from common import (
    MODEL_CHOICES,
    build_engine,
    build_model,
    build_mvtec_datamodule,
    find_latest_checkpoint,
    model_kwargs,
    normalize_model_name,
    safe_json,
    update_latest_symlink,
)
from data_config import DATA_ROOT, EPOCH_PRESETS, RESULTS_ROOT, categories_from_arg


def default_train_batch_size(model: str) -> int:
    slug = normalize_model_name(model)
    if slug in {"patchcore", "padim"}:
        return 16
    if slug in {"cflow", "draem"}:
        return 4
    if slug in {"fastflow", "reverse_distillation"}:
        return 8
    if slug == "stfpm":
        return 8
    return 1


def default_eval_batch_size(model: str) -> int:
    slug = normalize_model_name(model)
    if slug in {"patchcore", "padim"}:
        return 16
    if slug in {"cflow", "draem"}:
        return 4
    if slug in {"fastflow", "reverse_distillation"}:
        return 8
    if slug == "stfpm":
        return 8
    return 1


def run_one(
    model_name: str,
    category: str,
    max_epochs: int | None,
    max_steps: int | None,
    data_root: Path,
    results_root: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
) -> dict:
    slug = normalize_model_name(model_name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if max_steps is not None and max_epochs is not None:
        run_name = f"epochs{max_epochs}_steps{max_steps}_{run_id}"
    elif max_steps is not None:
        run_name = f"steps{max_steps}_{run_id}"
    else:
        run_name = f"epochs{max_epochs}_{run_id}"
    output_dir = results_root / slug / category / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] model={slug} category={category} epochs={max_epochs} steps={max_steps} output={output_dir}")
    datamodule = build_mvtec_datamodule(
        category=category,
        data_root=data_root,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    model = build_model(slug)
    engine = build_engine(max_epochs=max_epochs, max_steps=max_steps, default_root_dir=output_dir)
    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)

    checkpoint = find_latest_checkpoint(output_dir)
    latest = update_latest_symlink(checkpoint, results_root / slug / category)
    summary = {
        "model": slug,
        "model_kwargs": safe_json(model_kwargs(slug)),
        "category": category,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest),
        "test_result": safe_json(test_result),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] summary={summary_path}")
    print(f"[train] latest_ckpt={latest}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an anomaly detection model on MVTec AD.")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--category", required=True, choices=["bottle", "hazelnut", "metal_nut", "all"])
    parser.add_argument("--preset", choices=sorted(EPOCH_PRESETS), default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Train for a fixed number of optimizer steps. Useful for EfficientAD reproduction.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_steps = args.max_steps
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    elif max_steps is not None:
        max_epochs = None
    else:
        if args.preset is None:
            raise SystemExit("Choose --preset or pass --max-epochs/--max-steps explicitly.")
        max_epochs = EPOCH_PRESETS[args.preset].max_epochs

    train_batch_size = args.train_batch_size or default_train_batch_size(args.model)
    eval_batch_size = args.eval_batch_size or default_eval_batch_size(args.model)
    categories = categories_from_arg(args.category)

    summaries = [
        run_one(
            model_name=args.model,
            category=category,
            max_epochs=max_epochs,
            max_steps=max_steps,
            data_root=args.data_root,
            results_root=args.results_root,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        for category in categories
    ]

    aggregate_dir = args.results_root / normalize_model_name(args.model)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / "last_train_summary.json"
    aggregate_path.write_text(json.dumps(safe_json(summaries), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] aggregate_summary={aggregate_path}")


if __name__ == "__main__":
    main()
