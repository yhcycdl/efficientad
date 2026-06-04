"""Check that Anomalib can see non-empty train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_mvtec_datamodule  # noqa: E402
from data_config import DATA_ROOT, MAIN_CATEGORIES  # noqa: E402


def safe_len(value) -> int | str:
    try:
        return len(value)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def check_category(category: str, data_root: Path) -> None:
    datamodule = build_mvtec_datamodule(
        category=category,
        data_root=data_root,
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup()
    print(f"[data] category={category}")
    for attr in ("train_data", "val_data", "test_data"):
        print(f"[data]   {attr}={safe_len(getattr(datamodule, attr, None))}")
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    print(f"[data]   train_loader.dataset={safe_len(train_loader.dataset)}")
    print(f"[data]   test_loader.dataset={safe_len(test_loader.dataset)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Anomalib datamodule lengths.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--category", default="all", choices=[*MAIN_CATEGORIES, "all"])
    args = parser.parse_args()

    categories = MAIN_CATEGORIES if args.category == "all" else (args.category,)
    for category in categories:
        check_category(category, args.data_root)


if __name__ == "__main__":
    main()
