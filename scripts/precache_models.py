"""Pre-cache model weights before launching parallel experiments."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cache_patchcore_backbone(backbone: str) -> None:
    import timm

    print(f"[precache] downloading timm backbone: {backbone}")
    timm.create_model(backbone, pretrained=True)
    print(f"[precache] cached timm backbone: {backbone}")


def cache_anomalib_model(model: str) -> None:
    from common import build_model, normalize_model_name

    slug = normalize_model_name(model)
    print(f"[precache] instantiating anomalib model: {slug}")
    anomalib_model = build_model(slug)
    if slug.startswith("efficientad"):
        cache_efficientad_assets(anomalib_model)
    print(f"[precache] ready: {slug}")


def _image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in extensions)


def _move_incomplete_dir(path: Path) -> None:
    if not path.exists():
        return
    backup_root = PROJECT_ROOT / ".cache" / "incomplete_assets"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"{path.name}_{os.getpid()}"
    print(f"[precache] moving incomplete asset dir {path} -> {backup}")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.move(str(path), str(backup))


def cache_efficientad_assets(model) -> None:
    print("[precache] preparing EfficientAD pretrained teacher weights")
    model.prepare_pretrained_model()

    imagenet_dir = Path(getattr(model, "imagenet_dir", "datasets/imagenette"))
    if imagenet_dir.exists() and _image_count(imagenet_dir) == 0:
        _move_incomplete_dir(imagenet_dir)

    print(f"[precache] preparing EfficientAD ImageNette data at {imagenet_dir}")
    model.prepare_imagenette_data((256, 256))
    image_count = _image_count(imagenet_dir)
    if image_count == 0:
        raise RuntimeError(f"EfficientAD ImageNette cache has no images: {imagenet_dir}")
    print(f"[precache] EfficientAD ImageNette images={image_count}")


def parse_args() -> argparse.Namespace:
    from common import MODEL_CHOICES

    parser = argparse.ArgumentParser(description="Pre-cache Anomalib/timm weights.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2")
    parser.add_argument("--models", nargs="+", default=["patchcore"], choices=MODEL_CHOICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[precache] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<default>')}")
    print(f"[precache] HF_HOME={os.environ.get('HF_HOME', '<default>')}")
    if "patchcore" in args.models:
        cache_patchcore_backbone(args.patchcore_backbone)
    for model in args.models:
        cache_anomalib_model(model)


if __name__ == "__main__":
    main()
