"""Pre-cache model weights before launching parallel experiments."""

from __future__ import annotations

import argparse
import os


def cache_patchcore_backbone(backbone: str) -> None:
    import timm

    print(f"[precache] downloading timm backbone: {backbone}")
    timm.create_model(backbone, pretrained=True)
    print(f"[precache] cached timm backbone: {backbone}")


def cache_anomalib_model(model: str) -> None:
    from common import build_model

    print(f"[precache] instantiating anomalib model: {model}")
    build_model(model)
    print(f"[precache] ready: {model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache Anomalib/timm weights.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2")
    parser.add_argument("--models", nargs="+", default=["patchcore"], choices=["patchcore", "efficientad"])
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
