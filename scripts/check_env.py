"""Check Python, PyTorch, CUDA, and Anomalib availability."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"platform={platform.platform()}")

    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gb = props.total_memory / 1024**3
                print(f"gpu[{i}]={props.name} memory={gb:.2f}GiB capability={props.major}.{props.minor}")
    except Exception as exc:
        print(f"torch_check_failed={type(exc).__name__}: {exc}")

    try:
        import anomalib

        print(f"anomalib={getattr(anomalib, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"anomalib_check_failed={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
