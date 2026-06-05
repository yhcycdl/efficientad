"""Shared helpers for Anomalib training, inference, and evaluation."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

from data_config import DATA_ROOT, RESULTS_ROOT


MODEL_ALIASES = {
    "cflow": "cflow",
    "c-flow": "cflow",
    "c_flow": "cflow",
    "C-Flow": "cflow",
    "CFlow": "cflow",
    "Cflow": "cflow",
    "draem": "draem",
    "DRAEM": "draem",
    "Draem": "draem",
    "fastflow": "fastflow",
    "fast_flow": "fastflow",
    "FastFlow": "fastflow",
    "Fastflow": "fastflow",
    "padim": "padim",
    "PaDiM": "padim",
    "Padim": "padim",
    "reverse_distillation": "reverse_distillation",
    "reverse-distillation": "reverse_distillation",
    "ReverseDistillation": "reverse_distillation",
    "Reverse_Distillation": "reverse_distillation",
    "stfpm": "stfpm",
    "STFPM": "stfpm",
    "Stfpm": "stfpm",
    "patchcore": "patchcore",
    "patch_core": "patchcore",
    "PatchCore": "patchcore",
    "Patchcore": "patchcore",
    "efficientad": "efficientad",
    "efficient_ad": "efficientad",
    "EfficientAD": "efficientad",
    "EfficientAd": "efficientad",
    "efficientad_s": "efficientad",
    "efficientad-m": "efficientad_m",
    "efficientad_m": "efficientad_m",
    "EfficientAD-M": "efficientad_m",
    "EfficientAD_M": "efficientad_m",
    "efficientad-m-pad": "efficientad_m_pad",
    "efficientad_m_pad": "efficientad_m_pad",
    "EfficientAD-M-Pad": "efficientad_m_pad",
    "EfficientAD_M_Pad": "efficientad_m_pad",
}

ANOMALIB_MODEL_NAMES = {
    "cflow": "Cflow",
    "draem": "Draem",
    "fastflow": "Fastflow",
    "padim": "Padim",
    "reverse_distillation": "ReverseDistillation",
    "stfpm": "Stfpm",
    "patchcore": "Patchcore",
    "efficientad": "EfficientAd",
    "efficientad_m": "EfficientAd",
    "efficientad_m_pad": "EfficientAd",
}

MODEL_CHOICES = tuple(MODEL_ALIASES)

EFFICIENTAD_CONFIGS: dict[str, dict[str, Any]] = {
    "efficientad": {"model_size": "s", "padding": False},
    "efficientad_m": {"model_size": "m", "padding": False},
    "efficientad_m_pad": {"model_size": "m", "padding": True},
}


def normalize_model_name(model: str) -> str:
    try:
        return MODEL_ALIASES[model]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(MODEL_ALIASES)))
        raise ValueError(f"Unknown model '{model}'. Allowed aliases: {allowed}") from exc


def display_model_name(model: str) -> str:
    return ANOMALIB_MODEL_NAMES[normalize_model_name(model)]


def is_efficientad_variant(model: str) -> bool:
    return normalize_model_name(model) in EFFICIENTAD_CONFIGS


def import_anomalib() -> None:
    try:
        import anomalib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Anomalib is not installed. Create the conda environment first:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate efficientad-ad"
        ) from exc


def build_model(model: str) -> Any:
    import_anomalib()
    slug = normalize_model_name(model)
    import anomalib.models as anomalib_models

    class_name = ANOMALIB_MODEL_NAMES[slug]
    try:
        model_cls = getattr(anomalib_models, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Anomalib model class '{class_name}' is not available in this environment. "
            "Run `python - <<'PY'\nimport anomalib.models as m\nprint([x for x in dir(m) if not x.startswith('_')])\nPY` "
            "on the server to inspect installed model names."
        ) from exc
    kwargs = accepted_model_kwargs(model_cls, model_kwargs(slug))
    return model_cls(**kwargs)


def model_kwargs(model: str) -> dict[str, Any]:
    slug = normalize_model_name(model)
    if slug not in EFFICIENTAD_CONFIGS:
        return {}
    config = dict(EFFICIENTAD_CONFIGS[slug])
    config["model_size"] = efficientad_model_size(config["model_size"])
    return config


def accepted_model_kwargs(model_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        signature = inspect.signature(model_cls)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def efficientad_model_size(value: str) -> Any:
    try:
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize

        return getattr(EfficientAdModelSize, value.upper())
    except Exception:
        return value


def build_engine(max_epochs: int | None = None, default_root_dir: Path | str | None = None) -> Any:
    import_anomalib()
    from anomalib.engine import Engine

    kwargs: dict[str, Any] = {
        "accelerator": "auto",
        "devices": 1,
    }
    if max_epochs is not None:
        kwargs["max_epochs"] = max_epochs
    if default_root_dir is not None:
        kwargs["default_root_dir"] = str(default_root_dir)
    return Engine(**kwargs)


def build_mvtec_datamodule(
    category: str,
    data_root: Path | str = DATA_ROOT,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
) -> Any:
    import_anomalib()
    from anomalib.data import Folder
    try:
        from anomalib.data.utils import TestSplitMode, ValSplitMode
    except ImportError:
        from anomalib.data.utils.split import TestSplitMode, ValSplitMode

    category_root = Path(data_root) / category
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    defect_dirs = sorted(path.name for path in test_root.iterdir() if path.is_dir() and path.name != "good")
    if not defect_dirs:
        raise FileNotFoundError(f"No defect test directories found under {test_root}")
    missing_mask_dirs = [name for name in defect_dirs if not (ground_truth_root / name).exists()]
    if missing_mask_dirs:
        raise FileNotFoundError(f"Missing ground_truth directories for {category}: {missing_mask_dirs}")

    return Folder(
        name=f"mvtec_{category}",
        root=str(category_root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=[f"test/{name}" for name in defect_dirs],
        mask_dir=[f"ground_truth/{name}" for name in defect_dirs],
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.2,
        seed=seed,
    )


def build_predict_dataset(path: Path | str, image_size: int = 256) -> Any:
    import_anomalib()
    from anomalib.data import PredictDataset

    return PredictDataset(path=str(path), image_size=(image_size, image_size))


def result_dir(results_root: Path | str, model: str, category: str) -> Path:
    return Path(results_root) / normalize_model_name(model) / category


def latest_ckpt_search_dir(results_root: Path | str = RESULTS_ROOT, model: str | None = None, category: str | None = None) -> Path:
    path = Path(results_root)
    if model:
        path = path / normalize_model_name(model)
    if category:
        path = path / category
    return path


def find_latest_checkpoint(
    results_root: Path | str = RESULTS_ROOT,
    model: str | None = None,
    category: str | None = None,
) -> Path:
    search_dir = latest_ckpt_search_dir(results_root, model, category)
    checkpoints = sorted(search_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {search_dir}")
    return checkpoints[-1]


def update_latest_symlink(checkpoint: Path, target_dir: Path) -> Path:
    latest = target_dir / "latest.ckpt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(checkpoint.resolve())
    except OSError:
        latest.write_text(str(checkpoint.resolve()), encoding="utf-8")
    return latest


def resolve_checkpoint(path_or_pointer: Path | str) -> Path:
    path = Path(path_or_pointer)
    if path.suffix == ".ckpt":
        return path
    if path.exists() and path.is_file():
        target = Path(path.read_text(encoding="utf-8").strip())
        if target.suffix == ".ckpt":
            return target
    raise FileNotFoundError(f"Checkpoint path is not a .ckpt file: {path}")


def tensor_to_numpy(value: Any):
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return tensor_to_numpy(value[0])
    return np.asarray(value)


def scalar_value(value: Any, default: float | None = None) -> float | None:
    arr = tensor_to_numpy(value)
    if arr is None:
        return default
    if isinstance(arr, (str, Path)):
        return default
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def path_value(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, (list, tuple)) and value:
        return path_value(value[0])
    try:
        arr = tensor_to_numpy(value)
        if isinstance(arr, (str, Path)):
            return Path(arr)
    except Exception:
        pass
    return None


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
