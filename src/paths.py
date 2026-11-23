"""Where models, datasets and outputs live."""

from __future__ import annotations

import os

NETWORK_VOLUME = "/runpod-volume"
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMAFACTORY_DIR = os.environ.get("LLAMAFACTORY_DIR", "/opt/llamafactory")


def has_volume() -> bool:
    return os.path.isdir(NETWORK_VOLUME) and os.access(NETWORK_VOLUME, os.W_OK)


def _root() -> str:
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "llama-factory")
    return "/tmp/llama-factory"


def model_cache_dir() -> str:
    return os.environ.get("MODEL_CACHE_DIR") or os.path.join(_root(), "models")


def output_root() -> str:
    return os.environ.get("OUTPUT_ROOT") or os.path.join(_root(), "runs")


def work_root() -> str:
    """Scratch space for downloads / extracted datasets (always local disk)."""
    return os.environ.get("WORK_ROOT") or "/tmp/llama-factory/work"


def bundled_datasets_dir() -> str:
    return os.path.join(APP_DIR, "examples", "datasets")


def upstream_data_dir() -> str:
    return os.path.join(LLAMAFACTORY_DIR, "data")


def is_persistent(path: str) -> bool:
    return has_volume() and os.path.abspath(path).startswith(NETWORK_VOLUME + "/")


def configure_caches() -> None:
    root = _root()
    os.environ.setdefault("HF_HOME", os.path.join(root, "hf-home"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(root, "hf-home", "datasets"))
