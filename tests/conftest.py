import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SECRET_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
              "BUCKET_SECRET_ACCESS_KEY")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No real volume, no secrets, all worker paths under tmp."""
    for k in SECRET_ENV:
        monkeypatch.delenv(k, raising=False)
    for k in list(os.environ):
        if k in {"MODEL_NAME_OR_PATH", "MODEL_REVISION", "STAGE", "QUANTIZATION_BIT", "DATASET", "DATASET_URL",
                 "NUM_TRAIN_EPOCHS", "MAX_STEPS", "LORA_RANK", "HF_OUTPUT_REPO", "TRAIN_TIMEOUT_SECONDS",
                 "DRY_RUN", "TEMPLATE", "PREF_LOSS"}:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LLAMAFACTORY_DIR", str(tmp_path / "no-upstream"))
