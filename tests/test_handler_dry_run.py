import json
import os

import handler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_dry_run_bundled_dataset_builds_yaml():
    with open(os.path.join(ROOT, "test_input.json")) as f:
        job = {"id": "local-test", **json.load(f)}
    r = handler.handler(job)
    assert r["status"] == "ok" and r["dry_run"] is True and r["refresh_worker"] is True
    assert "error" not in r
    y = r["train_yaml"]
    for line in ("stage: sft", "finetuning_type: lora", "quantization_bit: 4", "quantization_method: bnb",
                 "template: qwen3_nothink", "bf16: true", "use_unsloth: false", "report_to: none",
                 "max_steps: 5", "max_samples: 16", "cutoff_len: 512", "dataset: tiny_sft"):
        assert line in y, line
    assert r["dataset"]["rows"] == {"tiny_sft": 16}
    assert r["config"]["model_revision"] == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert any("no network volume" in w for w in r["warnings"])


def test_invalid_input_returns_error_not_exception():
    r = handler.handler({"id": "local", "input": {"dataset": "tiny_sft", "hf_token": "x"}})
    assert r["error"].startswith("invalid input") and r["refresh_worker"] is True
    r = handler.handler({"id": "local", "input": {"dataset": "tiny_sft", "stage": "dpo", "dry_run": True,
                                                  "check_model": False}})
    assert r["error"].startswith("dataset:")


def test_no_gpu_refuses_training(monkeypatch):
    import src.train as tr
    monkeypatch.setattr(tr, "gpu_info", lambda: {"gpu": None})
    monkeypatch.delenv("ALLOW_CPU_TRAINING", raising=False)
    r = handler.handler({"id": "local", "input": {"dataset": "tiny_sft", "check_model": False}})
    assert r["error"] == "no CUDA GPU visible to the worker"


def test_dpo_dry_run():
    r = handler.handler({"id": "local", "input": {"dataset": "tiny_dpo", "stage": "dpo", "pref_loss": "orpo",
                                                  "dry_run": True, "check_model": False}})
    assert r["status"] == "ok" and "pref_loss: orpo" in r["train_yaml"] and "stage: dpo" in r["train_yaml"]
