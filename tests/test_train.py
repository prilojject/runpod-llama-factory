import json
import os
import sys

import pytest

from src.train import (
    TrainingError,
    build_command,
    check_adapter,
    find_checkpoints,
    fix_adapter_config,
    run_training,
    summarize_metrics,
)


def test_build_command_is_official_cli():
    assert build_command("/r/worker/train_config.yaml") == ["llamafactory-cli", "train", "/r/worker/train_config.yaml"]


def _fake_cli(tmp_path, body):
    p = tmp_path / "fake_cli.py"
    p.write_text("import json, os, sys, time\nout = sys.argv[1]\n" + body)
    return [sys.executable, str(p), str(tmp_path / "out")]


def test_run_training_parses_metrics_and_redacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    import src.secret_utils  # noqa: F401  (RedactFilter reads env at call time)
    cmd = _fake_cli(tmp_path, """
os.makedirs(out, exist_ok=True)
print('loading with token', os.environ['HF_TOKEN'])
with open(os.path.join(out, 'trainer_log.jsonl'), 'w') as f:
    for s in range(1, 6):
        f.write(json.dumps({'current_steps': s, 'total_steps': 5, 'loss': 2.0 - s * 0.1}) + '\\n')
json.dump({'train_loss': 1.7, 'train_runtime': 12.5, 'epoch': 2.5}, open(os.path.join(out, 'train_results.json'), 'w'))
""")
    seen = []
    res = run_training(cmd, str(tmp_path / "out"), str(tmp_path / "train.log"), seen.append, timeout_s=60,
                       cwd=str(tmp_path))
    assert res["global_step"] == 5 and res["total_steps"] == 5
    assert res["first_loss"] == pytest.approx(1.9) and res["last_loss"] == pytest.approx(1.5)
    assert res["train_loss"] == 1.7 and res["train_runtime_seconds"] == 12.5
    log = open(tmp_path / "train.log").read()
    assert "zzzzzzzz" not in log and "***" in log


def test_run_training_failure_and_timeout(tmp_path):
    with pytest.raises(TrainingError, match="exited with code 3"):
        run_training(_fake_cli(tmp_path, "print('CUDA out of memory'); sys.exit(3)"), str(tmp_path / "o"),
                     str(tmp_path / "l.log"), lambda p: None, cwd=str(tmp_path))
    with pytest.raises(TrainingError, match="timed out"):
        run_training(_fake_cli(tmp_path, "time.sleep(30)"), str(tmp_path / "o"), str(tmp_path / "l2.log"),
                     lambda p: None, timeout_s=1, cwd=str(tmp_path))


def test_check_adapter_requires_nonempty_files(tmp_path):
    with pytest.raises(TrainingError, match="adapter_model.safetensors"):
        check_adapter(str(tmp_path), "lora")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(TrainingError, match="empty"):
        check_adapter(str(tmp_path), "lora")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x" * 2048)
    a = check_adapter(str(tmp_path), "lora")
    assert a["adapter_files"]["adapter_model.safetensors"] == 2048


def test_fix_adapter_config_points_to_hf_id(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/cache/snap/abc", "r": 8}))
    fix_adapter_config(str(tmp_path), "Qwen/Qwen3-4B-Instruct-2507", "cdbee75")
    c = json.load(open(tmp_path / "adapter_config.json"))
    assert c["base_model_name_or_path"] == "Qwen/Qwen3-4B-Instruct-2507" and c["revision"] == "cdbee75"


def test_checkpoints_sorted_numerically(tmp_path):
    for n in (10, 2, 100):
        os.makedirs(tmp_path / f"checkpoint-{n}")
    os.makedirs(tmp_path / "checkpoint-x")
    assert [os.path.basename(p) for p in find_checkpoints(str(tmp_path))] == [
        "checkpoint-2", "checkpoint-10", "checkpoint-100"]


def test_summarize_metrics_empty(tmp_path):
    assert summarize_metrics(str(tmp_path)) == {}
