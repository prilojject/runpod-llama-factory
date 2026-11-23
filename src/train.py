"""Model download, `llamafactory-cli train` subprocess, log/metrics parsing, output discovery."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from .paths import LLAMAFACTORY_DIR
from .secret_utils import redact

log = logging.getLogger(__name__)

WEIGHT_IGNORE = ["*.bin", "*.pth", "*.pt", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "original/*",
                 "*.ckpt", "consolidated*"]
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


class TrainingError(RuntimeError):
    pass


class ModelError(RuntimeError):
    pass


# --------------------------------------------------------------------------- GPU
def gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return {"gpu": None}
    if not out:
        return {"gpu": None}
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines()]
    name, mem, cc, drv = (rows[0] + [None] * 4)[:4]
    info = {"gpu": name, "gpu_count": len(rows), "vram_gb": round(float(mem) / 1024, 1) if mem else None,
            "compute_capability": cc, "driver": drv}
    try:
        info["bf16_supported"] = float(cc) >= 8.0
    except (TypeError, ValueError):
        info["bf16_supported"] = None
    return info


class VramSampler:
    """Sample nvidia-smi memory.used while training runs (peak VRAM of the whole GPU)."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                vals = [float(x) for x in out.split() if x.strip()]
                if vals:
                    self.peak_mib = max(self.peak_mib, max(vals))
            except Exception:
                pass

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

    @property
    def peak_gb(self) -> float | None:
        return round(self.peak_mib / 1024, 2) if self.peak_mib else None


# --------------------------------------------------------------------------- model
def ensure_model(model_id: str, revision: str | None, cache_dir: str, token: str | None,
                 trust_remote_code: bool, progress: Callable[[dict], None]) -> dict[str, Any]:
    """Download (or reuse) the base model snapshot and verify the weight file sizes."""
    if model_id.startswith("/"):
        if not os.path.isfile(os.path.join(model_id, "config.json")):
            raise ModelError(f"{model_id} has no config.json")
        return {"path": model_id, "source": "volume", "revision": None}
    from huggingface_hub import HfApi, snapshot_download

    t0 = time.time()
    try:
        info = HfApi(token=token).model_info(model_id, revision=revision, files_metadata=True)
    except Exception as exc:
        msg = redact(str(exc)).splitlines()[0][:300]
        raise ModelError(f"cannot read {model_id}@{revision or 'main'} on Hugging Face "
                         f"(gated/private models need HF_TOKEN with access): {msg}") from None
    files = {s.rfilename: s.size for s in (info.siblings or [])}
    has_st = any(f.endswith(".safetensors") for f in files)
    ignore = list(WEIGHT_IGNORE) if has_st else [p for p in WEIGHT_IGNORE if p != "*.bin"]
    if not trust_remote_code:
        ignore.append("*.py")
    weights = {f: s for f, s in files.items() if f.endswith(".safetensors") or (not has_st and f.endswith(".bin"))}
    total = sum(s or 0 for s in weights.values())
    progress({"stage": "downloading_model", "model": model_id, "size_gb": round(total / 2**30, 2)})
    try:
        path = snapshot_download(model_id, revision=info.sha or revision, cache_dir=cache_dir, token=token,
                                 ignore_patterns=ignore)
    except Exception as exc:
        raise ModelError(f"download of {model_id} failed: {redact(str(exc)).splitlines()[0][:300]}") from None
    bad = []
    for f, size in weights.items():
        p = os.path.join(path, f)
        if not os.path.isfile(p):
            if not any(f.startswith(x.rstrip("*")) for x in ("original/", "consolidated")):
                bad.append(f"{f}: missing")
        elif size and os.path.getsize(p) != size:
            bad.append(f"{f}: {os.path.getsize(p)} != {size} bytes")
    if bad:
        raise ModelError("model files failed the size check (partial download?): " + "; ".join(bad[:5]))
    return {"path": path, "source": "hf", "revision": info.sha, "size_gb": round(total / 2**30, 2),
            "download_seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- training
def build_command(yaml_path: str) -> list[str]:
    return ["llamafactory-cli", "train", yaml_path]


def read_trainer_log(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "trainer_log.jsonl")
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_training(cmd: list[str], output_dir: str, log_path: str, progress: Callable[[dict], None],
                 timeout_s: int | None = None, extra_env: dict[str, str] | None = None,
                 cwd: str | None = None) -> dict[str, Any]:
    """Run the CLI, stream + redact its log, report progress from trainer_log.jsonl, enforce a timeout."""
    log.info("launching: %s", " ".join(shlex.quote(c) for c in cmd))
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "DISABLE_VERSION_CHECK": "0"})
    env.pop("FORCE_TORCHRUN", None)
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    tail: list[str] = []
    timed_out = threading.Event()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    workdir = cwd or (LLAMAFACTORY_DIR if os.path.isdir(LLAMAFACTORY_DIR) else None)

    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, bufsize=1, errors="replace", start_new_session=True)

        def watchdog() -> None:
            if not timeout_s:
                return
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out.set()
                log.error("training exceeded %ss, terminating", timeout_s)
                _kill_tree(proc)

        stop = threading.Event()

        def reporter() -> None:
            last_step = -1
            while not stop.wait(15):
                entries = read_trainer_log(output_dir)
                payload: dict[str, Any] = {"stage": "training", "elapsed_s": round(time.time() - t0)}
                if entries:
                    e = entries[-1]
                    payload.update({k: e.get(k) for k in ("current_steps", "total_steps", "loss", "percentage",
                                                          "remaining_time") if e.get(k) is not None})
                    if e.get("current_steps") == last_step and time.time() - t0 < 60:
                        continue
                    last_step = e.get("current_steps", last_step)
                else:
                    payload["last_log"] = (tail or [""])[-1][:300]
                try:
                    progress(payload)
                except Exception:
                    log.debug("progress failed", exc_info=True)

        threading.Thread(target=watchdog, daemon=True).start()
        threading.Thread(target=reporter, daemon=True).start()
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = redact(raw.rstrip("\n"))
                for part in line.split("\r"):
                    if part.strip():
                        logf.write(part + "\n")
                        tail.append(part)
                if len(tail) > 200:
                    del tail[:-120]
                logf.flush()
        finally:
            stop.set()
        rc = proc.wait()

    elapsed = round(time.time() - t0, 1)
    if timed_out.is_set():
        raise TrainingError(f"training timed out after {timeout_s}s (TRAIN_TIMEOUT_SECONDS / timeout_seconds). "
                            "Last log lines:\n" + "\n".join(tail[-20:])[-3000:])
    if rc != 0:
        raise TrainingError(f"llamafactory-cli exited with code {rc}. Last log lines:\n"
                            + "\n".join(tail[-30:])[-4000:])
    return {"total_seconds": elapsed, **summarize_metrics(output_dir)}


def _kill_tree(proc: subprocess.Popen) -> None:
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def summarize_metrics(output_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    entries = [e for e in read_trainer_log(output_dir) if "current_steps" in e]
    if entries:
        out["global_step"] = entries[-1].get("current_steps")
        out["total_steps"] = entries[-1].get("total_steps")
        losses = [e["loss"] for e in entries if isinstance(e.get("loss"), (int, float))]
        if losses:
            out["first_loss"] = losses[0]
            out["last_loss"] = losses[-1]
    for fn in ("train_results.json", "all_results.json"):
        p = os.path.join(output_dir, fn)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    res = json.load(f)
                out["train_loss"] = res.get("train_loss")
                out["train_runtime_seconds"] = res.get("train_runtime")
                out["epoch"] = res.get("epoch")
                break
            except (OSError, json.JSONDecodeError):
                pass
    return out


# --------------------------------------------------------------------------- outputs
def find_checkpoints(output_dir: str) -> list[str]:
    if not os.path.isdir(output_dir):
        return []
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
    return [os.path.join(output_dir, d) for d in sorted(cks, key=lambda d: int(d.split("-")[-1]))]


def check_adapter(output_dir: str, finetuning_type: str) -> dict[str, Any]:
    """Verify the trained artifacts exist and are non-empty."""
    if finetuning_type == "lora":
        sizes = {}
        for f in ADAPTER_FILES:
            p = os.path.join(output_dir, f)
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                raise TrainingError(f"training finished but {f} is missing or empty in the output dir")
            sizes[f] = os.path.getsize(p)
        return {"adapter_dir": output_dir, "adapter_files": sizes,
                "adapter_size_mb": round(sizes["adapter_model.safetensors"] / 2**20, 2)}
    weights = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
    if not weights:
        raise TrainingError("training finished but no .safetensors weights were saved")
    total = sum(os.path.getsize(os.path.join(output_dir, f)) for f in weights)
    return {"model_dir": output_dir, "weight_files": sorted(weights), "size_gb": round(total / 2**30, 2)}


def fix_adapter_config(output_dir: str, model_id: str, revision: str | None) -> None:
    """Point adapter_config.json at the HF id (not the local cache path) so the adapter is shareable."""
    p = os.path.join(output_dir, "adapter_config.json")
    if not os.path.isfile(p) or model_id.startswith("/"):
        return
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = model_id
    if revision:
        cfg["revision"] = revision
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def remove_checkpoints(output_dir: str) -> int:
    n = 0
    for ck in find_checkpoints(output_dir):
        shutil.rmtree(ck, ignore_errors=True)
        n += 1
    return n
