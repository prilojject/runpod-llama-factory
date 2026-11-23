"""Runpod Serverless entry point: headless LlamaFactory training (`llamafactory-cli train <yaml>`).

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset": "tiny_sft", "check_model": false}}'
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import LLAMAFACTORY_SHA, LLAMAFACTORY_TAG, __version__  # noqa: E402
from src.config import ConfigError, build_train_yaml, dump_yaml, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset  # noqa: E402
from src.paths import configure_caches, is_persistent, model_cache_dir, output_root, work_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _h in logging.getLogger().handlers:
    _h.addFilter(RedactFilter())
log = logging.getLogger("llama-factory-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod

                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def _check_model_meta(model_id: str, revision: str | None, token: str | None) -> dict[str, Any]:
    """Metadata-only check (no weights) that the base model is reachable."""
    if model_id.startswith("/"):
        return {"exists": os.path.isfile(os.path.join(model_id, "config.json")), "local": True}
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(model_id, revision=revision, files_metadata=True)
        size = sum((s.size or 0) for s in (info.siblings or []) if s.rfilename.endswith(".safetensors"))
        return {"exists": True, "sha": info.sha, "gated": bool(getattr(info, "gated", False)),
                "license": getattr(getattr(info, "card_data", None), "license", None),
                "weights_gb": round(size / 2**30, 2)}
    except Exception as exc:
        return {"exists": False, "error": redact(str(exc)).splitlines()[0][:300]}


def handler(job: dict) -> dict[str, Any]:
    try:
        result = _handle(job)
    except Exception as exc:  # last-resort guard: never leak a traceback with secrets
        log.error("unhandled error: %s", redact(traceback.format_exc()))
        result = {"error": f"internal error: {redact(str(exc)).splitlines()[0][:400] if str(exc) else type(exc).__name__}"}
    # Trainer: always start the next job on a fresh worker (frees VRAM, drops partial state).
    result["refresh_worker"] = True
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    run_dir = os.path.join(output_root(), run_name)
    work_dir = os.path.join(work_root(), run_name)
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = is_persistent(run_dir)
    result: dict[str, Any] = {
        "status": "ok",
        "worker_version": __version__,
        "llamafactory": {"tag": LLAMAFACTORY_TAG, "commit": LLAMAFACTORY_SHA},
        "run_name": run_name,
        "config": cfg.public_dict(),
        "config_sources": {k: v for k, v in cfg.sources.items() if v != "default"},
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append(
            "no network volume: the base model is re-downloaded on every cold start (~7.5 GiB for the default) "
            "and outputs live on the worker disk only; attach a volume to cache under /runpod-volume/llama-factory")

    # --- dataset (before any GPU work) ---------------------------------------
    try:
        progress({"stage": "preparing_dataset"})
        dataset_dir, report = prepare_dataset(cfg, work_dir)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", "run_name": run_name, "config": result["config"]}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}", "run_name": run_name}
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    if cfg.dry_run:
        preview_yaml = build_train_yaml(cfg, model_path=cfg.model_id, dataset_names=report.names,
                                        dataset_dir=dataset_dir, output_dir=run_dir, cache_dir=model_cache_dir())
        result.update({
            "dry_run": True,
            "train_yaml": dump_yaml(preview_yaml),
            "hf_token_configured": bool(token),
            "bucket_upload_configured": all(os.environ.get(k) for k in (
                "BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")),
            "persistent_storage": persistent,
        })
        if cfg.check_model:
            chk = _check_model_meta(cfg.model_id, cfg.model_revision, token)
            result["base_model_check"] = chk
            if not chk.get("exists"):
                result["status"] = "error"
                result["error"] = f"base model not reachable: {chk.get('error', 'not found')}"
        if cfg.hf_repo_id and not token:
            result["warnings"].append("hf_repo_id is set but HF_TOKEN is not configured")
        shutil.rmtree(work_dir, ignore_errors=True)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        return result

    from src.outputs import deliver
    from src.train import (
        ModelError,
        TrainingError,
        VramSampler,
        build_command,
        check_adapter,
        ensure_model,
        find_checkpoints,
        fix_adapter_config,
        gpu_info,
        run_training,
        summarize_metrics,
    )

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu.get("gpu") and os.environ.get("ALLOW_CPU_TRAINING") != "1":
        return {"error": "no CUDA GPU visible to the worker", "run_name": run_name, "gpu": gpu}
    if gpu.get("gpu") and cfg.train_args.get("bf16") and gpu.get("bf16_supported") is False:
        return {"error": f"{gpu['gpu']} has no bfloat16 support; deploy on Ampere/Ada (RTX 4090, L40S, A6000, A40)",
                "run_name": run_name, "gpu": gpu}

    # --- run dir / resume -----------------------------------------------------
    resume_ck = None
    if cfg.resume:
        cks = find_checkpoints(run_dir)
        if cks:
            resume_ck = cks[-1]
            result["resumed_from"] = os.path.basename(resume_ck)
        else:
            result["warnings"].append(f"resume=true but no checkpoint-* found in {run_dir}; starting fresh")
    elif os.path.isdir(run_dir) and os.listdir(run_dir):
        old = f"{run_dir}-old-{int(started)}"
        os.replace(run_dir, old)
        result["warnings"].append(f"run_name '{run_name}' already existed; previous outputs moved to {old}")
    meta_dir = os.path.join(run_dir, "worker")
    os.makedirs(meta_dir, exist_ok=True)

    # --- model ------------------------------------------------------------------
    try:
        model = ensure_model(cfg.model_id, cfg.model_revision, model_cache_dir(), token,
                             bool(cfg.train_args.get("trust_remote_code")), progress)
    except ModelError as exc:
        return {"error": str(exc), "run_name": run_name, "config": result["config"], "gpu": gpu}
    result["model"] = model
    result["config"]["model_revision"] = model.get("revision") or cfg.model_revision

    train_yaml = build_train_yaml(cfg, model_path=model["path"], dataset_names=report.names,
                                  dataset_dir=dataset_dir, output_dir=run_dir, cache_dir=model_cache_dir(),
                                  resume_checkpoint=resume_ck)
    yaml_path = os.path.join(meta_dir, "train_config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(dump_yaml(train_yaml))
    if cfg.user_yaml:
        with open(os.path.join(meta_dir, "user_config.yaml"), "w", encoding="utf-8") as f:
            f.write(dump_yaml(cfg.user_yaml))
    shutil.copy2(os.path.join(dataset_dir, "dataset_info.json"), os.path.join(meta_dir, "dataset_info.json"))
    result["config"]["yaml_path"] = yaml_path

    # --- train ------------------------------------------------------------------
    log_path = os.path.join(meta_dir, "train.log")
    timeout_s = cfg.timeout_seconds or None
    progress({"stage": "training", "yaml": yaml_path})
    try:
        with VramSampler() as vram:
            train = run_training(build_command(yaml_path), run_dir, log_path, progress, timeout_s=timeout_s)
        train["peak_vram_gb"] = vram.peak_gb
        arts = check_adapter(run_dir, cfg.train_args.get("finetuning_type", "lora"))
    except TrainingError as exc:
        partial = summarize_metrics(run_dir)
        return {"error": redact(str(exc)), "run_name": run_name, "config": result["config"], "gpu": gpu,
                "training": partial, "log_path": log_path, "warnings": result["warnings"],
                "checkpoints": [os.path.basename(c) for c in find_checkpoints(run_dir)]}
    fix_adapter_config(run_dir, cfg.model_id, result["config"]["model_revision"])
    result["training"] = train

    # --- deliver ------------------------------------------------------------------
    progress({"stage": "delivering"})
    delivery, dwarn = deliver(cfg, run_name, run_dir, token, persistent, train)
    result["artifacts"] = {**arts, **delivery, "log_path": log_path}
    result["warnings"] += dwarn
    with open(os.path.join(meta_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "warnings"}, f, indent=2, default=str)
    shutil.rmtree(work_dir, ignore_errors=True)
    result["elapsed_seconds"] = round(time.time() - started, 1)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
