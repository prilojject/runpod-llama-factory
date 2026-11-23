"""Deliver the adapter: network volume (always, when attached), tar.gz archive, HF Hub repo,
S3-compatible bucket (BUCKET_* env), presigned PUT URL."""

from __future__ import annotations

import logging
import os
import tarfile
from typing import Any

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
# What goes into the archive / HF repo (no checkpoints, no optimizer state, no copies of the data).
SKIP_DIRS = ("checkpoint-", "runs", "data")
SKIP_FILES = ("train.log", "training_args.bin")


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def deliverable_files(output_dir: str) -> list[str]:
    out = []
    for fn in sorted(os.listdir(output_dir)):
        p = os.path.join(output_dir, fn)
        if os.path.isdir(p) or fn.startswith(".") or fn in SKIP_FILES or fn.endswith(".tar.gz"):
            continue
        out.append(fn)
    return out


def make_archive(output_dir: str, archive_path: str) -> str:
    with tarfile.open(archive_path, "w:gz") as t:
        for fn in deliverable_files(output_dir):
            t.add(os.path.join(output_dir, fn), arcname=fn)
    return archive_path


def push_to_hf(repo_id: str, output_dir: str, token: str, private: bool, commit_message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=output_dir, commit_message=commit_message,
                      allow_patterns=deliverable_files(output_dir))
    return f"https://huggingface.co/{repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket

    return upload_file_to_bucket(file_name=os.path.basename(path), file_location=path,
                                 bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix)


def put_presigned(url: str, path: str) -> int:
    import requests

    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/gzip"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}")
    return r.status_code


def deliver(cfg, run_name: str, output_dir: str, token: str | None, persistent: bool,
            train: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {"output_dir": output_dir, "persistent_storage": persistent,
                           "files": deliverable_files(output_dir)}
    warnings: list[str] = []
    delivered = persistent
    archive = None
    if cfg.archive or cfg.upload_url or (cfg.upload_to_bucket and bucket_configured()):
        archive = make_archive(output_dir, os.path.join(output_dir, f"{run_name}-adapter.tar.gz"))
        out["archive"] = archive
        out["archive_size_mb"] = round(os.path.getsize(archive) / 2**20, 2)

    if cfg.hf_repo_id:
        if not token:
            warnings.append("hf_repo_id is set but HF_TOKEN is not configured (add it as a Runpod secret); "
                            "the adapter was not pushed")
        else:
            try:
                msg = f"LlamaFactory {cfg.stage} adapter ({run_name}) trained on Runpod Serverless"
                out["hf_repo_url"] = push_to_hf(cfg.hf_repo_id, output_dir, token, cfg.hf_private, msg)
                delivered = True
            except Exception as exc:
                out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
                log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url and archive:
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, archive)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured() and archive:
        try:
            out["bucket_url"] = upload_bucket(archive, f"llama-factory/{run_name}")
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append("the adapter was not delivered anywhere persistent: attach a network volume, or set "
                        "hf_repo_id (+HF_TOKEN secret), upload_url, or BUCKET_* env vars. It is lost when the "
                        "worker stops.")
    return out, warnings
