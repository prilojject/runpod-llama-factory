"""Resolve job input + Hub env into one LlamaFactory training YAML.

Priority: job input (flat fields > config_yaml/config_url) > env (Hub form) > preset > defaults.
The worker always owns the service paths (output_dir, dataset_dir, cache_dir, ...) and never
lets a token travel through job input or YAML.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Pinned HF revision of the default weights (lastModified 2025-09-17). Never float on main.
DEFAULT_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DEFAULT_TEMPLATE = "qwen3_nothink"

STAGES = {"sft", "dpo", "kto"}
FINETUNING_TYPES = {"lora", "freeze", "full"}
PREF_LOSSES = {"sigmoid", "orpo", "simpo"}  # ORPO/SimPO are pref_loss values of stage dpo
DATASET_FORMATS = {"auto", "alpaca", "sharegpt"}
QUANT_BITS = {4, 8}

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
TOKEN_VALUE_RE = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b")
SECRETISH_KEY_RE = re.compile(r"(token|secret|password|api_key|apikey|access_key)", re.I)

# Keys the worker sets itself; user values are dropped with a warning.
WORKER_OWNED = {
    "output_dir", "dataset_dir", "cache_dir", "media_dir", "tokenized_path", "logging_dir",
    "overwrite_output_dir", "resume_from_checkpoint", "report_to", "do_train", "do_eval",
    "do_predict",
}
# Keys that are refused outright (would bypass the worker, write elsewhere, or need extras
# that are not installed in this image).
FORBIDDEN = {
    "hf_hub_token", "ms_hub_token", "om_hub_token", "hub_token", "push_to_hub", "hub_model_id",
    "hub_strategy", "export_dir", "export_hub_model_id", "deepspeed", "use_ray", "ray_run_name",
    "ray_storage_path", "ray_num_workers", "ray_init_kwargs", "use_mca", "use_v1_kernels",
    "use_ms_hub", "use_om_hub", "predict_with_generate",
}
WORKER_KEYS = {
    # job-only keys that are not LlamaFactory arguments
    "config_yaml", "config_url", "dataset_url", "dataset_inline", "dataset_format", "run_name",
    "hf_repo_id", "hf_private", "resume", "dry_run", "check_model", "upload_url", "upload_to_bucket",
    "archive", "extra_args", "timeout_seconds",
}

MAX_YAML_BYTES = 256 * 1024


class ConfigError(ValueError):
    pass


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"not a boolean: {v!r}")


def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ConfigError(f"not an integer: {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"not an integer: {v!r}") from None
    if f != int(f):
        raise ConfigError(f"not an integer: {v!r}")
    return int(f)


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ConfigError(f"not a number: {v!r}")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"not a number: {v!r}") from None


def _quant(v: Any) -> int | None:
    if v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null", "0", "16", "off"}):
        return None
    if v == 0 or v == 16:
        return None
    b = _to_int(v)
    if b not in QUANT_BITS:
        raise ConfigError(f"quantization_bit must be 4, 8 or null (16-bit LoRA), got {v!r}")
    return b


@dataclass(frozen=True)
class Param:
    name: str  # LlamaFactory key (or worker key)
    env: str | None
    conv: Callable[[Any], Any]
    lo: float | None = None
    hi: float | None = None
    choices: frozenset | None = None


PARAMS: list[Param] = [
    Param("model_name_or_path", "MODEL_NAME_OR_PATH", str),
    Param("model_revision", "MODEL_REVISION", str),
    Param("template", "TEMPLATE", str),
    Param("stage", "STAGE", str),  # validated below with an ORPO hint
    Param("finetuning_type", "FINETUNING_TYPE", str, choices=frozenset(FINETUNING_TYPES)),
    Param("quantization_bit", "QUANTIZATION_BIT", _quant),
    Param("lora_rank", "LORA_RANK", _to_int, 1, 1024),
    Param("lora_alpha", "LORA_ALPHA", _to_int, 1, 4096),
    Param("lora_dropout", None, _to_float, 0, 0.9),
    Param("lora_target", None, str),
    Param("pref_loss", "PREF_LOSS", str, choices=frozenset(PREF_LOSSES)),
    Param("pref_beta", None, _to_float, 0, 100),
    Param("dataset", "DATASET", str),
    Param("dataset_url", "DATASET_URL", str),
    Param("dataset_format", "DATASET_FORMAT", str, choices=frozenset(DATASET_FORMATS)),
    Param("cutoff_len", "CUTOFF_LEN", _to_int, 16, 131072),
    Param("max_samples", "MAX_SAMPLES", _to_int, 0, 10**9),
    Param("num_train_epochs", "NUM_TRAIN_EPOCHS", _to_float, 0.01, 1000),
    Param("max_steps", "MAX_STEPS", _to_int, 0, 10**7),
    Param("learning_rate", "LEARNING_RATE", _to_float, 1e-9, 1.0),
    Param("per_device_train_batch_size", "BATCH_SIZE", _to_int, 1, 1024),
    Param("gradient_accumulation_steps", "GRAD_ACCUM", _to_int, 1, 4096),
    Param("lr_scheduler_type", None, str),
    Param("warmup_ratio", None, _to_float, 0, 1),
    Param("logging_steps", None, _to_int, 1, 10**6),
    Param("save_steps", "SAVE_STEPS", _to_int, 1, 10**7),
    Param("bf16", None, _to_bool),
    Param("seed", None, _to_int, 0, 2**31 - 1),
    Param("trust_remote_code", "TRUST_REMOTE_CODE", _to_bool),
    Param("enable_thinking", None, _to_bool),
    Param("hf_repo_id", "HF_OUTPUT_REPO", str),
    Param("hf_private", "HF_PRIVATE", _to_bool),
    Param("timeout_seconds", "TRAIN_TIMEOUT_SECONDS", _to_int, 0, 7 * 86400),
    Param("dry_run", "DRY_RUN", _to_bool),
    Param("archive", "ARCHIVE_ADAPTER", _to_bool),
    Param("upload_to_bucket", "UPLOAD_TO_BUCKET", _to_bool),
]
PARAM_BY_NAME = {p.name: p for p in PARAMS}
WORKER_PARAMS = {"dataset_url", "dataset_format", "hf_repo_id", "hf_private", "timeout_seconds", "dry_run",
                 "archive", "upload_to_bucket", "model_revision"}

BASE_DEFAULTS: dict[str, Any] = {
    "model_name_or_path": DEFAULT_MODEL,
    "quantization_bit": 4,
    "quantization_method": "bnb",
    "trust_remote_code": False,
    "use_unsloth": False,
    "stage": "sft",
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_target": "all",
    "template": DEFAULT_TEMPLATE,
    "cutoff_len": 2048,
    "preprocessing_num_workers": 4,
    "dataloader_num_workers": 2,
    "logging_steps": 5,
    "save_steps": 500,
    "plot_loss": True,
    "save_only_model": False,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "ddp_timeout": 180000000,
}
STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "sft": {"learning_rate": 1.0e-4},
    "dpo": {"learning_rate": 5.0e-6, "pref_beta": 0.1, "pref_loss": "sigmoid"},
    "kto": {"learning_rate": 5.0e-6, "pref_beta": 0.1},
}
WORKER_DEFAULTS: dict[str, Any] = {
    "dataset_format": "auto",
    "hf_private": True,
    "timeout_seconds": 6 * 3600,
    "dry_run": False,
    "archive": True,
    "upload_to_bucket": True,
    "resume": False,
    "check_model": True,
}


@dataclass
class ResolvedConfig:
    train_args: dict[str, Any]  # LlamaFactory YAML keys (service paths added later)
    model_id: str
    model_revision: str | None
    dataset_names: list[str]
    dataset_url: str | None
    dataset_inline: list | None
    dataset_format: str
    run_name: str | None
    hf_repo_id: str | None
    hf_private: bool
    resume: bool
    dry_run: bool
    check_model: bool
    upload_url: str | None
    upload_to_bucket: bool
    archive: bool
    timeout_seconds: int
    user_yaml: dict[str, Any] | None
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def stage(self) -> str:
        return self.train_args["stage"]

    @property
    def quantization_bit(self) -> int | None:
        return self.train_args.get("quantization_bit")

    @property
    def dtype(self) -> str:
        if self.train_args.get("bf16"):
            return "bfloat16"
        if self.train_args.get("fp16"):
            return "float16"
        return "float32"

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "model_revision": self.model_revision,
            "stage": self.stage,
            "pref_loss": self.train_args.get("pref_loss") if self.stage == "dpo" else None,
            "finetuning_type": self.train_args.get("finetuning_type"),
            "quantization_bit": self.quantization_bit,
            "quantization_method": self.train_args.get("quantization_method") if self.quantization_bit else None,
            "dtype": self.dtype,
            "template": self.train_args.get("template"),
            "dataset": self.dataset_names,
            "dataset_url": _strip_query(self.dataset_url) if self.dataset_url else None,
            "dataset_format": self.dataset_format,
            "max_steps": self.train_args.get("max_steps"),
            "num_train_epochs": self.train_args.get("num_train_epochs"),
            "max_samples": self.train_args.get("max_samples"),
            "cutoff_len": self.train_args.get("cutoff_len"),
            "lora_rank": self.train_args.get("lora_rank") if self.train_args.get("finetuning_type") == "lora" else None,
            "learning_rate": self.train_args.get("learning_rate"),
            "use_unsloth": bool(self.train_args.get("use_unsloth")),
            "hf_repo_id": self.hf_repo_id,
            "resume": self.resume,
        }


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


ALLOWED_TOKENISH_KEYS = {"new_special_tokens", "add_tokens", "add_special_tokens", "new_special_tokens_config",
                         "max_new_tokens", "skip_special_tokens", "resize_vocab"}


def _check_no_secrets(obj: Any, where: str) -> None:
    """Refuse tokens in job input / YAML: they belong in Runpod secrets (env)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if SECRETISH_KEY_RE.search(ks) and ks not in ALLOWED_TOKENISH_KEYS and not ks.endswith("_tokens"):
                raise ConfigError(f"{where}.{ks}: tokens/secrets are not accepted in job input; "
                                  "set HF_TOKEN as a Runpod secret in the endpoint env instead")
            _check_no_secrets(v, f"{where}.{ks}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            _check_no_secrets(v, f"{where}[{i}]")
    elif isinstance(obj, str) and TOKEN_VALUE_RE.search(obj):
        raise ConfigError(f"{where}: looks like an access token; tokens are not accepted in job input, "
                          "use a Runpod secret (HF_TOKEN) instead")


def parse_yaml_text(text: str, where: str = "config_yaml") -> dict[str, Any]:
    if len(text.encode("utf-8", "replace")) > MAX_YAML_BYTES:
        raise ConfigError(f"{where} is larger than {MAX_YAML_BYTES // 1024} KiB")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{where} is not valid YAML: {str(exc).splitlines()[0]}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a YAML mapping of LlamaFactory arguments")
    return {str(k): v for k, v in data.items()}


def fetch_config_url(url: str) -> str:
    import requests

    if not re.match(r"^https?://", url):
        raise ConfigError("config_url must be an http(s) URL")
    r = requests.get(url, timeout=(10, 60), stream=True)
    if r.status_code >= 400:
        raise ConfigError(f"config_url returned HTTP {r.status_code}")
    body = b""
    for chunk in r.iter_content(65536):
        body += chunk
        if len(body) > MAX_YAML_BYTES:
            raise ConfigError("config_url body is too large")
    return body.decode("utf-8", "replace")


def _convert(p: Param, value: Any, src: str) -> Any:
    try:
        v = p.conv(value)
    except ConfigError as exc:
        raise ConfigError(f"{p.name} ({src}): {exc}") from None
    if isinstance(v, str):
        v = v.strip()
    if p.choices is not None and v not in p.choices:
        raise ConfigError(f"{p.name} ({src}) must be one of {sorted(p.choices)}, got {v!r}")
    if p.lo is not None and v is not None and not (p.lo <= v <= p.hi):
        raise ConfigError(f"{p.name} ({src}) must be in [{p.lo}, {p.hi}], got {v!r}")
    return v


def _env_value(p: Param, env: dict[str, str]) -> Any:
    if not p.env:
        return None
    raw = env.get(p.env)
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw == "" or raw.startswith("{{"):
        return None
    if p.name == "max_steps" and raw in {"0", "0.0"}:
        return None
    return raw


def resolve_config(job_input: dict[str, Any], env: dict[str, str] | None = None,
                   yaml_fetcher: Callable[[str], str] = fetch_config_url) -> ResolvedConfig:
    env = dict(os.environ) if env is None else env
    if not isinstance(job_input, dict):
        raise ConfigError("input must be a JSON object")
    _check_no_secrets(job_input, "input")
    warnings: list[str] = []
    sources: dict[str, str] = {}

    # 1) user YAML (config_url first, config_yaml on top)
    user_yaml: dict[str, Any] = {}
    if job_input.get("config_url"):
        user_yaml.update(parse_yaml_text(yaml_fetcher(str(job_input["config_url"])), "config_url"))
    if job_input.get("config_yaml"):
        cy = job_input["config_yaml"]
        user_yaml.update(cy if isinstance(cy, dict) else parse_yaml_text(str(cy)))
    _check_no_secrets(user_yaml, "config_yaml")

    extra = job_input.get("extra_args") or {}
    if not isinstance(extra, dict):
        raise ConfigError("extra_args must be an object of LlamaFactory arguments")

    # 2) job flat fields (override YAML, like CLI overrides in llamafactory-cli)
    flat: dict[str, Any] = {}
    unknown = []
    for k, v in job_input.items():
        if k in WORKER_KEYS and k not in PARAM_BY_NAME:
            continue
        if k in PARAM_BY_NAME:
            if v is None and k != "quantization_bit":
                continue
            flat[k] = v
        else:
            unknown.append(k)
    if unknown:
        raise ConfigError(f"unknown input field(s): {sorted(unknown)}; put extra LlamaFactory "
                          "arguments into config_yaml or extra_args")

    job_layer: dict[str, Any] = {}
    for k, v in {**user_yaml, **extra}.items():
        src = "extra_args" if k in extra else "config_yaml"
        if k in PARAM_BY_NAME and k != "dataset" and (v is not None or k == "quantization_bit"):
            v = _convert(PARAM_BY_NAME[k], v, src)
        job_layer[k] = v
        sources[k] = src
    for k, v in flat.items():
        job_layer[k] = _convert(PARAM_BY_NAME[k], v, "input")
        sources[k] = "input"

    for k in list(job_layer):
        if k in FORBIDDEN:
            raise ConfigError(f"'{k}' is not supported by this worker"
                              + (" (adapter delivery to the Hub is done with hf_repo_id + HF_TOKEN secret)"
                                 if "hub" in k else ""))
        if k in WORKER_OWNED:
            warnings.append(f"'{k}' is set by the worker; your value was ignored")
            job_layer.pop(k)
            sources.pop(k, None)
    if _to_bool(job_layer.get("use_unsloth", False)):
        raise ConfigError("use_unsloth=true is not supported: this worker runs the stock LlamaFactory "
                          "trainer (Unsloth is not installed). For Unsloth use prilojject/runpod-unsloth-finetune")

    # 3) env (Hub form) for keys the job did not set
    env_layer: dict[str, Any] = {}
    job_has_data = any(k in job_layer for k in ("dataset", "dataset_url")) or job_input.get("dataset_inline") is not None
    for p in PARAMS:
        if p.name in job_layer:
            continue
        if job_has_data and p.name in {"dataset", "dataset_url"}:
            continue  # the job brought its own data: do not mix in the endpoint default dataset
        if p.name == "model_revision" and "model_name_or_path" in job_layer:
            continue  # an env revision belongs to the env model, not to a model chosen per job
        raw = _env_value(p, env)
        if raw is not None:
            env_layer[p.name] = _convert(p, raw, f"env {p.env}")
            sources[p.name] = f"env:{p.env}"

    merged = {**env_layer, **job_layer}

    # 4) defaults
    stage = merged.get("stage") or BASE_DEFAULTS["stage"]
    if stage not in STAGES:
        raise ConfigError(f"stage must be one of {sorted(STAGES)}; ORPO/SimPO are stage=dpo with "
                          f"pref_loss=orpo|simpo (got {stage!r})")
    args: dict[str, Any] = {**BASE_DEFAULTS, **STAGE_DEFAULTS[stage]}
    for k, v in merged.items():
        if k in WORKER_PARAMS:
            continue
        args[k] = v
    for k in args:
        sources.setdefault(k, "default")

    worker = {**WORKER_DEFAULTS}
    for k in WORKER_PARAMS | {"resume", "check_model"}:
        if k in merged:
            worker[k] = merged[k]
    for k in ("resume", "check_model"):
        if k in job_input:
            worker[k] = _to_bool(job_input[k])

    # --- model + revision ---------------------------------------------------
    model_id = str(args["model_name_or_path"]).strip()
    if not model_id:
        raise ConfigError("model_name_or_path is empty")
    if model_id.startswith("/"):
        if not model_id.startswith("/runpod-volume/"):
            raise ConfigError("local model paths must be under /runpod-volume/")
    elif not HF_ID_RE.match(model_id):
        raise ConfigError(f"model_name_or_path must be a Hugging Face id (owner/name), got {model_id!r}")
    revision = merged.get("model_revision")
    if revision is not None:
        revision = str(revision).strip() or None
    if revision is None and model_id == DEFAULT_MODEL:
        revision = DEFAULT_MODEL_REVISION
    if revision and not re.match(r"^[A-Za-z0-9._/-]{1,100}$", revision):
        raise ConfigError("model_revision has invalid characters")
    if model_id != DEFAULT_MODEL and not model_id.startswith("/") and revision is None:
        warnings.append(f"{model_id} is not pinned to a revision (model_revision); the current 'main' is used")

    if "template" not in merged and "qwen3" not in model_id.lower():
        raise ConfigError(f"template is required for {model_id}: the default '{DEFAULT_TEMPLATE}' only fits "
                          "Qwen3 models (see LlamaFactory's supported models table)")

    # --- method consistency -------------------------------------------------
    ft = args.get("finetuning_type")
    if ft not in FINETUNING_TYPES:
        raise ConfigError(f"finetuning_type must be one of {sorted(FINETUNING_TYPES)}")
    q = args.get("quantization_bit")
    if q is not None:
        q = _quant(q)
        args["quantization_bit"] = q
    if q is None:
        args.pop("quantization_bit", None)
        args.pop("quantization_method", None)
    elif ft != "lora":
        raise ConfigError("quantization (QLoRA) only works with finetuning_type=lora; "
                          "set quantization_bit to null for freeze/full tuning")
    if stage == "dpo":
        pl = args.get("pref_loss", "sigmoid")
        if pl not in PREF_LOSSES:
            raise ConfigError(f"pref_loss must be one of {sorted(PREF_LOSSES)}")
    else:
        if sources.get("pref_loss") in {"input", "config_yaml", "extra_args"}:
            raise ConfigError("pref_loss only applies to stage=dpo (ORPO = stage dpo + pref_loss orpo)")
        if stage == "sft":
            args.pop("pref_beta", None)
        args.pop("pref_loss", None)
    if ft == "full":
        warnings.append("full fine-tuning of a 4B model needs ~48 GB+ VRAM; 24 GB GPUs will OOM")
    if args.get("bf16") and args.get("fp16"):
        raise ConfigError("set only one of bf16 / fp16")
    if not args.get("bf16") and model_id == DEFAULT_MODEL:
        warnings.append("Qwen3-4B-Instruct-2507 weights are bfloat16; bf16=false is slower and not recommended")
    ms = args.get("max_steps")
    if ms is not None:
        ms = _to_int(ms)
        if ms <= 0:
            args.pop("max_steps")
        else:
            args["max_steps"] = ms
    mx = args.get("max_samples")
    if mx is not None and _to_int(mx) <= 0:
        args.pop("max_samples")

    # --- dataset --------------------------------------------------------------
    inline = job_input.get("dataset_inline")
    if inline is not None:
        if not isinstance(inline, list) or not inline:
            raise ConfigError("dataset_inline must be a non-empty list of records")
        if len(inline) > 5000:
            raise ConfigError("dataset_inline is limited to 5000 records; use dataset_url for more")
    dataset_url = worker.get("dataset_url")
    if dataset_url:
        dataset_url = str(dataset_url).strip()
        if not re.match(r"^https?://", dataset_url):
            raise ConfigError("dataset_url must be an http(s) URL")
    ds = args.pop("dataset", None)
    names: list[str] = []
    if isinstance(ds, list):
        names = [str(x).strip() for x in ds if str(x).strip()]
    elif ds:
        names = [x.strip() for x in str(ds).split(",") if x.strip()]
    if inline is not None and dataset_url:
        raise ConfigError("send either dataset_inline or dataset_url, not both")
    if not names and inline is None and not dataset_url:
        raise ConfigError("no training data: set dataset_url, dataset_inline, or dataset "
                          "(HF dataset id, or bundled 'tiny_sft' / 'tiny_dpo' / 'tiny_kto')")
    fmt = worker.get("dataset_format", "auto")
    if fmt not in DATASET_FORMATS:
        raise ConfigError(f"dataset_format must be one of {sorted(DATASET_FORMATS)}")
    for n in names:
        if "alpaca" in n.lower() and "/" not in n:
            warnings.append(f"dataset '{n}' is derived from Stanford Alpaca (CC BY-NC 4.0): non-commercial only")

    # --- worker options -----------------------------------------------------
    run_name = job_input.get("run_name")
    if run_name is not None:
        run_name = str(run_name).strip()
        if not RUN_NAME_RE.match(run_name):
            raise ConfigError("run_name must be 1-64 chars of letters, digits, '.', '_' or '-'")
    resume = bool(worker["resume"])
    if resume and not run_name:
        raise ConfigError("resume=true needs the run_name of the run to continue")
    hf_repo = worker.get("hf_repo_id")
    if hf_repo:
        hf_repo = str(hf_repo).strip()
        if not HF_ID_RE.match(hf_repo):
            raise ConfigError("hf_repo_id must look like 'owner/name'")
    else:
        hf_repo = None
    upload_url = job_input.get("upload_url")
    if upload_url is not None:
        upload_url = str(upload_url).strip() or None
        if upload_url and not upload_url.startswith("https://"):
            raise ConfigError("upload_url must be an https presigned PUT URL")

    return ResolvedConfig(
        train_args=args,
        model_id=model_id,
        model_revision=revision,
        dataset_names=names,
        dataset_url=dataset_url or None,
        dataset_inline=inline,
        dataset_format=fmt,
        run_name=run_name,
        hf_repo_id=hf_repo,
        hf_private=_to_bool(worker["hf_private"]),
        resume=resume,
        dry_run=_to_bool(worker["dry_run"]),
        check_model=_to_bool(worker["check_model"]),
        upload_url=upload_url,
        upload_to_bucket=_to_bool(worker["upload_to_bucket"]),
        archive=_to_bool(worker["archive"]),
        timeout_seconds=_to_int(worker["timeout_seconds"]) or WORKER_DEFAULTS["timeout_seconds"],
        user_yaml=user_yaml or None,
        sources=sources,
        warnings=warnings,
    )


def build_train_yaml(cfg: ResolvedConfig, *, model_path: str, dataset_names: list[str], dataset_dir: str,
                     output_dir: str, cache_dir: str, resume_checkpoint: str | None = None) -> dict[str, Any]:
    """Final LlamaFactory YAML: user/default args + worker-owned service paths."""
    y = dict(cfg.train_args)
    y["model_name_or_path"] = model_path
    y["dataset"] = ",".join(dataset_names)
    y["dataset_dir"] = dataset_dir
    y["media_dir"] = dataset_dir
    y["cache_dir"] = cache_dir
    y["output_dir"] = output_dir
    y["do_train"] = True
    y["report_to"] = "none"
    y["use_unsloth"] = False
    if resume_checkpoint:
        y["resume_from_checkpoint"] = resume_checkpoint
        y["overwrite_output_dir"] = False
    else:
        y["overwrite_output_dir"] = True
    return y


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
