# LLaMA-Factory Trainer — Runpod Serverless worker

Headless [LlamaFactory](https://github.com/hiyouga/LlamaFactory) **v0.9.5** fine-tuning on Runpod Serverless.
Send an **official LlamaFactory YAML** (or a few form fields) plus an alpaca / sharegpt dataset; get a
**LoRA / QLoRA adapter** (`adapter_model.safetensors` + `adapter_config.json` + tokenizer) on your network
volume, Hugging Face repo, or S3 bucket.

- Stages: **SFT**, **DPO**, **ORPO** and **SimPO** (`stage: dpo` + `pref_loss`), **KTO**
- Methods: **QLoRA** 4/8-bit (bitsandbytes), 16-bit **LoRA**, **freeze**, **full**
- Default: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, not gated, bf16) at a pinned revision, QLoRA 4-bit,
  template `qwen3_nothink`, `lora_rank: 8`, on one **24 GB** GPU (RTX 4090, L4, A5000)
- The worker writes the YAML and runs the official `llamafactory-cli train`. No Gradio / LLaMA Board
  (gradio is not even installed), no weights baked into the image.

> Listing status: TODO (set after Hub moderation). Measured GPU cost: TODO (filled after the GPU smoke test).

## How this differs from `runpod-unsloth-finetune`

| | this worker (LlamaFactory) | [runpod-unsloth-finetune](https://github.com/prilojject/runpod-unsloth-finetune) |
|---|---|---|
| Config | official LlamaFactory YAML first-class, or form fields | JSON fields only |
| Stages | SFT, DPO, ORPO, SimPO, KTO | SFT only |
| Methods | LoRA, QLoRA, freeze, full | LoRA, QLoRA |
| Models | anything LlamaFactory supports (100+ families, chat templates incl. VLMs) | causal LMs Unsloth supports |
| Runtime | stock Hugging Face / PEFT / TRL trainers (`use_unsloth` is always false) | Unsloth kernels (faster SFT) |

Pick Unsloth for the fastest plain SFT; pick this one for YAML configs, preference tuning, or models and
templates Unsloth does not cover.

## What you send / what you get

Send (async `POST /run`) any of:

- `config_yaml` (string or object) / `config_url` — a LlamaFactory training YAML (`examples/configs/`)
- flat fields that override the YAML: `model_name_or_path`, `model_revision`, `template`, `stage`,
  `pref_loss`, `finetuning_type`, `quantization_bit` (`4` default, `null` = 16-bit LoRA), `lora_rank`,
  `cutoff_len`, `max_samples`, `num_train_epochs`, `max_steps`, `learning_rate`, ...
- data: `dataset_url` (.json/.jsonl/.csv/.parquet or .zip/.tar.gz, up to 2 GiB), `dataset_inline`
  (list of records), or `dataset` (public HF dataset id, bundled `tiny_sft` / `tiny_dpo` / `tiny_kto`, or an
  entry of LlamaFactory's `data/dataset_info.json`)
- worker options: `run_name`, `hf_repo_id`, `resume`, `dry_run`, `upload_url`, `extra_args`

You get JSON with `status`, `worker_version`, `gpu`, `config` (model, revision, stage, dtype, quantization,
YAML path), `training` (steps, first/last loss, seconds, peak VRAM), `artifacts` (adapter dir, archive,
HF URL), `warnings`, `elapsed_seconds`.

Priority: job input fields > `config_yaml` > endpoint env (Hub form) > preset > defaults.
The worker always sets `output_dir` (`/runpod-volume/llama-factory/runs/<run_name>/`), `dataset_dir`,
`cache_dir` and `report_to: none`; tokens are refused in job input and YAML.

## Quick start

```bash
# 1) validate data + YAML without a GPU job doing any training
curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' -d '{"input": {"dry_run": true, "dataset": "tiny_sft"}}'

# 2) a 5-step QLoRA smoke run on the bundled MIT dataset
curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H 'Content-Type: application/json' -d @examples/requests/smoke_test_request.json

# 3) poll
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job id> -H "Authorization: Bearer $RUNPOD_API_KEY"
```

ORPO with a YAML: `examples/requests/orpo_request.json`. Your own data: `examples/requests/train_request_fields.json`.

## Dataset formats

As in LlamaFactory's `data/README.md`:

- **alpaca**: `instruction`, `input` (optional), `output` (+ `system`, `history`)
- **sharegpt**: `conversations` with `from`/`value`, or OpenAI-style `messages` with `role`/`content`
- **preference** (stage `dpo`): alpaca `instruction` + `chosen` + `rejected`, or sharegpt + `chosen`/`rejected`
- **KTO**: alpaca/sharegpt rows plus a boolean `kto_tag`

A `.zip`/`.tar.gz` that contains its own `dataset_info.json` is used as-is (pick entries with `dataset`).

The bundled `tiny_*` sets were written for this repo (MIT). Stanford Alpaca-derived sets such as
`alpaca_en_demo` are CC BY-NC 4.0 — the worker warns if you use them.

## Storage, checkpoints, resume

With a network volume: models cache under `/runpod-volume/llama-factory/models`, runs under
`/runpod-volume/llama-factory/runs/<run_name>/` (adapter, `trainer_log.jsonl`, `training_loss.png`,
`worker/train_config.yaml`, `worker/train.log`). Re-send the same `run_name` with `"resume": true` to continue
from the latest `checkpoint-*`. Without a volume the default model (~7.5 GiB) is downloaded on every cold
start and outputs are lost when the worker stops unless you set `hf_repo_id`, `upload_url` or `BUCKET_*`.

## Secrets

`HF_TOKEN` only as a Runpod secret (`{{ RUNPOD_SECRET_hf_token }}`) in the endpoint env; needed for gated or
private models/datasets and for pushing adapters. `BUCKET_ENDPOINT_URL` / `BUCKET_ACCESS_KEY_ID` /
`BUCKET_SECRET_ACCESS_KEY` (+ optional `BUCKET_NAME`) for S3 upload. Secrets are redacted from logs.

## GPUs

| Job | VRAM (estimate, not yet measured) | GPU |
|---|---|---|
| 4B QLoRA SFT/DPO (default) | ~10-16 GB | 24 GB (RTX 4090, A5000, L4) |
| 4B 16-bit LoRA, 7-8B QLoRA | ~18-24 GB | 24 GB, 48 GB for long context |
| 7-8B 16-bit LoRA, 4B full | 40 GB+ | 48 GB (L40S, A6000, A40) |

Training must use async `/run`; raise the endpoint `executionTimeout`. `TRAIN_TIMEOUT_SECONDS` is the
worker-side watchdog (default 6 h). Each job ends with `refresh_worker` so the next job starts clean.

## Local development (no GPU)

```bash
pip install -r requirements-dev.txt && pytest -q && ruff check .
python scripts/dry_run.py test_input.json
docker build -t runpod-llama-factory:dev . && scripts/cpu_e2e.sh   # real CLI on a tiny random model, CPU
```

## License and acknowledgements

Worker code: MIT (`LICENSE`). LlamaFactory: Apache-2.0, pinned at v0.9.5
(`7af909522a951e3ad9f022ea6f88b6755257eaa5`); see `NOTICE`. Each base model keeps its own license; the default
Qwen3-4B-Instruct-2507 is Apache-2.0. Please cite LlamaFactory (Zheng et al., ACL 2024, arXiv:2403.13372).
