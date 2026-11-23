# Agent notes — runpod-llama-factory

## What this worker is

A Runpod Serverless **headless LlamaFactory trainer**. Input: an official LlamaFactory YAML (or form
fields) plus an alpaca / sharegpt dataset. Output: a LoRA / QLoRA / freeze / full adapter
(`adapter_model.safetensors` + `adapter_config.json` + tokenizer).

It turns job input into a LlamaFactory YAML, prepares the dataset + `dataset_info.json`, runs
`llamafactory-cli train` in a subprocess, and delivers the adapter. It is **not** a second Unsloth
worker (`use_unsloth` is always false) and never starts Gradio / LLaMA Board.

## Pinned upstream

- Code: `hiyouga/LlamaFactory` tag **v0.9.5**, SHA `7af909522a951e3ad9f022ea6f88b6755257eaa5`
  (Dockerfile ARGs and `src/__init__.py` must match; a test checks the SHA at build time).
  Do **not** clone a floating `main` (it is already ahead of the tag).
- Default weights: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`
  (~7.49 GiB, Apache-2.0, `gated: false`). Cache under `/runpod-volume/llama-factory/models/`.
  Do not bake weights into the image. Do not change the default to a gated Llama/Gemma model.
- Dependency pins must stay inside the tag's `pyproject.toml` ranges. `transformers` must not be
  `4.52.0` or `4.57.0`. This image uses **transformers==4.57.1**.
- The upstream SHA is the Docker pin only. GitHub releases of *this* repo must use
  `targetCommitish=main` (a SHA target breaks Runpod Hub's `git clone -b`).

## Defaults to keep

- Qwen3-4B-Instruct-2507, **bfloat16** (not fp16), QLoRA 4-bit bitsandbytes, template `qwen3_nothink`,
  `lora_rank=8`, stage `sft`, 24 GB GPU (RTX 4090)
- `trust_remote_code` defaults to **false** (Qwen3 is native in transformers). Users may set it.
- `use_unsloth` is always false
- `refresh_worker=true` (each job starts clean)
- `containerDiskInGb=80`
- Bundled smoke data is `examples/datasets/tiny_sft.json` (MIT, written for this repo). Never put
  Stanford Alpaca / `alpaca_en_demo` (CC BY-NC) into defaults or Hub tests.

## Scope limits

- Do **not** install or start Gradio / LLaMA Board. Gradio is intentionally omitted: `gradio<=5.50`
  pins `tomlkit<0.14`, which conflicts with `runpod==2.4.0` (`tomlkit>=0.15.2`). LlamaFactory is
  installed with `--no-deps`; other deps come from `requirements.txt`.
- Do **not** expose stages `pt`, `rm`, or `ppo`. Supported: SFT, DPO (ORPO/SimPO via `pref_loss`), KTO.
  ORPO is `pref_loss: orpo` on `stage: dpo`, not a separate CLI.
- Do not fold this into `runpod-unsloth-finetune` or silently turn on Unsloth kernels.

## License (strict)

- Worker code: MIT
- Upstream LlamaFactory: Apache-2.0 (see `NOTICE`)
- Default weights: Apache-2.0. **Each user-chosen base model keeps its own license** (Llama Community,
  Gemma, etc.) — that is the deployer's responsibility. Never claim “all models are MIT.”

## Tests

- Unit tests (`tests/`) must run without a GPU, without model weights, and without network.
- The Hub smoke test is one case only: `llamafactory_qwen3_4b_qlora_sft_smoke` — default 4B QLoRA,
  5 steps, 16 samples, cutoff 512, RTX 4090, timeout 30 min. After the GitHub repo is public the
  test uses `dataset_url` on `tiny_sft.json`; before that, use `"dataset": "tiny_sft"`.
- After a paid GPU / Serverless test: delete the endpoint, pod and any registry auth created for the
  test. Confirm with GraphQL that leftover smoke endpoints are gone (`endpoints=[]`). Leave protected
  pods `92anyumh6q7jrf` and `oejzh2afh8hcym` (and volumes `13faqbphuo`, `fv2c638zex`) untouched.
  If a template is locked ~90 minutes after billing, leave a background retry — do not wait.

## Secrets

Never print `RUNPOD_API_KEY`, `HF_TOKEN`, `gh` tokens, or `BUCKET_*` values. Never accept tokens in
the job input or YAML.
