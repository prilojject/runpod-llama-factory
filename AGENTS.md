# AGENTS.md

Instructions for coding agents working in `prilojject/runpod-llama-factory`.

## What this repo is

A Runpod Serverless worker that wraps the official `llamafactory-cli train` of hiyouga/LlamaFactory
(pinned tag v0.9.5, commit `7af909522a951e3ad9f022ea6f88b6755257eaa5`). It turns job input into a
LlamaFactory YAML, prepares the dataset + `dataset_info.json`, runs the CLI in a subprocess and delivers
the adapter. It is not a second Unsloth worker (`use_unsloth` is always false) and never starts Gradio /
LLaMA Board.

## Layout

- `handler.py` — entry point: config → dataset → (dry run | GPU check → model download → train → deliver)
- `src/config.py` — input > YAML > env > defaults, validation, forbidden/worker-owned keys, YAML builder
- `src/dataset.py` — inline / URL / archive / named datasets, format detection, zip-slip-safe extraction
- `src/train.py` — model snapshot + size check, CLI subprocess with watchdog, metrics from `trainer_log.jsonl`
- `src/outputs.py` — archive, HF push, S3 bucket, presigned PUT
- `src/secret_utils.py` — `HF_TOKEN` from env only, log redaction
- `.runpod/hub.json`, `.runpod/tests.json` — Hub form and the single smoke test
- `examples/datasets/` — bundled MIT datasets (`tiny_sft`, `tiny_dpo`, `tiny_kto`)
- `tests/` — CPU unit tests (no torch, no weights, no network)

## Local checks (no GPU, nothing paid)

```bash
pip install -r requirements-dev.txt
pytest -q && ruff check .
python scripts/dry_run.py test_input.json
docker build -t runpod-llama-factory:dev . && scripts/cpu_e2e.sh
```

## Rules

- Keep the upstream pin on the release tag + SHA (Dockerfile ARGs and `src/__init__.py` must match; a test
  checks). Do not float on `main`. Dependency pins must stay inside the tag's `pyproject.toml` ranges.
- Never put Stanford Alpaca / `alpaca_en_demo` into defaults or Hub tests (CC BY-NC).
- Never accept tokens in job input or YAML; never log secrets.
- Do not start paid GPU pods/endpoints unless a human approved a test plan; delete them right after.
- Hub releases of this repo use `targetCommitish=main`, not a SHA.
