# HANDOFF — runpod-llama-factory (Coder 4 → GPU smoke test → Coder)

Status 23 Nov 2026 (Asia/Bangkok): local build done, **GPU smoke test not run yet**. Nothing was
published, no GitHub repo was created, no image was pushed, no Runpod resources were created.

## Done locally (no GPU, no paid calls)

- Worker: `handler.py` + `src/` (config/YAML builder, dataset prep, CLI runner, delivery, secret redaction).
- Upstream pin: LlamaFactory tag `v0.9.5`, SHA `7af909522a951e3ad9f022ea6f88b6755257eaa5` (checked in the
  Docker build with `test "$(git rev-parse HEAD)" = ...`). Default model `Qwen/Qwen3-4B-Instruct-2507` @
  `cdbee75f17c01a7cc42f958dc650907174af0554` (HF metadata check: sha matches, gated=false, apache-2.0,
  safetensors 7.49 GiB).
- Image `runpod-llama-factory:dev` builds on the box (9.08 GB): python 3.11.13, torch 2.7.1+cu128, CUDA 12.8,
  transformers 4.57.1, peft 0.18.1, trl 0.24.0, datasets 4.0.0, accelerate 1.11.0, bitsandbytes 0.48.2,
  llamafactory 0.9.5.
- Unit tests: `pytest -q` (CPU, mocks, no network), `ruff check .` clean.
- CPU end-to-end in the image (`scripts/cpu_e2e.sh`): the real `llamafactory-cli train` on
  `llamafactory/tiny-random-Llama-3` for SFT (`tiny_sft` + upstream `identity`), DPO with `pref_loss: orpo`
  (`tiny_dpo`) and KTO (`tiny_kto`) — all finish and write a non-empty `adapter_model.safetensors` and
  `adapter_config.json`. Dry run with the real default model metadata works.
- Bundled smoke data `examples/datasets/tiny_sft.json`: 16 alpaca rows written for this repo (MIT), no Alpaca.
- Hub files: `.runpod/hub.json` (serverless, GPU, 80 GB disk, 24 GB+ pools minus Blackwell MIG, CUDA ≥ 12.8,
  3 presets, env form), `.runpod/tests.json` (exactly the spec smoke test, RTX 4090, gpuCount 1,
  30 min timeout, `TRAIN_TIMEOUT_SECONDS=1500`), `test_input.json` (dry run).
- Drafts for Coder: `README.md`, `llms.txt`, `AGENTS.md`, `NOTICE`, `LICENSE` (MIT).

## Deviations from the spec (for Coder's list)

1. **gradio is not installed.** `gradio<=5.50` (LlamaFactory range) pins `tomlkit<0.14`, `runpod==2.4.0` needs
   `tomlkit>=0.15.2` → pip ResolutionImpossible. LlamaFactory is installed with `--no-deps` and all its other
   pyproject deps are installed from `requirements.txt` (pins inside the tag's ranges). The train path never
   imports gradio (checked: only `llamafactory/webui/*`), and the CPU e2e proves it. Side effect:
   `llamafactory-cli webui` cannot run in this image — which the spec forbids anyway.
2. transformers pinned to **4.57.1** (the version LlamaFactory's own CI tests on v0.9.5 for backward compat;
   it is inside the range and != 4.57.0), not a 5.x, to stay contemporary with trl 0.24.0 / peft 0.18.1.
3. `trust_remote_code` defaults to **false** (Qwen3 is native in transformers); official YAMLs say true.
   Users can set it.
4. Stage `pt` / `rm` / `ppo` are not exposed (spec scope: SFT + DPO/ORPO/SimPO + KTO).
5. `.runpod/icon.png` is referenced in hub.json but not created — Coder adds the icon/banner.
6. `openapi.yaml`, `CITATION.cff`, `context7.json`, Pages — left to Coder.

## Needs the GPU smoke test

- QLoRA 4-bit (bitsandbytes 0.48.2 CUDA kernels) on the real 4B model — not testable on CPU.
- bf16 training path, peak VRAM, real step time, cold-start time (image pull + 7.49 GiB download).
- `progress_update` from a real Runpod worker; `nvidia-smi` VRAM sampling.
- The test fetches `dataset_url` from `raw.githubusercontent.com/prilojject/runpod-llama-factory/main/...`,
  so **the GitHub repo must exist first** (or run the same job with `"dataset": "tiny_sft"` instead, which
  uses the copy baked into the image — recommended for the pre-publish test).

## GPU test plan (≤ $5 hard budget, target < $1)

Prerequisite: an image in a registry Runpod can pull (e.g. `ghcr.io/prilojject/runpod-llama-factory:dev`,
built from this repo's `main`); pushing/making it public is a publish step — decide who does it.

1. Note the account balance before the test.
2. Create a Serverless template + endpoint:
   - image as above, `containerDiskInGb: 80`, GPU **RTX 4090** (`ADA_24`), `gpuCount 1`,
   - **workersMin 0, workersMax 1**, idleTimeout 5 s, flashboot off,
   - `executionTimeout` **1 800 000 ms** (30 min), job `ttl` ≤ 1 h,
   - env: `TRAIN_TIMEOUT_SECONDS=1500`, `RUNPOD_INIT_TIMEOUT=900`. No HF_TOKEN needed (model not gated).
   - no network volume (cold download is part of the test).
3. Dry run first (`examples/requests/dry_run_request.json`) via async `/run`; expect `status: ok`,
   `base_model_check.sha = cdbee75…`.
4. Smoke job: the input from `.runpod/tests.json` (with `"dataset": "tiny_sft"` instead of `dataset_url`
   if the repo is not public yet) via async `/run`. Poll `/status` every 30-60 s.
   Success = no `error`, `training.global_step == 5`, `artifacts.adapter_files` with
   `adapter_model.safetensors` > 0 bytes and `adapter_config.json` > 0 bytes.
   Record: `training.total_seconds`, `peak_vram_gb`, `delayTime`, `executionTime`.
5. Kill switch: if the job is still running after 30 min, or balance drop reaches ~$3, cancel the job and
   go to cleanup. Expected cost: 4090 serverless ≈ $1.34/h → a 30 min worst case ≈ $0.70; realistic
   (5-15 min incl. image pull + 7.5 GiB download) ≈ $0.10-0.35.
6. Cleanup right after (even on failure): set workersMax 0, **delete the endpoint**, delete the template,
   delete any registry auth created for the test, confirm no pods remain; note balance after.
7. Report: endpoint id (deleted), image digest, timings, cost, deviations → Coder fills README prices.
