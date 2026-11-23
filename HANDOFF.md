# HANDOFF — runpod-llama-factory (Coder 4 → GPU smoke test → Coder)

Status 23 Nov 2026 (Asia/Bangkok): local build done, **GPU smoke test PASSED on an RTX 4090 pod**
(14:29-14:41 UTC+7). Nothing was published: no GitHub repo, no image pushed to any registry. The test pod was
terminated (deleted); no pods, endpoints or templates created by this test remain.

## GPU smoke test result (23 Nov 2026)

- Setup: one Runpod **Secure Cloud RTX 4090** pod (community had no supply), $0.89/h, image
  `runpod/pytorch:1.6.0-cu1281-torch271-ubuntu2204` (torch 2.7.1+cu128, **Python 3.12.16** — the Docker image
  uses 3.11.13; both are in LlamaFactory's 3.11-3.13 range), driver 595.112.07. Repo copied with `git archive`,
  deps installed exactly as the Dockerfile (constraints from the base torch stack, `requirements.txt`,
  LlamaFactory v0.9.5 `--no-deps`, SHA checked). Resolved: transformers 4.57.1, peft 0.18.1, trl 0.24.0,
  datasets 4.0.0, accelerate 1.11.0, bitsandbytes 0.48.2, llamafactory 0.9.5.
- Job: `python handler.py --test_input` with the smoke input (default model @ `cdbee75…`, `dataset: tiny_sft`,
  QLoRA 4-bit bnb, sft, 5 steps, `max_samples` 16, `cutoff_len` 512, `lora_rank` 8), no network volume.
- Result: `status: ok`, `global_step 5 / 5`, `adapter_model.safetensors` 66 126 768 bytes (63 MB),
  `adapter_config.json` 1 142 bytes, archive 61 MB, `train_loss` 3.587.
- Timings: model download 7.49 GiB in **13.1 s** (hf_transfer), training subprocess **43.5 s** (load + 4-bit
  quantize + 5 steps), train loop **20.4 s** (~**4.1 s/step**, batch 1 x grad-accum 8), whole job **62 s**.
- Peak VRAM **4.59 GB** (nvidia-smi sampled every 5 s → may under-read short peaks; sampling is now 2 s).
- Pod lifetime 14:29:40-14:41:32 UTC+7 = 11 min 52 s (most of it pip install + a polling mistake on my side)
  → **cost ≈ $0.18** (0.198 h x $0.89/h). Account balance 55.0926 → 54.8131 (−$0.28), but other agents'
  pods/endpoints were billing at the same time (spend rate 0.90-2.24 $/h), so the balance delta is not ours alone.
- No functional bugs found. Only change after the test: VRAM sampler interval 5 s → 2 s.
  Upstream log notes (harmless): "recommend `upcast_layernorm` in quantized training", `torch_dtype` deprecation.
- Still untested on real Serverless: the Docker image itself on a Runpod worker (cold start / image pull time),
  `progress_update`, `dataset_url` from raw.githubusercontent (repo does not exist yet). Those are covered by
  the Hub test after Coder publishes.

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

## Was open before the GPU smoke test (now covered except Serverless-specific items above)

- QLoRA 4-bit (bitsandbytes 0.48.2 CUDA kernels) on the real 4B model — not testable on CPU.
- bf16 training path, peak VRAM, real step time, cold-start time (image pull + 7.49 GiB download).
- `progress_update` from a real Runpod worker; `nvidia-smi` VRAM sampling.
- The test fetches `dataset_url` from `raw.githubusercontent.com/prilojject/runpod-llama-factory/main/...`,
  so **the GitHub repo must exist first** (or run the same job with `"dataset": "tiny_sft"` instead, which
  uses the copy baked into the image — recommended for the pre-publish test).

## Serverless re-test plan for Coder after publishing (≤ $5 hard budget, target < $1)

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
