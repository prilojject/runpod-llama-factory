# PROGRESS — runpod-llama-factory

Times are Asia/Bangkok (UTC+7). Workdir: `/workspace/runpod-hub-llama-factory` (copy of `/workspace/runpod-hub-promo/runpod-llama-factory` with git history; started at `90eceeb`).

## 14:46 — start (publish handoff from Coder 4)

- Read HANDOFF.md, llama-factory-spec.md, llama-factory-to-coder.txt, and JoyCaption / Qwen-VL publish pattern.
- Local image `runpod-llama-factory:dev` existed (9.08 GB) but still had VRAM sampler interval 5 s (pre-`90eceeb`). Rebuilding last layers so the published image matches current `src/train.py` (2 s). Log: `/workspace/build_llama_factory.log`.
- Will add icon, openapi.yaml, CITATION.cff, context7.json, Pages, polish README/llms.txt/AGENTS.md, create public GitHub repo, push GHCR `:v1.0.0` + `:latest`, thin `.runpod/Dockerfile`, release `v1.0.0` with `targetCommitish=main`, Pages, topics, Awesome-Runpod#28 line, optional Serverless re-test.
- Will NOT: toggle GHCR package Public, submit to Runpod Hub (Add Repo), touch protected pods `92anyumh6q7jrf` / `oejzh2afh8hcym` or volumes `13faqbphuo` / `fv2c638zex`, print secrets.

## Next

- Finish missing files + unit tests
- Push repo + image
- Thin Dockerfile + release + Pages + Awesome PR
- Serverless smoke on ADA_24 / RTX 4090, then delete leftovers

## 14:54 Asia/Bangkok — docs + tests; image rebuild done; GHCR push started

- Added `.runpod/icon.png` (512×512), `openapi.yaml`, `CITATION.cff`, `context7.json`, `index.html`, `.nojekyll`.
- Polished README.md / llms.txt / AGENTS.md. README now has an Implementation notes section covering Coder 4 deviations (no Gradio / tomlkit conflict, transformers 4.57.1, trust_remote_code default false, no pt/rm/ppo).
- Unit tests: 63 passed, `ruff check .` clean. Required-files test now includes the publish extras.
- Image rebuilt from current `src/` (VRAM sampler 2 s). Local id `sha256:3c1e7a7728a4590f6ef528ce56890f2885e1567a2e3844361853f27493b2f80c` (9.08 GB).
- GHCR push of `:v1.0.0` started (log `/workspace/push_llama_factory.log`). Will also push `:latest` after, record digest, then add thin `.runpod/Dockerfile`.
- Serverless re-test: deferred if usage is tight (Coder 4 already passed a 4090 pod smoke). Prefer finishing repo / release / Pages / Awesome#28. No endpoints created yet.

## 16:29 Asia/Bangkok — :latest, thin Dockerfile, release, Awesome#28

- Pushed `:latest` to the same digest as `:v1.0.0`: `sha256:d0988f53a87919554856f9a3cca1dda4f2d2942406bed75e9de955788dfdd0f4` (remote tags now `latest` + `v1.0.0`).
- Adding thin `.runpod/Dockerfile` FROM that digest (root Dockerfile kept). Then release `v1.0.0` with `targetCommitish=main`, and the Awesome-Runpod#28 line (still missing after the earlier failed commit).
- No Serverless re-test (Coder 4's 4090 pod smoke already passed). No endpoints created. GHCR stays private; Hub Add Repo left to the parent.
