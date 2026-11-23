#!/usr/bin/env bash
# CPU end-to-end check inside the built image: real `llamafactory-cli train` on a ~10 MB random
# model (llamafactory/tiny-random-Llama-3, public, no token). No GPU, nothing paid.
#   docker build -t runpod-llama-factory:dev . && scripts/cpu_e2e.sh
set -euo pipefail
IMAGE="${1:-runpod-llama-factory:dev}"
docker run --rm -i -e ALLOW_CPU_TRAINING=1 "$IMAGE" python - <<'PY'
import json, handler
base = {"model_name_or_path": "llamafactory/tiny-random-Llama-3", "template": "llama3", "quantization_bit": None,
        "bf16": False, "max_steps": 2, "cutoff_len": 128, "lora_rank": 4, "check_model": False}
cases = [("sft", "tiny_sft,identity", {}), ("dpo", "tiny_dpo", {"pref_loss": "orpo"}), ("kto", "tiny_kto", {})]
ok = True
for stage, ds, extra in cases:
    r = handler.handler({"id": f"local-{stage}", "input": {**base, "stage": stage, "dataset": ds,
                                                          "run_name": f"cpu-{stage}", **extra}})
    files = (r.get("artifacts") or {}).get("adapter_files") or {}
    good = not r.get("error") and files.get("adapter_model.safetensors", 0) > 0
    ok &= good
    print(stage, "OK" if good else "FAIL", json.dumps(r.get("training")), (r.get("error") or "")[-800:])
raise SystemExit(0 if ok else 1)
PY
