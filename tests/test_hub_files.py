import json
import os
import re

from src import LLAMAFACTORY_SHA, LLAMAFACTORY_TAG
from src.config import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, PARAMS, resolve_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "RUNPOD_INIT_TIMEOUT"}  # read by secret_utils / the runpod SDK


def load(name):
    with open(os.path.join(ROOT, ".runpod", name), encoding="utf-8") as fh:
        return json.load(fh)


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def test_required_repo_files_exist():
    for f in ("handler.py", "Dockerfile", "README.md", "LICENSE", "NOTICE", "llms.txt", "AGENTS.md", "HANDOFF.md",
              "test_input.json", ".runpod/hub.json", ".runpod/tests.json", "examples/datasets/tiny_sft.json"):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json_required_fields():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config"):
        assert hub.get(k), k
    assert hub["type"] == "serverless" and hub["category"] == "language"
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and cfg["gpuCount"] == 1 and cfg["containerDiskInGb"] == 80
    assert cfg["gpuIds"].startswith("ADA_24,AMPERE_24,ADA_48_PRO,AMPERE_48")
    assert "-NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb" in cfg["gpuIds"]
    assert "12.8" in cfg["allowedCudaVersions"] and "12.6" not in cfg["allowedCudaVersions"]
    assert len(hub["description"]) <= 500


def test_hub_env_keys_are_consumed_by_worker():
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in load("hub.json")["config"]["env"]]
    assert len(keys) == len(set(keys))
    assert set(keys) <= known, set(keys) - known
    for preset in load("hub.json")["config"]["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]


def test_hub_inputs_well_formed():
    for e in load("hub.json")["config"]["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        assert inp.get("name") and inp.get("description")
        if inp["type"] == "number":
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        if "options" in inp:
            assert inp["default"] in [o["value"] for o in inp["options"]], e["key"]


def test_hub_defaults_and_presets_resolve():
    hub = load("hub.json")
    base_env = {e["key"]: str(e["input"]["default"]).lower() if isinstance(e["input"]["default"], bool)
                else str(e["input"]["default"]) for e in hub["config"]["env"]}
    c = resolve_config({"dataset": "tiny_sft"}, env=base_env)
    assert c.model_id == DEFAULT_MODEL and c.model_revision == DEFAULT_MODEL_REVISION
    assert c.train_args["quantization_bit"] == 4 and c.train_args["template"] == "qwen3_nothink"
    for preset in hub["config"]["presets"]:
        env = {**base_env, **{k: str(v) for k, v in preset["defaults"].items()}}
        c = resolve_config({} if "DATASET" in preset["defaults"] else {"dataset": "tiny_sft"}, env=env)
        assert c.train_args["use_unsloth"] is False


def test_tests_json_matches_smoke_spec():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090" and t["config"]["gpuCount"] == 1
    assert len(t["tests"]) == 1
    case = t["tests"][0]
    inp = case["input"]
    assert case["name"] == "llamafactory_qwen3_4b_qlora_sft_smoke"
    assert inp["model_name_or_path"] == DEFAULT_MODEL and inp["quantization_bit"] == 4
    assert inp["max_steps"] == 5 and inp["max_samples"] == 16 and inp["cutoff_len"] == 512
    assert inp["stage"] == "sft" and inp["finetuning_type"] == "lora" and inp["lora_rank"] == 8
    assert inp["dataset_url"].endswith("/prilojject/runpod-llama-factory/main/examples/datasets/tiny_sft.json")
    assert "alpaca_en" not in json.dumps(t)
    assert case["timeout"] == 1800000
    env = {e["key"]: e["value"] for e in t["config"].get("env", [])}
    assert 0 < int(env["TRAIN_TIMEOUT_SECONDS"]) * 1000 < case["timeout"]
    resolve_config(inp, env={})  # must be valid input


def test_dockerfile_pins_match():
    df = read("Dockerfile")
    assert f"ARG LLAMAFACTORY_TAG={LLAMAFACTORY_TAG}" in df and LLAMAFACTORY_TAG == "v0.9.5"
    assert f"ARG LLAMAFACTORY_SHA={LLAMAFACTORY_SHA}" in df
    assert LLAMAFACTORY_SHA == "7af909522a951e3ad9f022ea6f88b6755257eaa5"
    assert 'test "$(git rev-parse HEAD)" = "${LLAMAFACTORY_SHA}"' in df
    assert "webui" not in df.split("CMD")[-1] and "HF_TOKEN=" not in df


def test_requirements_within_upstream_ranges():
    req = read("requirements.txt")
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([0-9.]+)$", req, re.M))
    v = lambda s: tuple(int(x) for x in s.split("."))  # noqa: E731
    assert v("4.55.0") <= v(pins["transformers"]) <= v("5.6.0") and pins["transformers"] not in {"4.52.0", "4.57.0"}
    assert v("0.18.0") <= v(pins["peft"]) <= v("0.18.1")
    assert v("0.18.0") <= v(pins["trl"]) <= v("0.24.0")
    assert v("2.16.0") <= v(pins["datasets"]) <= v("4.0.0")
    assert v("1.3.0") <= v(pins["accelerate"]) <= v("1.11.0")
    assert "gradio" not in [ln.split("=")[0].split("<")[0].split(">")[0].strip() for ln in req.splitlines()
                            if ln and not ln.startswith("#")]


def test_test_input_is_a_dry_run():
    ti = json.loads(read("test_input.json"))["input"]
    assert ti["dry_run"] is True
    resolve_config(ti, env={})
