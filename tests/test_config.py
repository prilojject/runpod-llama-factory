import pytest
import yaml

from src.config import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, ConfigError, build_train_yaml, dump_yaml, resolve_config

BASE = {"dataset": "tiny_sft"}


def r(inp, env=None, **kw):
    return resolve_config(inp, env=env or {}, **kw)


def test_defaults_match_spec():
    c = r(BASE)
    a = c.train_args
    assert c.model_id == DEFAULT_MODEL == "Qwen/Qwen3-4B-Instruct-2507"
    assert c.model_revision == DEFAULT_MODEL_REVISION == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert a["stage"] == "sft" and a["finetuning_type"] == "lora"
    assert a["quantization_bit"] == 4 and a["quantization_method"] == "bnb"
    assert a["template"] == "qwen3_nothink"
    assert a["lora_rank"] == 8 and a["lora_target"] == "all"
    assert a["bf16"] is True and "fp16" not in a
    assert a["use_unsloth"] is False
    assert c.dtype == "bfloat16"
    assert c.dataset_names == ["tiny_sft"]


def test_use_unsloth_never_default_and_refused():
    y = build_train_yaml(r(BASE), model_path="/m", dataset_names=["tiny_sft"], dataset_dir="/d",
                         output_dir="/o", cache_dir="/c")
    assert y["use_unsloth"] is False
    with pytest.raises(ConfigError, match="use_unsloth"):
        r({**BASE, "config_yaml": "use_unsloth: true"})
    with pytest.raises(ConfigError, match="use_unsloth"):
        r({**BASE, "extra_args": {"use_unsloth": True}})


def test_smoke_input_from_spec():
    inp = {"model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
           "dataset_url": "https://raw.githubusercontent.com/prilojject/runpod-llama-factory/main/examples/datasets/tiny_sft.json",
           "dataset_format": "alpaca", "stage": "sft", "finetuning_type": "lora", "quantization_bit": 4,
           "max_steps": 5, "max_samples": 16, "cutoff_len": 512, "lora_rank": 8, "run_name": "hub-test-llama-factory"}
    c = r(inp)
    assert c.train_args["max_steps"] == 5 and c.train_args["max_samples"] == 16
    assert c.train_args["cutoff_len"] == 512 and c.run_name == "hub-test-llama-factory"
    assert c.dataset_url.endswith("tiny_sft.json") and c.dataset_names == []
    assert c.model_revision == DEFAULT_MODEL_REVISION


def test_priority_input_over_yaml_over_env_over_default():
    env = {"LORA_RANK": "32", "MAX_STEPS": "100", "LEARNING_RATE": "3e-4", "STAGE": "sft"}
    c = r({**BASE, "config_yaml": "lora_rank: 16\nlearning_rate: 2.0e-4\n", "lora_rank": 64}, env)
    assert c.train_args["lora_rank"] == 64 and c.sources["lora_rank"] == "input"
    assert c.train_args["learning_rate"] == 2e-4 and c.sources["learning_rate"] == "config_yaml"
    assert c.train_args["max_steps"] == 100 and c.sources["max_steps"] == "env:MAX_STEPS"
    assert c.sources["cutoff_len"] == "default"


def test_env_max_steps_zero_means_epochs_and_unresolved_secret_ignored():
    c = r(BASE, {"MAX_STEPS": "0", "HF_OUTPUT_REPO": "{{ RUNPOD_SECRET_x }}"})
    assert "max_steps" not in c.train_args and c.hf_repo_id is None


def test_quantization_null_is_16bit_lora():
    for v in (None, "none", 0, "16"):
        c = r({**BASE, "quantization_bit": v})
        assert "quantization_bit" not in c.train_args and "quantization_method" not in c.train_args
        assert c.public_dict()["quantization_bit"] is None
    with pytest.raises(ConfigError):
        r({**BASE, "quantization_bit": 3})


def test_quantization_requires_lora():
    with pytest.raises(ConfigError, match="only works with finetuning_type=lora"):
        r({**BASE, "finetuning_type": "full"})
    c = r({**BASE, "finetuning_type": "full", "quantization_bit": None})
    assert any("48 GB" in w for w in c.warnings)


def test_dpo_orpo_is_pref_loss_not_stage():
    c = r({"dataset": "tiny_dpo", "stage": "dpo", "pref_loss": "orpo"})
    assert c.train_args["stage"] == "dpo" and c.train_args["pref_loss"] == "orpo"
    assert c.train_args["learning_rate"] == 5e-6 and c.train_args["pref_beta"] == 0.1
    with pytest.raises(ConfigError, match="pref_loss=orpo"):
        r({**BASE, "stage": "orpo"})
    with pytest.raises(ConfigError, match="only applies to stage=dpo"):
        r({**BASE, "pref_loss": "orpo"})


def test_kto_stage_defaults():
    c = r({"dataset": "tiny_kto", "stage": "kto"})
    assert c.train_args["pref_beta"] == 0.1 and "pref_loss" not in c.train_args


def test_tokens_refused_everywhere():
    with pytest.raises(ConfigError, match="tokens"):
        r({**BASE, "hf_token": "abc"})
    with pytest.raises(ConfigError, match="token"):
        r({**BASE, "config_yaml": "hf_hub_token: hf_" + "a" * 30})
    with pytest.raises(ConfigError, match="access token"):
        r({**BASE, "config_yaml": "model_name_or_path: hf_" + "b" * 34})
    with pytest.raises(ConfigError, match="access token"):
        r({**BASE, "dataset_inline": [{"instruction": "x", "output": "hf_" + "c" * 30}]})


def test_special_tokens_keys_allowed():
    c = r({**BASE, "config_yaml": "new_special_tokens: '<a>,<b>'\n"})
    assert c.train_args["new_special_tokens"] == "<a>,<b>"


def test_worker_owned_paths_overridden():
    c = r({**BASE, "config_yaml": "output_dir: /etc\ndataset_dir: /\ncache_dir: /root\nreport_to: wandb\n"})
    for k in ("output_dir", "dataset_dir", "cache_dir", "report_to"):
        assert k not in c.train_args
    assert sum("set by the worker" in w for w in c.warnings) == 4
    y = build_train_yaml(c, model_path="/m", dataset_names=["tiny_sft"], dataset_dir="/w/data",
                         output_dir="/runpod-volume/llama-factory/runs/x", cache_dir="/c")
    assert y["output_dir"] == "/runpod-volume/llama-factory/runs/x" and y["report_to"] == "none"
    assert y["dataset_dir"] == "/w/data" and y["overwrite_output_dir"] is True


def test_forbidden_keys():
    for k in ("push_to_hub: true", "deepspeed: /x.json", "export_dir: /x", "use_ray: true"):
        with pytest.raises(ConfigError, match="not supported"):
            r({**BASE, "config_yaml": k})


def test_full_official_yaml_roundtrip():
    official = """
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
quantization_bit: 4
quantization_method: bnb
trust_remote_code: true
### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
### dataset
dataset: identity,tiny_sft
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
### output
output_dir: saves/qwen3-4b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none
### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
"""
    c = r({"config_yaml": official})
    assert c.dataset_names == ["identity", "tiny_sft"]
    assert c.train_args["trust_remote_code"] is True and c.train_args["logging_steps"] == 10
    y = build_train_yaml(c, model_path="/cache/snap", dataset_names=c.dataset_names, dataset_dir="/d",
                         output_dir="/o", cache_dir="/c")
    assert yaml.safe_load(dump_yaml(y))["dataset"] == "identity,tiny_sft"
    assert "do_train" in y and y["do_train"] is True


def test_config_url_fetcher_and_input_override():
    c = r({"config_url": "https://example.com/c.yaml", "max_steps": 7},
          yaml_fetcher=lambda u: "dataset: tiny_sft\nmax_steps: 100\nlora_rank: 4\n")
    assert c.train_args["max_steps"] == 7 and c.train_args["lora_rank"] == 4


def test_non_qwen3_model_needs_template_and_warns_unpinned():
    with pytest.raises(ConfigError, match="template is required"):
        r({**BASE, "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3"})
    c = r({**BASE, "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3", "template": "mistral"})
    assert c.model_revision is None and any("not pinned" in w for w in c.warnings)


def test_env_revision_not_applied_to_job_model():
    c = r({**BASE, "model_name_or_path": "Qwen/Qwen3-8B"}, {"MODEL_REVISION": "abc123"})
    assert c.model_revision is None
    c = r(BASE, {"MODEL_NAME_OR_PATH": "Qwen/Qwen3-8B", "MODEL_REVISION": "abc123"})
    assert c.model_id == "Qwen/Qwen3-8B" and c.model_revision == "abc123"


def test_job_dataset_beats_env_dataset():
    c = r({"dataset_inline": [{"instruction": "a", "output": "b"}]},
          {"DATASET_URL": "https://example.com/x.json", "DATASET": "tiny_sft"})
    assert c.dataset_url is None and c.dataset_names == [] and c.dataset_inline


def test_validation_errors():
    bad = [
        {},  # no data
        {**BASE, "stage": "ppo"},
        {**BASE, "lora_rank": 0},
        {**BASE, "max_steps": "abc"},
        {**BASE, "run_name": "../etc"},
        {**BASE, "model_name_or_path": "/etc/passwd"},
        {**BASE, "model_name_or_path": "not a repo"},
        {**BASE, "dataset_url": "file:///etc/passwd"},
        {**BASE, "hf_repo_id": "bad"},
        {**BASE, "upload_url": "http://insecure"},
        {**BASE, "resume": True},
        {**BASE, "unknown_field": 1},
        {**BASE, "bf16": True, "extra_args": {"fp16": True}},
        {"dataset_inline": [], "dataset": "tiny_sft"},
        {"dataset_inline": [{"instruction": "a", "output": "b"}], "dataset_url": "https://x/y.json"},
        {**BASE, "config_yaml": "- a\n- b\n"},
        {**BASE, "config_yaml": "a: [unclosed"},
    ]
    for inp in bad:
        with pytest.raises(ConfigError):
            r(inp)


def test_public_dict_has_no_secrets_and_strips_query():
    c = r({"dataset_url": "https://bucket.example.com/d.json?X-Amz-Signature=secret"})
    pd = c.public_dict()
    assert pd["dataset_url"] == "https://bucket.example.com/d.json"
    assert "secret" not in str(pd)


def test_alpaca_dataset_name_warns_license():
    c = r({"dataset": "alpaca_en_demo"})
    assert any("CC BY-NC" in w for w in c.warnings)
