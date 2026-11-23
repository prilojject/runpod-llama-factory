import tarfile
from types import SimpleNamespace

from src import outputs


def _out(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    for f in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "README.md", "training_args.bin"):
        (d / f).write_bytes(b"x" * 10)
    (d / "checkpoint-5").mkdir()
    (d / "checkpoint-5" / "optimizer.pt").write_bytes(b"x")
    (d / "worker").mkdir()
    (d / "worker" / "train.log").write_text("log")
    return d


def cfg(**kw):
    base = dict(archive=True, upload_url=None, upload_to_bucket=True, hf_repo_id=None, hf_private=True, stage="sft")
    base.update(kw)
    return SimpleNamespace(**base)


def test_archive_excludes_checkpoints_and_logs(tmp_path):
    d = _out(tmp_path)
    res, warn = outputs.deliver(cfg(), "r1", str(d), None, False, {})
    names = tarfile.open(res["archive"]).getnames()
    assert "adapter_model.safetensors" in names and "adapter_config.json" in names
    assert not any(n.startswith(("checkpoint", "worker")) or n == "training_args.bin" for n in names)
    assert any("not delivered anywhere persistent" in w for w in warn)


def test_volume_counts_as_delivered(tmp_path):
    res, warn = outputs.deliver(cfg(archive=False), "r1", str(_out(tmp_path)), None, True, {})
    assert warn == [] and "archive" not in res


def test_hf_without_token_warns(tmp_path):
    res, warn = outputs.deliver(cfg(hf_repo_id="me/x"), "r1", str(_out(tmp_path)), None, True, {})
    assert any("HF_TOKEN is not configured" in w for w in warn) and "hf_repo_url" not in res


def test_hf_push_and_presigned(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(outputs, "push_to_hf", lambda repo, d, tok, priv, msg: calls.setdefault("hf", repo)
                        and f"https://huggingface.co/{repo}")
    monkeypatch.setattr(outputs, "put_presigned", lambda url, path: 200)
    url = "https://bucket.example/obj?X-Amz-Signature=abc"
    res, warn = outputs.deliver(cfg(hf_repo_id="me/x", upload_url=url), "r1", str(_out(tmp_path)),
                                "hf_" + "t" * 30, False, {})
    assert res["hf_repo_url"] == "https://huggingface.co/me/x" and res["upload_url_status"] == 200
    assert warn == []


def test_presigned_error_hides_url(tmp_path, monkeypatch):
    url = "https://bucket.example/obj?X-Amz-Signature=abc"

    def boom(u, p):
        raise RuntimeError(f"failed {u}")
    monkeypatch.setattr(outputs, "put_presigned", boom)
    res, _ = outputs.deliver(cfg(upload_url=url), "r1", str(_out(tmp_path)), None, False, {})
    assert "Signature" not in res["upload_url_error"]
