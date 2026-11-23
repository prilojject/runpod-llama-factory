import io
import json
import os
import tarfile
import zipfile

import pytest

from src import dataset as ds
from src.config import resolve_config
from src.dataset import DatasetError, detect_entry, prepare_dataset, safe_extract, validate_records

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "examples", "datasets")


def cfg(inp):
    return resolve_config(inp, env={})


def load(name):
    with open(os.path.join(TINY, name), encoding="utf-8") as f:
        return json.load(f)


def test_tiny_sft_is_small_own_alpaca():
    rows = load("tiny_sft.json")
    assert 8 <= len(rows) <= 16
    for r in rows:
        assert set(r) == {"instruction", "input", "output"} and r["instruction"] and r["output"]
    text = json.dumps(rows).lower()
    # not Stanford Alpaca: its first demo rows start like this
    assert "give three tips for staying healthy" not in text


def test_bundled_sets_detect_and_validate():
    for name, stage, ranking, kto in (("tiny_sft.json", "sft", False, False), ("tiny_dpo.json", "dpo", True, False),
                                      ("tiny_kto.json", "kto", False, True)):
        rows = load(name)
        e = detect_entry(rows, "auto", name)
        assert bool(e.get("ranking")) is ranking and ("kto_tag" in e["columns"]) is kto
        assert validate_records(rows, e) == (len(rows), 0)
        ds.check_stage(e, stage, name)


def test_sharegpt_and_openai_messages():
    sg = [{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}]
    e = detect_entry(sg, "auto", "a.json")
    assert e["formatting"] == "sharegpt" and "tags" not in e and e["columns"]["messages"] == "conversations"
    oa = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
    e = detect_entry(oa, "sharegpt", "b.json")
    assert e["tags"]["role_tag"] == "role" and e["tags"]["user_tag"] == "user"
    with pytest.raises(DatasetError):
        detect_entry(oa, "alpaca", "b.json")
    with pytest.raises(DatasetError, match="pre-training"):
        detect_entry([{"text": "doc"}], "auto", "c.json")


def test_stage_mismatch(tmp_path):
    with pytest.raises(DatasetError, match="stage=dpo"):
        prepare_dataset(cfg({"dataset": "tiny_sft", "stage": "dpo"}), str(tmp_path / "w1"))
    with pytest.raises(DatasetError, match="needs stage=dpo"):
        prepare_dataset(cfg({"dataset": "tiny_dpo"}), str(tmp_path / "w2"))


def test_prepare_inline_and_bundled(tmp_path):
    c = cfg({"dataset": "tiny_sft", "dataset_inline": [{"instruction": "a", "input": "", "output": "b"},
                                                       {"instruction": "", "output": "x"}]})
    d, rep = prepare_dataset(c, str(tmp_path))
    info = json.load(open(os.path.join(d, "dataset_info.json")))
    assert rep.names == ["inline", "tiny_sft"] and set(info) == {"inline", "tiny_sft"}
    assert info["tiny_sft"]["file_name"] == "tiny_sft.json"
    assert os.path.isfile(os.path.join(d, "tiny_sft.json"))
    assert rep.invalid_rows == {"inline": 1} and rep.rows["tiny_sft"] == 16


def test_unknown_name_and_hf_id(tmp_path):
    with pytest.raises(DatasetError, match="unknown dataset"):
        prepare_dataset(cfg({"dataset": "nope"}), str(tmp_path))
    d, rep = prepare_dataset(cfg({"dataset": "someone/public-set"}), str(tmp_path))
    assert rep.entries["someone/public-set"] == {"hf_hub_url": "someone/public-set"}


def test_upstream_entry_copied(tmp_path, monkeypatch):
    up = tmp_path / "up" / "data"
    up.mkdir(parents=True)
    (up / "identity.json").write_text(json.dumps([{"instruction": "hi", "input": "", "output": "I am {{name}}"}]))
    (up / "dataset_info.json").write_text(json.dumps({"identity": {"file_name": "identity.json"}}))
    monkeypatch.setattr(ds, "upstream_data_dir", lambda: str(up))
    d, rep = prepare_dataset(cfg({"dataset": "identity"}), str(tmp_path / "w"))
    assert os.path.isfile(os.path.join(d, "identity.json")) and rep.names == ["identity"]


def _fake_download(path):
    def f(url, dest):
        return path
    return f


def test_dataset_url_json(tmp_path, monkeypatch):
    src = tmp_path / "dl" / "my.jsonl"
    src.parent.mkdir()
    src.write_text("\n".join(json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) for i in range(5)))
    monkeypatch.setattr(ds, "download", _fake_download(str(src)))
    d, rep = prepare_dataset(cfg({"dataset_url": "https://x/my.jsonl"}), str(tmp_path / "w"))
    assert rep.names == ["user_data"] and rep.rows["user_data"] == 5
    assert json.load(open(os.path.join(d, "dataset_info.json")))["user_data"]["file_name"] == "user_data.jsonl"


def test_dataset_url_zip_with_own_dataset_info(tmp_path, monkeypatch):
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("pack/dataset_info.json", json.dumps({"mine": {"file_name": "rows.json"}}))
        zf.writestr("pack/rows.json", json.dumps([{"instruction": "a", "output": "b"}]))
    monkeypatch.setattr(ds, "download", _fake_download(str(z)))
    d, rep = prepare_dataset(cfg({"dataset_url": "https://x/d.zip", "dataset": "mine"}), str(tmp_path / "w"))
    info = json.load(open(os.path.join(d, "dataset_info.json")))
    assert info["mine"]["file_name"] == os.path.join("user", "pack", "rows.json")
    assert os.path.isfile(os.path.join(d, info["mine"]["file_name"]))


def test_archive_entry_cannot_point_remote(tmp_path, monkeypatch):
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dataset_info.json", json.dumps({"x": {"hf_hub_url": "a/b"}}))
    monkeypatch.setattr(ds, "download", _fake_download(str(z)))
    with pytest.raises(DatasetError, match="only local files"):
        prepare_dataset(cfg({"dataset_url": "https://x/d.zip"}), str(tmp_path / "w"))


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../evil.json", "[]")
    with pytest.raises(DatasetError, match="escapes"):
        safe_extract(str(z), str(tmp_path / "out"))
    assert not (tmp_path / "evil.json").exists()


def test_tar_slip_and_symlink_rejected(tmp_path):
    t = tmp_path / "evil.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        data = b"[]"
        info = tarfile.TarInfo("../evil.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(DatasetError, match="escapes"):
        safe_extract(str(t), str(tmp_path / "out"))
    t2 = tmp_path / "link.tar"
    with tarfile.open(t2, "w") as tf:
        info = tarfile.TarInfo("ok/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(DatasetError, match="links"):
        safe_extract(str(t2), str(tmp_path / "out2"))


def test_not_an_archive(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"garbage")
    with pytest.raises(DatasetError):
        safe_extract(str(p), str(tmp_path / "o"))


def test_download_size_cap(tmp_path, monkeypatch):
    class Resp:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, n):
            for _ in range(4):
                yield b"x" * 1024

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(ds, "MAX_DOWNLOAD_BYTES", 2048)
    with pytest.raises(DatasetError, match="larger"):
        ds.download("https://x/y.json", str(tmp_path))
