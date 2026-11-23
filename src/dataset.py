"""Prepare training data for LlamaFactory: files + dataset_info.json in a per-run dataset_dir.

Sources: dataset_inline (list), dataset_url (.json/.jsonl/.csv/.parquet or .zip/.tar.gz archive,
zip-slip safe, size-capped), dataset names (bundled tiny_* sets, upstream LlamaFactory data entries,
or public Hugging Face dataset ids).
Formats (LlamaFactory data/README.md): alpaca and sharegpt, incl. preference (ranking) and KTO.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any

from .paths import bundled_datasets_dir, upstream_data_dir

MAX_DOWNLOAD_BYTES = 2 * 2**30
MAX_EXTRACT_BYTES = 8 * 2**30
DATA_EXTS = (".json", ".jsonl", ".csv", ".parquet", ".arrow")
BUNDLED = {
    "tiny_sft": "tiny_sft.json",
    "tiny_dpo": "tiny_dpo.json",
    "tiny_kto": "tiny_kto.json",
}
OPENAI_TAGS = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
               "assistant_tag": "assistant", "system_tag": "system", "observation_tag": "tool",
               "function_tag": "function_call"}


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    names: list[str] = field(default_factory=list)
    entries: dict[str, dict] = field(default_factory=dict)
    rows: dict[str, int | None] = field(default_factory=dict)
    invalid_rows: dict[str, int] = field(default_factory=dict)
    preview: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"names": self.names, "rows": self.rows, "invalid_rows": self.invalid_rows,
                "formats": {k: v.get("formatting", "alpaca") + (" (ranking)" if v.get("ranking") else "")
                            for k, v in self.entries.items()},
                "preview": self.preview}


# --------------------------------------------------------------------------- reading
def read_records(path: str, limit: int | None = None) -> list[dict]:
    """Read .json (list or {"data": [...]}) or .jsonl records."""
    if path.endswith(".jsonl"):
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{os.path.basename(path)} line {i}: invalid JSON ({exc.msg})") from None
                if limit and len(out) >= limit:
                    break
        return out
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{os.path.basename(path)}: invalid JSON ({exc.msg} at line {exc.lineno})") from None
    if isinstance(data, dict):
        for key in ("data", "train", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise DatasetError(f"{os.path.basename(path)}: expected a JSON list of records")
    return data[:limit] if limit else data


def detect_entry(records: list[dict], fmt: str, file_name: str) -> dict[str, Any]:
    """Build a dataset_info.json entry from the record keys."""
    rec = next((r for r in records if isinstance(r, dict)), None)
    if rec is None:
        raise DatasetError(f"{file_name}: no JSON object records found")
    keys = set(rec)
    entry: dict[str, Any] = {"file_name": file_name}
    msg_col = "messages" if "messages" in keys else ("conversations" if "conversations" in keys else None)
    is_sharegpt = msg_col is not None
    if fmt == "alpaca" and is_sharegpt and "instruction" not in keys:
        raise DatasetError(f"{file_name}: dataset_format=alpaca but records have '{msg_col}' (sharegpt)")
    if fmt == "sharegpt" and not is_sharegpt:
        raise DatasetError(f"{file_name}: dataset_format=sharegpt but records have no 'messages'/'conversations'")
    ranking = "chosen" in keys and "rejected" in keys
    if is_sharegpt and fmt != "alpaca":
        entry["formatting"] = "sharegpt"
        cols: dict[str, str] = {"messages": msg_col}
        if "system" in keys:
            cols["system"] = "system"
        if "tools" in keys:
            cols["tools"] = "tools"
        msgs = rec.get(msg_col) or []
        first = next((m for m in msgs if isinstance(m, dict)), {})
        if "role" in first and "content" in first:
            entry["tags"] = dict(OPENAI_TAGS)
        if ranking:
            entry["ranking"] = True
            cols["chosen"] = "chosen"
            cols["rejected"] = "rejected"
        if "kto_tag" in keys:
            cols["kto_tag"] = "kto_tag"
        entry["columns"] = cols
        return entry
    if "instruction" not in keys and "prompt" not in keys:
        if "text" in keys:
            raise DatasetError(f"{file_name}: plain 'text' records are pre-training data (stage pt), "
                               "which this worker does not run; use alpaca or sharegpt records")
        raise DatasetError(f"{file_name}: unrecognised records; expected alpaca (instruction/input/output) "
                           f"or sharegpt (conversations/messages). Keys seen: {sorted(keys)[:10]}")
    prompt = "instruction" if "instruction" in keys else "prompt"
    cols = {"prompt": prompt}
    if "input" in keys:
        cols["query"] = "input"
    if ranking:
        entry["ranking"] = True
        cols["chosen"] = "chosen"
        cols["rejected"] = "rejected"
    else:
        cols["response"] = "output" if "output" in keys else "response"
    for opt in ("system", "history", "kto_tag"):
        if opt in keys:
            cols[opt] = opt
    entry["columns"] = cols
    return entry


def validate_records(records: list[dict], entry: dict[str, Any]) -> tuple[int, int]:
    """Return (valid, invalid) counts for json/jsonl records."""
    cols = entry.get("columns", {})
    valid = invalid = 0
    for r in records:
        ok = isinstance(r, dict)
        if ok and entry.get("formatting") == "sharegpt":
            msgs = r.get(cols.get("messages", "conversations"))
            ok = isinstance(msgs, list) and len(msgs) >= (1 if entry.get("ranking") else 2)
            if ok and entry.get("ranking"):
                ok = bool(r.get("chosen")) and bool(r.get("rejected"))
        elif ok:
            ok = isinstance(r.get(cols.get("prompt", "instruction")), str) and bool(r.get(cols["prompt"]).strip())
            if ok and entry.get("ranking"):
                ok = bool(r.get("chosen")) and bool(r.get("rejected"))
            elif ok:
                resp = r.get(cols.get("response", "output"))
                ok = isinstance(resp, str) and bool(resp.strip())
        if ok and "kto_tag" in cols:
            ok = isinstance(r.get("kto_tag"), bool)
        valid += ok
        invalid += not ok
    return valid, invalid


def check_stage(entry: dict[str, Any], stage: str, name: str) -> None:
    ranking = bool(entry.get("ranking"))
    kto = "kto_tag" in entry.get("columns", {})
    if stage == "dpo" and not ranking:
        raise DatasetError(f"{name}: stage=dpo needs a preference dataset (chosen/rejected columns)")
    if stage == "sft" and ranking:
        raise DatasetError(f"{name}: chosen/rejected (preference) data needs stage=dpo, not sft")
    if stage == "kto" and not kto:
        raise DatasetError(f"{name}: stage=kto needs a boolean 'kto_tag' column")


# --------------------------------------------------------------------------- archives
def _safe_target(base: str, member: str) -> str:
    target = os.path.realpath(os.path.join(base, member))
    if not (target == os.path.realpath(base) or target.startswith(os.path.realpath(base) + os.sep)):
        raise DatasetError(f"archive member escapes the target dir: {member!r}")
    return target


def safe_extract(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    total = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = _safe_target(dest, info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise DatasetError(f"symlinks are not allowed in archives: {info.filename!r}")
                total += info.file_size
                if total > MAX_EXTRACT_BYTES:
                    raise DatasetError("archive expands to more than 8 GiB")
                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                _safe_target(dest, m.name)
                if m.issym() or m.islnk() or m.isdev():
                    raise DatasetError(f"links/devices are not allowed in archives: {m.name!r}")
                total += m.size
                if total > MAX_EXTRACT_BYTES:
                    raise DatasetError("archive expands to more than 8 GiB")
            t.extractall(dest, filter="data")
        return
    raise DatasetError("dataset_url is neither a JSON/JSONL/CSV/Parquet file nor a zip/tar archive")


def download(url: str, dest_dir: str) -> str:
    import requests

    os.makedirs(dest_dir, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(url.split("?", 1)[0])) or "dataset"
    path = os.path.join(dest_dir, name)
    got = 0
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code >= 400:
            raise DatasetError(f"dataset_url returned HTTP {r.status_code}")
        with open(path + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                got += len(chunk)
                if got > MAX_DOWNLOAD_BYTES:
                    raise DatasetError("dataset_url is larger than 2 GiB")
                f.write(chunk)
    os.replace(path + ".part", path)
    return path


# --------------------------------------------------------------------------- main entry
def _register_file(path: str, name: str, data_dir: str, fmt: str, stage: str, report: DatasetReport,
                   sample_limit: int | None) -> None:
    file_name = os.path.relpath(path, data_dir)
    if path.endswith((".json", ".jsonl")):
        recs = read_records(path)
        if not recs:
            raise DatasetError(f"{file_name}: dataset is empty")
        entry = detect_entry(recs, fmt, file_name)
        valid, invalid = validate_records(recs, entry)
        if valid == 0:
            raise DatasetError(f"{file_name}: no valid records for the detected format")
        report.rows[name] = valid + invalid
        if invalid:
            report.invalid_rows[name] = invalid
            report.warnings.append(f"{name}: {invalid} of {valid + invalid} records look invalid and may be "
                                   "skipped or fail in LlamaFactory")
        if report.preview is None:
            report.preview = recs[0]
    else:
        if fmt == "auto":
            raise DatasetError(f"{file_name}: set dataset_format (alpaca/sharegpt) for csv/parquet/arrow files")
        entry = {"file_name": file_name, "formatting": fmt}
        report.rows[name] = None
    check_stage(entry, stage, name)
    report.entries[name] = entry
    report.names.append(name)


def prepare_dataset(cfg, work_dir: str, sample_limit: int | None = None) -> tuple[str, DatasetReport]:
    """Materialise all training data under <work_dir>/data and write dataset_info.json.

    Returns (dataset_dir, report). Raises DatasetError on any user-data problem.
    """
    data_dir = os.path.join(work_dir, "data")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)
    report = DatasetReport()
    stage = cfg.stage
    fmt = cfg.dataset_format
    names = list(cfg.dataset_names)

    if cfg.dataset_inline is not None:
        path = os.path.join(data_dir, "inline.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.dataset_inline, f, ensure_ascii=False)
        _register_file(path, "inline", data_dir, fmt, stage, report, sample_limit)

    if cfg.dataset_url:
        raw = download(cfg.dataset_url, os.path.join(work_dir, "download"))
        low = raw.lower()
        if low.endswith(DATA_EXTS):
            dst = os.path.join(data_dir, "user_data" + os.path.splitext(low)[1])
            shutil.move(raw, dst)
            _register_file(dst, "user_data", data_dir, fmt, stage, report, sample_limit)
        else:
            ext_dir = os.path.join(data_dir, "user")
            safe_extract(raw, ext_dir)
            os.remove(raw)
            info_path = _find(ext_dir, "dataset_info.json")
            if info_path:
                # Archive ships its own LlamaFactory dataset_info.json: names select entries in it.
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                base = os.path.dirname(info_path)
                pick = names or list(info)
                for n in pick:
                    if n not in info:
                        raise DatasetError(f"dataset '{n}' is not in the archive's dataset_info.json "
                                           f"(entries: {sorted(info)[:20]})")
                    e = dict(info[n])
                    for remote in ("hf_hub_url", "ms_hub_url", "om_hub_url", "script_url", "cloud_file_name"):
                        if remote in e:
                            raise DatasetError(f"archive entry '{n}' uses {remote}; only local files are allowed")
                    fpath = os.path.join(base, e.get("file_name", ""))
                    _safe_target(ext_dir, os.path.relpath(fpath, ext_dir))
                    if not os.path.exists(fpath):
                        raise DatasetError(f"archive entry '{n}': file {e.get('file_name')!r} not found")
                    e["file_name"] = os.path.relpath(fpath, data_dir)
                    check_stage(e, stage, n)
                    report.entries[n] = e
                    report.names.append(n)
                    report.rows[n] = None
                names = []
            else:
                files = sorted(p for p in _walk(ext_dir) if p.lower().endswith(DATA_EXTS))
                if not files:
                    raise DatasetError("archive has no .json/.jsonl/.csv/.parquet data files")
                for i, p in enumerate(files):
                    _register_file(p, "user_data" if i == 0 else f"user_data_{i}", data_dir, fmt, stage,
                                   report, sample_limit)

    for n in names:
        if n in BUNDLED:
            src = os.path.join(bundled_datasets_dir(), BUNDLED[n])
            dst = os.path.join(data_dir, BUNDLED[n])
            shutil.copy2(src, dst)
            _register_file(dst, n, data_dir, fmt, stage, report, sample_limit)
            continue
        up = _upstream_entry(n)
        if up is not None:
            e = dict(up)
            if "file_name" in e and not any(k in e for k in ("hf_hub_url", "ms_hub_url", "script_url")):
                src = os.path.join(upstream_data_dir(), e["file_name"])
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(data_dir, os.path.basename(src)))
                    e["file_name"] = os.path.basename(src)
            check_stage(e, stage, n)
            report.entries[n] = e
            report.names.append(n)
            report.rows[n] = None
            continue
        if "/" in n and re.match(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", n):
            e = {"hf_hub_url": n}
            if fmt == "sharegpt":
                e["formatting"] = "sharegpt"
            if stage == "dpo":
                e["ranking"] = True
                e["columns"] = {"prompt": "instruction", "query": "input", "chosen": "chosen",
                                "rejected": "rejected"} if fmt != "sharegpt" else {
                    "messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
            report.entries[n] = e
            report.names.append(n)
            report.rows[n] = None
            report.warnings.append(f"{n}: Hugging Face dataset; columns are assumed to follow the default "
                                   f"{fmt if fmt != 'auto' else 'alpaca'} layout. Use config_yaml + a dataset "
                                   "archive with its own dataset_info.json for custom column names")
            continue
        raise DatasetError(f"unknown dataset '{n}': use a Hugging Face id (owner/name), a bundled set "
                           f"({', '.join(sorted(BUNDLED))}), or an entry of LlamaFactory's data/dataset_info.json")

    if not report.names:
        raise DatasetError("no training data was prepared")
    with open(os.path.join(data_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(report.entries, f, indent=2, ensure_ascii=False)
    return data_dir, report


def _upstream_entry(name: str) -> dict | None:
    path = os.path.join(upstream_data_dir(), "dataset_info.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get(name)


def _walk(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith((".", "__MACOSX"))]
        for fn in files:
            if not fn.startswith("."):
                yield os.path.join(d, fn)


def _find(root: str, fname: str) -> str | None:
    hits = sorted((p for p in _walk(root) if os.path.basename(p) == fname), key=len)
    return hits[0] if hits else None
