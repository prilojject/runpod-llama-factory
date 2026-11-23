import logging

from src.secret_utils import RedactFilter, get_hf_token, redact


def test_unresolved_secret_is_none(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "{{ RUNPOD_SECRET_hf_token }}")
    assert get_hf_token() is None


def test_redaction(monkeypatch, caplog):
    tok = "hf_" + "s" * 30
    monkeypatch.setenv("HF_TOKEN", tok)
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "bucketsecret123")
    assert tok not in redact(f"x {tok} y") and "bucketsecret123" not in redact("k=bucketsecret123")
    logger = logging.getLogger("t")
    logger.addFilter(RedactFilter())
    with caplog.at_level(logging.INFO):
        logger.info("token=%s", tok)
    assert tok not in caplog.text and "***" in caplog.text
