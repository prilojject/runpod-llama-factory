"""Runpod Serverless worker for hiyouga/LlamaFactory (headless `llamafactory-cli train`)."""

__version__ = "1.0.0"

# Upstream pins (the Dockerfile ARGs must match; tests check this).
LLAMAFACTORY_TAG = "v0.9.5"
LLAMAFACTORY_SHA = "7af909522a951e3ad9f022ea6f88b6755257eaa5"
