# Headless LlamaFactory trainer for Runpod Serverless (LoRA / QLoRA / freeze / full; SFT, DPO/ORPO/SimPO, KTO).
# No model weights are baked in: the base model is downloaded at job start and cached on the network
# volume (/runpod-volume/llama-factory/models) when one is attached.
# No Gradio / LLaMA Board: gradio is not even installed; the worker writes a YAML and runs `llamafactory-cli train`.
#
# Upstream pin: release tag v0.9.5 + its commit SHA (checked at build time). Never the floating main.
# This pin is only for the image; our own GitHub release for the Hub uses targetCommitish=main.
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG LLAMAFACTORY_TAG=v0.9.5
ARG LLAMAFACTORY_SHA=7af909522a951e3ad9f022ea6f88b6755257eaa5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_ANALYTICS_ENABLED=False \
    LLAMAFACTORY_DIR=/opt/llamafactory

# git: clone the pinned upstream; build-essential: triton/bitsandbytes JIT helpers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch ${LLAMAFACTORY_TAG} --depth 1 https://github.com/hiyouga/LlamaFactory.git ${LLAMAFACTORY_DIR} \
    && cd ${LLAMAFACTORY_DIR} \
    && test "$(git rev-parse HEAD)" = "${LLAMAFACTORY_SHA}" \
    && git rev-parse HEAD > /opt/llamafactory.commit \
    && rm -rf .git

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Keep the base image's torch stack: freeze it as constraints so pip cannot swap it.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && pip install --no-deps "${LLAMAFACTORY_DIR}" \
    && pip freeze > /app/requirements.lock \
    && python -c "import torch, transformers, peft, trl, datasets, accelerate, bitsandbytes, runpod, llamafactory; \
print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, \
'| transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__, \
'| datasets', datasets.__version__, '| accelerate', accelerate.__version__, '| bnb', bitsandbytes.__version__, \
'| llamafactory', llamafactory.__version__, open('/opt/llamafactory.commit').read().strip())" \
    && llamafactory-cli version

COPY handler.py /app/handler.py
COPY src /app/src
COPY examples/datasets /app/examples/datasets

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
CMD ["python", "-u", "/app/handler.py"]
