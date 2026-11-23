# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built public image instead of compiling the worker again.
# The real build recipe lives in the repository-root Dockerfile.
#
# Pin the digest (not a floating :latest). Releases of this repo must still
# use targetCommitish=main — this digest is the image pin, not the git target.
FROM ghcr.io/prilojject/runpod-llama-factory:v1.0.0@sha256:d0988f53a87919554856f9a3cca1dda4f2d2942406bed75e9de955788dfdd0f4

# v1.0.1: build-time check only (no packages change). The image is
# pytorch/pytorch:2.7.1-cuda12.8 + bitsandbytes 0.48.2 (no flash-attn / xformers /
# other compiled torch extensions), so it has Blackwell kernels (sm_100 / sm_120).
# Fail the Hub build if that ever stops being true. cu128 wheels want a CUDA 12.8+
# driver, so allowedCudaVersions stays 12.8-13.4 on purpose.
RUN python -c "import torch, bitsandbytes; flags = torch._C._cuda_getArchFlags(); \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| archs', flags, '| bnb', bitsandbytes.__version__); \
assert torch.version.cuda == '12.8' and 'sm_120' in flags and 'sm_100' in flags and 'sm_89' in flags, flags"
