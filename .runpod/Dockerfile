# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built public image instead of compiling the worker again.
# The real build recipe lives in the repository-root Dockerfile.
#
# Pin the digest (not a floating :latest). Releases of this repo must still
# use targetCommitish=main — this digest is the image pin, not the git target.
FROM ghcr.io/prilojject/runpod-llama-factory:v1.0.0@sha256:d0988f53a87919554856f9a3cca1dda4f2d2942406bed75e9de955788dfdd0f4
