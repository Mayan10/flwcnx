# The ML decision service: the research package with its FastAPI server.
#
# The build context is the repository root, because that is where the package
# now lives. Everything the image does not need is excluded in .dockerignore,
# which matters more here than it did when the context was a subdirectory:
# the consoles, their node_modules and the recorded results are all in the
# context now and none of them belong in the image.
FROM python:3.11-slim

WORKDIR /app

# The pip cache is a BuildKit cache mount, so a build interrupted by a network
# timeout keeps what it already downloaded and the retry resumes from there.
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU torch first: the default wheel pulls several GB of CUDA libraries that a
# container on a laptop never uses. pip then sees torch satisfied and skips it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

# Install dependencies before copying the source so code edits reuse this layer.
COPY pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir flwcnx && touch flwcnx/__init__.py && \
    pip install -e ".[api]" && \
    rm -rf flwcnx

COPY flwcnx ./flwcnx

EXPOSE 8000

CMD ["python", "-m", "flwcnx.api.server", "--mode", "synthetic", "--host", "0.0.0.0", "--port", "8000"]
