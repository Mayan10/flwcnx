FROM python:3.11-slim

WORKDIR /app/flwcnx

# The pip cache is a BuildKit cache mount, so a build interrupted by a network
# timeout keeps what it already downloaded and the retry resumes from there.
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU torch first: the default wheel pulls several GB of CUDA libraries that a
# container on a laptop never uses. pip then sees torch satisfied and skips it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

# Install dependencies before copying the source so code edits reuse this layer.
COPY flwcnx/pyproject.toml flwcnx/README.md flwcnx/LICENSE ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir flwcnx && touch flwcnx/__init__.py && \
    pip install -e ".[api]" && \
    rm -rf flwcnx

COPY flwcnx/flwcnx ./flwcnx

EXPOSE 8000

CMD ["python", "-m", "flwcnx.api.server", "--mode", "synthetic", "--host", "0.0.0.0", "--port", "8000"]
