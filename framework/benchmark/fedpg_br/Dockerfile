# Flower FRL Benchmark Docker Image

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    swig \
    libosmesa6-dev \
    libgl1 \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: dependencies (cached unless pyproject.toml changes) ──────────────
COPY pyproject.toml .
# Minimal stub so hatchling can build the package metadata without real source
RUN mkdir -p frl_benchmark && touch frl_benchmark/__init__.py
# Install all dependencies (the stub package itself is thrown away after)
RUN pip install --no-cache-dir --disable-pip-version-check ".[dashboard]" && \
    pip uninstall -y flower-frl-benchmark

# ── Layer 2: source code (fast copy, no downloads) ────────────────────────────
COPY . .
# Install the real package — no deps to download, they're all cached above
RUN pip install --no-cache-dir --disable-pip-version-check --no-deps .

# Expose ports
# 8080 - Flower server
# 8050 - Web dashboard
EXPOSE 8080 8050

# Default command: start the dashboard (training is launched from the UI)
CMD ["frl-dashboard"]
