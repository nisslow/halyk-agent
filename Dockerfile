# Halyk AI Challenge Agent - Dockerfile
# Multi-stage build for optimal size

# =========================================================================
# Stage 1: Base with system dependencies
# =========================================================================
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    cmake \
    libpq-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kaz \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser
WORKDIR /app

# =========================================================================
# Stage 2: Python dependencies
# =========================================================================
FROM base AS deps

COPY pyproject.toml ./
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -e ".[dev]"

# Pre-download models to cache
RUN python3.11 -c "
from flag_embedding import FlagModel
model = FlagModel('BAAI/bge-m3', devices=['cpu'])
print('bge-m3 cached')
"

# =========================================================================
# Stage 3: Application
# =========================================================================
FROM base AS app

COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser pyproject.toml ./

# Create data directories
RUN mkdir -p /app/data/{raw,processed,kuzu_db} && chown -R appuser:appuser /app/data

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3.11 -c "import sys; sys.exit(0)"

# Default command
ENTRYPOINT ["python3.11", "-m", "halyk_agent.main"]
CMD ["--help"]

# =========================================================================
# Stage 4: Development (optional)
# =========================================================================
FROM app AS dev

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER appuser
CMD ["bash"]