FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    FFMPEG_THREADS=1 \
    RENDER_RESOLUTION=720p \
    WORKER_CONCURRENCY=1

# System packages: FFmpeg + Noto fonts (Latin, Devanagari for Hindi, Tamil for Tamil) + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-noto-core \
    fonts-noto-extra \
    fontconfig \
    curl \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY app /app/app
COPY scripts /app/scripts

RUN pip install --upgrade pip && pip install ".[dev]"

# Create non-root user compatible with Docker Compose AND Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/.local_storage /app/.tmp_jobs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
