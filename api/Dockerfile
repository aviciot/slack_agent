FROM python:3.12.8-slim

ENV PYTHONPATH=/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# App deps
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY src /app/src

EXPOSE 8090

HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD curl -fsS http://localhost:8090/healthz || exit 1

CMD ["python", "-m", "src.app.main"]
