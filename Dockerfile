# Test/evaluation image: runs the deterministic suite by default. Not a serving image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY requirements.lock ./
RUN python -m pip install --upgrade pip && pip install -r requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app

USER app

CMD ["pytest", "-v", "--cov=clinguard_rag", "--cov-fail-under=85"]
