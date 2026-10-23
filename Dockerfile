FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.lock README.md LICENSE ./
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN python -m pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.lock &&     pip install --no-cache-dir --no-deps -e .

ENV PYTHONPATH=/app/src:/app

CMD ["pytest", "-v", "--cov=clinguard_rag", "--cov-fail-under=85"]
