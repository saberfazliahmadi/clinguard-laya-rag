FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e ".[dev]"
ENV PYTHONPATH=/app/src:/app
CMD ["pytest", "-v", "--cov=clinguard_rag", "--cov-fail-under=85"]
