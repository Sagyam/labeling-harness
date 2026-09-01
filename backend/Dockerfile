FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir -e .

COPY alembic.ini ./
COPY migrations ./migrations
COPY tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
