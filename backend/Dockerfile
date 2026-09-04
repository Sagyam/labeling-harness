FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Dependencies come from uv.lock, not from a fresh resolve. Every version in pyproject.toml is an
# open `>=` floor, so an image rebuilt months from now would otherwise pick up a different
# onnxruntime or numpy -- and those decide CTC alignment numerics and float rounding on a corpus
# whose whole purpose is measurement. D5 freezes the splits and D42 pins the acoustic model for
# the same reason; this closes the last unpinned input. The export carries hashes alongside the
# versions, so a substituted wheel fails the install rather than shipping.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The app installs without touching dependencies -- they are pinned above. Copying it after the
# dependency layer also stops a one-line code change from reinstalling all 65 packages.
COPY app ./app
RUN pip install --no-cache-dir --no-deps -e .

COPY alembic.ini ./
COPY migrations ./migrations
COPY tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
