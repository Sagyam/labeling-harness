FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch and torchaudio (the model's feature extractor imports torchaudio), at the version
# the notebook ran. The rest is pinned in requirements.txt.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchaudio==2.11.0
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./

EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health', timeout=4)"

CMD ["python", "server.py"]
