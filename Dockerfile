FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=60 \
    TOKENIZERS_PARALLELISM=false \
    MANGA_COLORIZER_ISOLATE_SESSIONS=1 \
    MANGA_COLORIZER_REQUIRE_PASSWORD=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-ai.txt requirements-pdf.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements-base.txt \
    -r requirements-ai.txt \
    -r requirements-pdf.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]