# Finnie — Streamlit app (and MCP server) in one image.
# Build:  docker compose build      Run:  docker compose up      → http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# CPU-only PyTorch keeps the image ~2 GB smaller; results are identical for our embedding model.
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

COPY config.yaml .
COPY src ./src
COPY data ./data
COPY scripts ./scripts

# Build the knowledge-base index at image build time (also downloads the embedding model once).
RUN python -m src.rag.ingest

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/web_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
