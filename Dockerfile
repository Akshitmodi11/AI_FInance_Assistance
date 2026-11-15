# Finnie — React app + FastAPI backend (and the MCP server) in one image.
# Build:  docker compose build      Run:  docker compose up      → http://localhost:8000

# ---- 1. build the React app (only the static files are kept) ----
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

# ---- 2. Python runtime: API, agents, RAG index ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app
# uv from PyPI (not a second image registry) — one less network dependency during builds
RUN pip install --no-cache-dir uv==0.5.11

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
# ...and the cross-encoder for rerank mode, so the container can run with HF_HUB_OFFLINE=1.
RUN python -c "from src.core.config import get_settings; from src.rag.reranker import CrossEncoderReranker; \
CrossEncoderReranker(get_settings().rag.rerank_model).score('warm up', ['ok'])"

COPY --from=web /web/dist ./frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# The original Streamlit UI is still in the image:
#   docker compose run --rm -p 8501:8501 finnie streamlit run src/web_app/app.py --server.address=0.0.0.0
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
