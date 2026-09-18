# ============================================================
# Stage 1: Build the React frontend
# ============================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

COPY package*.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src
COPY public ./public

# Use same-origin API calls in production.
ENV VITE_API_URL=

RUN npm run build

# ============================================================
# Stage 2: FastAPI + built frontend
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN useradd -m -u 1000 appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --from=frontend-builder /frontend/dist ./dist
COPY backend ./backend
COPY retrieval ./retrieval
COPY ingestion ./ingestion
COPY data ./data
COPY README.md ./README.md
COPY README_SETUP.md ./README_SETUP.md

RUN mkdir -p /home/app/.cache/huggingface /app/data/papers /app/data/processed && \
    chown -R appuser:appuser /app /home/app

USER appuser
ENV HOME=/home/app
ENV PATH=/home/app/.local/bin:$PATH

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
