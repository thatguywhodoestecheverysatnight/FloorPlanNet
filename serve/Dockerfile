# FloorPlanNet inference API: ONNX Runtime + OpenCV vectoriser (no torch, ~450 MB image)
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FLOORPLANNET_MODEL=/app/models/floorplannet_lite.onnx OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY floorplannet ./floorplannet
COPY serve ./serve
COPY web/public/models/floorplannet_lite.onnx ./models/floorplannet_lite.onnx
RUN useradd -m fpn && chown -R fpn /app
USER fpn
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}"]
