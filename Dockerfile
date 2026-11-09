FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CHURN_APP_ENV=production \
    CHURN_LOG_LEVEL=INFO \
    CHURN_API_HOST=0.0.0.0 \
    CHURN_API_PORT=8080

WORKDIR /app

# Create non-root service user for container security
RUN groupadd -r churngrp && useradd -r -g churngrp churnuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN mkdir -p data/interim data/processed models_store reports/figures && \
    chown -R churnuser:churngrp /app

USER churnuser

# Train and serialize model artifact during image build so container is ready to serve
RUN python3 scripts/run_pipeline.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["python3", "scripts/serve_api.py"]
