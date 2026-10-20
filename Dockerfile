FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/heart_pipeline.joblib \
    MODEL_VERSION=1.0.0

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --default-timeout=1000 --retries 10 -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/heart_pipeline.joblib models/heart_pipeline.joblib

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
