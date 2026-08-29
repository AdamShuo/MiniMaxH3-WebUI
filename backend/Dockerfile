# Backend (FastAPI API + RQ worker) — shared image, two process modes.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app /app/app
COPY workflows /app/workflows
COPY prompt_guides /prompt_guides
COPY reference /app/reference

ENV DATA_ROOT=/data \
    PROMPT_GUIDES_DIR=/prompt_guides

EXPOSE 8000

# Default: run the API. Override command in compose to run the worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
