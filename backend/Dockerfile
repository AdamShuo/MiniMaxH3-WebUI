# Backend API + 挂载式 Gradio UI（单端口云部署）。
# 构建上下文为仓库根目录（需要 COPY frontend/ 与 comfy_core/ 之外的文件）。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY backend/app /app/app
COPY backend/workflows /app/workflows
COPY prompt_guides /prompt_guides
COPY reference /app/reference
COPY frontend /app/frontend

ENV DATA_ROOT=/data \
    PROMPT_GUIDES_DIR=/prompt_guides \
    FRONTEND_DIR=/app/frontend \
    PYTHONPATH=/app

EXPOSE 7860

# 单端口：UI + API 同进程；访问 <实例>:7860 即打开 WebUI，UI 通过同源相对路径调用 /api/v1/*
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
