# Gradio MVP 前端
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
