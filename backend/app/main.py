"""FastAPI entrypoint for the MiniMax-H3 WebUI backend.

Wires M1~M5 routers, health probes, and DB bootstrap. The worker (rq) imports
`app.worker.run_generation` directly; the API only enqueues tasks.
"""
from __future__ import annotations

import logging
import os
import sys

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import assets_router, generations_router, results_router, tasks_router, templates_router
from .config import settings
from .db import engine, init_db

# frontend 包可能位于仓库根的 frontend/（本地裸跑）或容器 /app/frontend（镜像）。
# 把它加入 sys.path 以便 `from frontend.app import build_ui` 可达。
_FRONTEND_DIR = os.environ.get("FRONTEND_DIR")
if _FRONTEND_DIR and _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("api")

app = FastAPI(
    title="MiniMax-H3 WebUI API",
    description="文字/图片/音频 -> 带音频视频 异步生成服务 (Linux WebUI)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (assets_router, templates_router, generations_router, tasks_router, results_router):
    app.include_router(r)

# 单端口云端部署：把 Gradio UI 直接挂载到后端 "/" 上，访问 <实例>:7860 即打开 UI，
# 且 UI 通过同源相对路径调用 /api/v1/*（无需第二端口 / CORS）。FRONTEND_MOUNT=0 时
# 可禁用挂载（例如单独的 Gradio 服务部署）。
if os.environ.get("FRONTEND_MOUNT", "1") != "0":
    try:
        from frontend.app import build_ui
        _ui = build_ui()
        app = gr.mount_gradio_app(app, _ui, path="/")
        log.info("Gradio UI mounted at /")
    except Exception as e:  # pragma: no cover
        log.warning("Gradio UI mount skipped: %s", e)


@app.on_event("startup")
def _startup():
    init_db()
    log.info("DB initialized; comfyui=%s fallback=%s minimax=%s",
             settings.comfyui_url, settings.use_fallback,
             bool(settings.minimax_api_key))


@app.get("/healthz", tags=["ops"])
def healthz():
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz():
    try:
        with engine.connect() as c:
            c.execute(__import__("sqlalchemy").text("SELECT 1"))
        return {"status": "ok", "db": "up"}
    except Exception as e:  # pragma: no cover
        return {"status": "degraded", "db": "down", "detail": str(e)}, 503
