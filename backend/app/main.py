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

from .api import assets_router, generations_router, loras_router, results_router, settings_router, tasks_router, templates_router
from .config import settings
from .db import engine, init_db

# frontend 包可能位于：
#   - 仓库根的 frontend/（本地裸跑：<root>/backend/app/main.py 与 <root>/frontend 同级）
#   - 容器 /app/frontend（镜像，由 Dockerfile ENV FRONTEND_DIR=/app/frontend 指定）
# 自动解析，优先用 FRONTEND_DIR，否则向上查找含 app.py 的 frontend 目录。
def _resolve_frontend_dir() -> str | None:
    env = os.environ.get("FRONTEND_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))  # .../backend/app
    cur = here
    for _ in range(4):  # 向上最多 4 层：app -> backend -> root -> ...
        cand = os.path.join(cur, "frontend")
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "app.py")):
            return cand
        cur = os.path.dirname(cur)
    if os.path.isdir("/app/frontend") and os.path.isfile("/app/frontend/app.py"):
        return "/app/frontend"
    return None


_FRONTEND_DIR = _resolve_frontend_dir()
if _FRONTEND_DIR:
    # `from frontend.app import build_ui` 需要 `frontend` 作为（namespace）包可被解析，
    # 因此必须把它的 *父目录* 加入 sys.path（Docker 里靠 PYTHONPATH=/app 生效）。
    # 把 frontend 目录本身加入是无效的。
    _frontend_parent = os.path.dirname(_FRONTEND_DIR)
    if _frontend_parent not in sys.path:
        sys.path.insert(0, _frontend_parent)

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

for r in (assets_router, templates_router, generations_router, tasks_router,
          results_router, loras_router, settings_router):
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
