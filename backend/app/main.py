"""FastAPI entrypoint for the MiniMax-H3 WebUI backend.

Wires M1~M5 routers, health probes, and DB bootstrap. The worker (rq) imports
`app.worker.run_generation` directly; the API only enqueues tasks.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import assets_router, generations_router, results_router, tasks_router, templates_router
from .config import settings
from .db import engine, init_db

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
