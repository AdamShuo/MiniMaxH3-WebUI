"""RQ worker — consumes generation tasks and drives the engine(s).

This is the async core wrapped in a sync entrypoint so it can run under `rq worker`
or be invoked directly. Mirrors 系统设计 §3.2.M4 state machine (PENDING -> RUNNING
-> SUCCEEDED | FAILED, with local_fail -> fallback switch to MiniMax).
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from .config import settings
from .db import SessionLocal
from .engine.base import GenParams
from .engine.router import build_engines, decide_primary
from .models import Asset, GenerationRequest, Result, Task, TaskProgress

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("worker")


def run_generation(task_id: int) -> None:
    asyncio.run(_run(task_id))


def _engines():
    return build_engines(
        comfyui_url=settings.comfyui_url if _comfyui_configured() else None,
        workflow_path=settings.workflow_path,
        models_dir=settings.models_dir,
        outputs_dir=settings.outputs_dir,
        minimax_base_url=settings.minimax_base_url,
        minimax_api_key=settings.minimax_api_key,
        minimax_model=settings.minimax_model,
    )


def _comfyui_configured() -> bool:
    # ComfyUI is optional (enabled via profile when a GPU is attached).
    return os.environ.get("ENABLE_COMFYUI", "false").lower() == "true"


async def _run(task_id: int) -> None:
    engines = _engines()
    has_comfyui = "comfyui" in engines
    has_minimax = "minimax" in engines

    db = SessionLocal()
    try:
        task = db.get(Task, task_id)
        if not task or task.status != "PENDING":
            log.warning("task %s not pending, skip", task_id)
            return
        gen = db.get(GenerationRequest, task.generation_request_id)
        task.status = "RUNNING"
        task.engine = decide_primary(task.force_engine, gen.use_fallback, has_comfyui, has_minimax)
        db.commit()

        params = GenParams(
            step=gen.step, seed=gen.seed, width=gen.width, height=gen.height,
            duration=gen.duration, fps=gen.fps,
            lora_name=settings.h3_lora_name, lora_strength=settings.h3_lora_strength,
            resolution=gen.optimized_prompt and "360P" or settings.default_resolution,
        )
        ref_paths = _reference_paths(db, gen.reference_asset_ids)
        audio_paths = _reference_paths(db, gen.reference_asset_ids, media="audio")

        async def progress_cb(progress: int, stage=None, message=None):
            _record_progress(db, task, progress, stage, message)

        engine = engines[task.engine]
        try:
            result = await engine.generate(
                prompt=gen.optimized_prompt or "",
                params=params, reference_paths=ref_paths, audio_paths=audio_paths,
                progress_cb=progress_cb, task_id=task.id,
            )
        except Exception as e:  # local fail -> fallback (BD-02)
            log.warning("engine %s failed: %s", task.engine, e)
            if task.engine == "comfyui" and has_minimax and gen.use_fallback:
                task.engine = "minimax"
                db.commit()
                log.info("switching to MiniMax fallback for task %s", task_id)
                result = await engines["minimax"].generate(
                    prompt=gen.optimized_prompt or "", params=params,
                    reference_paths=ref_paths, audio_paths=audio_paths,
                    progress_cb=progress_cb, task_id=task.id,
                )
            else:
                task.status = "FAILED"
                task.error_msg = str(e)[:500]
                db.commit()
                return

        # success
        rel = str(Path(result.file_path).resolve())
        rec = Result(
            tenant_id=task.tenant_id, task_id=task.id, engine=result.engine,
            file_path=rel, duration=result.duration, status="READY",
        )
        db.add(rec)
        db.flush()
        task.result_id = rec.id
        task.status = "SUCCEEDED"
        task.progress = 100
        db.commit()
        await progress_cb(100, "done", "完成")
    finally:
        db.close()


def _record_progress(db, task: Task, progress: int, stage, message):
    task.progress = max(task.progress, progress)
    db.add(TaskProgress(
        tenant_id=task.tenant_id, task_id=task.id,
        progress=progress, stage=stage, message=message,
    ))
    db.commit()


def _reference_paths(db, raw_ids, media: str | None = None) -> list[str]:
    if not raw_ids:
        return []
    try:
        ids = raw_ids if isinstance(raw_ids, list) else __import__("json").loads(raw_ids)
    except Exception:
        return []
    out = []
    for aid in ids:
        asset = db.get(Asset, int(aid))
        if asset and (media is None or asset.media_type == media):
            p = Path(asset.storage_path)
            if p.exists():
                out.append(str(p))
    return out
