"""M3 — 生成编排 (GenerationRequest 创建 + Task 提交).

Frontend flow (frontend/app.py):
  1. POST /api/v1/generations            -> 创建 GenerationRequest，返回 {id}
  2. POST /api/v1/generations/{id}/submit -> 派生 Task(PENDING) 并入队，返回 {id}

Note: the ORM t_generation_request table only carries `optimized_prompt`
(the text the worker ultimately renders). The user-entered `prompt` is stored
there directly; the M2 "/optimize" endpoint returns an optimized string that the
frontend writes back into the prompt box before submit, so by submit time the
box already holds the final text.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import GenerationRequest, Task
from ..schemas import (
    GenerationCreateRequest,
    GenerationRequestVO,
    PageResponse,
    TaskVO,
)
from ..tasks_queue import enqueue_generation

router = APIRouter(prefix="/api/v1/generations", tags=["generations"])


class _SubmitBody(BaseModel):
    force_engine: Optional[str] = None  # comfyui / minimax / None


@router.post("", response_model=GenerationRequestVO, status_code=201)
def create_generation(req: GenerationCreateRequest, db: Session = Depends(get_db)):
    use_fallback = settings.use_fallback if req.use_fallback is None else req.use_fallback
    gen = GenerationRequest(
        optimized_prompt=req.prompt,
        reference_asset_ids=json.dumps(req.reference_asset_ids),
        video_asset_ids=json.dumps(req.video_asset_ids),
        mode=req.mode or "reference",
        first_stage_resolution=req.first_stage_resolution or "360P",
        loras=json.dumps([s.model_dump() for s in req.loras]),
        step=req.step,
        seed=req.seed,
        width=req.width,
        height=req.height,
        duration=req.duration,
        fps=req.fps,
        lora_id=req.lora_id or "fl2v_turbo_8step_v1.0",
        use_fallback=bool(use_fallback),
        template_id=0,
        status="CREATED",
    )
    db.add(gen)
    db.commit()
    db.refresh(gen)
    return GenerationRequestVO.model_validate(gen)


@router.get("", response_model=PageResponse)
def list_generations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    q = db.query(GenerationRequest).order_by(desc(GenerationRequest.created_at))
    total = q.count()
    items = q.limit(page_size).offset((page - 1) * page_size).all()
    return PageResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[GenerationRequestVO.model_validate(g) for g in items],
    )


@router.get("/{generation_id}", response_model=GenerationRequestVO)
def get_generation(generation_id: int, db: Session = Depends(get_db)):
    g = db.get(GenerationRequest, generation_id)
    if not g:
        raise HTTPException(404, "generation request not found")
    return GenerationRequestVO.model_validate(g)


@router.post("/{generation_id}/submit", response_model=TaskVO, status_code=201)
def submit_generation(
    generation_id: int,
    body: Optional[_SubmitBody] = Body(default=None),
    db: Session = Depends(get_db),
):
    gen = db.get(GenerationRequest, generation_id)
    if not gen:
        raise HTTPException(404, "generation request not found")
    if gen.status not in ("CREATED", "SUBMITTED"):
        raise HTTPException(409, f"generation 状态不可提交: {gen.status}")

    force_engine = body.force_engine if body else None
    task = Task(
        generation_request_id=gen.id,
        tenant_id=gen.tenant_id,
        engine="comfyui",  # 占位，worker 在 _run 中按路由重写
        status="PENDING",
        progress=0,
        retry_count=0,
        force_engine=force_engine,
        version=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    gen.status = "SUBMITTED"
    db.commit()

    enqueue_generation(task.id)
    return TaskVO.model_validate(task)
