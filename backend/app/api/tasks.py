"""M4 — 任务队列与进度 (Task)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Task
from ..schemas import PageResponse, TaskRetryRequest, TaskVO
from ..tasks_queue import enqueue_generation

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("", response_model=PageResponse)
def list_tasks(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               db: Session = Depends(get_db)):
    q = db.query(Task).order_by(desc(Task.created_at))
    total = q.count()
    items = q.limit(page_size).offset((page - 1) * page_size).all()
    return PageResponse(
        total=total, page=page, page_size=page_size,
        items=[TaskVO.model_validate(t) for t in items],
    )


@router.get("/{task_id}", response_model=TaskVO)
def get_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(404, "task not found")
    return TaskVO.model_validate(t)


@router.post("/{task_id}/retry", response_model=TaskVO)
def retry_task(task_id: int, req: TaskRetryRequest | None = None,
               db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(404, "task not found")
    if t.status not in ("FAILED",):
        raise HTTPException(409, "仅失败任务可重试")
    t.status = "PENDING"
    t.retry_count = (t.retry_count or 0) + 1
    t.error_msg = None
    t.progress = 0
    if req and req.force_engine:
        t.force_engine = req.force_engine
    db.commit()
    db.refresh(t)
    enqueue_generation(t.id)
    return TaskVO.model_validate(t)
