"""M5 — 结果画廊与交付 (Result)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Result
from ..schemas import PageResponse, ResultVO

router = APIRouter(prefix="/api/v1/results", tags=["results"])


@router.get("", response_model=PageResponse)
def list_results(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                 db: Session = Depends(get_db)):
    q = db.query(Result).filter(Result.status == "READY").order_by(desc(Result.created_at))
    total = q.count()
    items = q.limit(page_size).offset((page - 1) * page_size).all()
    return PageResponse(
        total=total, page=page, page_size=page_size,
        items=[ResultVO.model_validate(r) for r in items],
    )


@router.get("/{result_id}", response_model=ResultVO)
def get_result(result_id: int, db: Session = Depends(get_db)):
    r = db.get(Result, result_id)
    if not r:
        raise HTTPException(404, "result not found")
    return ResultVO.model_validate(r)


@router.get("/{result_id}/download")
def download_result(result_id: int, db: Session = Depends(get_db)):
    r = db.get(Result, result_id)
    if not r:
        raise HTTPException(404, "result not found")
    p = Path(r.file_path)
    if not p.exists():
        raise HTTPException(404, "A050001 文件缺失")
    return FileResponse(str(p), media_type="video/mp4",
                       filename=p.name)
