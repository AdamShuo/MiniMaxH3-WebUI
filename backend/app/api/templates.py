"""M2 — 提示词模板与优化 (PromptTemplate / B3)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db, init_db
from ..models import PromptOptimizeRecord, PromptTemplate
from ..prompt_llm import optimize_prompt
from ..schemas import (
    PromptOptimizeRecordVO,
    PromptOptimizeRequest,
    PromptTemplateVO,
)

router = APIRouter(prefix="/api/v1/prompt-templates", tags=["templates"])


def _seed_templates(db: Session) -> None:
    """Load the scene guides from prompt_guides/manifest.json into the table (idempotent).

    manifest 结构：{"general": {...}, "scene_guides": [ {...}, ... ]}
    """
    manifest = Path(settings.data_root) / "prompt_guides" / "manifest.json"
    if not manifest.exists():
        manifest = Path("/prompt_guides/manifest.json")
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    existing = {t.template_key for t in db.query(PromptTemplate).all()}

    def _add(key, name, name_zh, guide_path, description=None):
        if not key or key in existing or key == "none":
            return
        db.add(PromptTemplate(
            template_key=key, name=name, name_zh=name_zh,
            description=description, guide_path=guide_path,
        ))

    gen = data.get("general") or {}
    if gen:
        _add(gen.get("id") or "h3_general", gen.get("name", "H3 General"),
             gen.get("name_zh"), gen.get("path"))

    for g in (data.get("scene_guides") or []):
        if not isinstance(g, dict):
            continue
        _add(g.get("id"), g.get("name", g.get("id")), g.get("name_zh"),
             g.get("path"))

    db.commit()


@router.on_event("startup")
def _startup():
    # 确保表已创建（不依赖 main 的 startup 顺序）
    init_db()
    db = next(get_db())
    try:
        _seed_templates(db)
    finally:
        db.close()


@router.get("", response_model=list[PromptTemplateVO])
def list_templates(db: Session = Depends(get_db)):
    _seed_templates(db)
    return [PromptTemplateVO.model_validate(t) for t in db.query(PromptTemplate).all()]


@router.get("/{template_id}", response_model=PromptTemplateVO)
def get_template(template_id: int, db: Session = Depends(get_db)):
    t = db.get(PromptTemplate, template_id)
    if not t:
        raise HTTPException(404, "template not found")
    return PromptTemplateVO.model_validate(t)


@router.post("/{template_id}/optimize", response_model=PromptOptimizeRecordVO)
async def optimize(template_id: int, req: PromptOptimizeRequest,
                  db: Session = Depends(get_db)):
    optimized = await optimize_prompt(req.prompt, req.model)
    rec = PromptOptimizeRecord(
        template_id=template_id, original_prompt=req.prompt,
        optimized_prompt=optimized, model=req.model or settings.llm_model,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return PromptOptimizeRecordVO.model_validate(rec)
