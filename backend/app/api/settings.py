"""M6 — 提示词优化器设置 (OptimizerSettings).

将第三方 LLM API 配置持久化到 EngineConfig 表，优先级高于环境变量。
"""
from __future__ import annotations

import httpx

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import EngineConfig
from ..prompt_llm import optimize_prompt
from ..schemas import OptimizerSettingsRequest

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

_OPTIMIZER_KEYS = {
    "api_format": "openai",
    "api_url": "",
    "api_key": "",
    "api_model": "",
    "scene_guide": "",
}


def _get_optimizer_settings(db: Session) -> dict:
    """从 EngineConfig 读取优化器设置，缺失字段使用默认值。"""
    rows = {
        r.config_key: r.config_value
        for r in db.query(EngineConfig).filter(
            EngineConfig.config_key.in_(_OPTIMIZER_KEYS.keys())
        ).all()
    }
    return {k: rows.get(k, v) for k, v in _OPTIMIZER_KEYS.items()}


def _set_optimizer_settings(db: Session, req: OptimizerSettingsRequest) -> dict:
    """保存优化器设置到 EngineConfig 表。"""
    data = {
        "api_format": req.api_format or "openai",
        "api_url": req.api_url or "",
        "api_key": req.api_key or "",
        "api_model": req.api_model or "",
        "scene_guide": req.scene_guide or "",
    }
    existing = {
        r.config_key: r
        for r in db.query(EngineConfig).filter(
            EngineConfig.config_key.in_(data.keys())
        ).all()
    }
    for k, v in data.items():
        if k in existing:
            existing[k].config_value = v
        else:
            db.add(EngineConfig(config_key=k, config_value=v,
                                description=f"提示词优化器设置: {k}"))
    db.commit()
    return data


@router.get("/optimizer", response_model=dict)
def get_optimizer_settings(db: Session = Depends(get_db)):
    """获取当前保存的优化器设置。"""
    return _get_optimizer_settings(db)


@router.put("/optimizer", response_model=dict)
def update_optimizer_settings(req: OptimizerSettingsRequest,
                              db: Session = Depends(get_db)):
    """保存优化器设置。"""
    return _set_optimizer_settings(db, req)


class _TestRequest(BaseModel):
    api_url: str
    api_key: str
    api_model: str


@router.post("/optimizer/test")
async def test_optimizer_settings(req: _TestRequest):
    """测试第三方 API 连通性（发送一条最小 chat completion）。"""
    if not req.api_url or not req.api_key:
        raise HTTPException(400, "API 地址和 API Key 不能为空")
    body = {
        "model": req.api_model or settings.llm_model,
        "messages": [
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 5,
    }
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                f"{req.api_url.rstrip('/')}/chat/completions",
                headers=headers, json=body,
            )
            r.raise_for_status()
            return {"ok": True, "status": r.status_code,
                    "model": r.json().get("model")}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
