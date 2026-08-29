"""Pydantic DTOs — mirror 系统设计 §3.2 interface contracts (M1~M5, B1~B3)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- M1 素材 (Asset) ----------
class AssetVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    media_type: str
    mime: Optional[str] = None
    size_bytes: int = 0
    status: str = "READY"
    created_at: Optional[datetime] = None


# ---------- M2 模板 / 优化 ----------
class PromptTemplateVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    template_key: str
    name: str
    name_zh: Optional[str] = None
    description: Optional[str] = None


class PromptOptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="原始提示词")
    template_key: Optional[str] = None
    model: Optional[str] = None


class PromptOptimizeRecordVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    original_prompt: str
    optimized_prompt: str
    model: Optional[str] = None
    created_at: Optional[datetime] = None


# ---------- M3 生成编排 (Generation) ----------
class GenerationCreateRequest(BaseModel):
    template_key: Optional[str] = None
    prompt: str = Field(..., min_length=1, description="H3 提示词（支持 Ref2VA/FL2VA 结构）")
    reference_asset_ids: list[int] = Field(default_factory=list)
    step: int = 8
    seed: int = -1
    width: int = 1376
    height: int = 768
    duration: int = 10
    fps: int = 24
    lora_id: str = "fl2v_turbo_8step_v1.0"
    use_fallback: Optional[bool] = None  # 不传则用全局开关
    force_engine: Optional[str] = Field(
        default=None, description="comfyui / minimax / None（覆盖路由）"
    )
    biz_id: Optional[str] = Field(default=None, description="幂等号")


class GenerationRequestVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    template_id: int = 0
    optimized_prompt: Optional[str] = None
    reference_asset_ids: Optional[Any] = None
    step: int
    seed: int
    width: int
    height: int
    duration: int
    fps: int
    lora_id: str
    use_fallback: bool
    status: str
    created_at: Optional[datetime] = None


# ---------- M4 任务 (Task) ----------
class TaskVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    generation_request_id: int
    engine: str
    status: str
    progress: int = 0
    result_id: Optional[int] = None
    error_msg: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TaskRetryRequest(BaseModel):
    force_engine: Optional[str] = None


class PageRequest(BaseModel):
    page: int = 1
    page_size: int = 20


class PageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Any]


# ---------- M5 结果 (Result) ----------
class ResultVO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    task_id: int
    engine: str
    file_path: str
    duration: Optional[int] = None
    thumbnail: Optional[str] = None
    created_at: Optional[datetime] = None


# ---------- Engine progress callback (internal) ----------
class ProgressEvent(BaseModel):
    task_id: int
    progress: int
    stage: Optional[str] = None
    message: Optional[str] = None
