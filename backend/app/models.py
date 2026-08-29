"""ORM models — one table per domain object from 系统设计 §4.2.

Columns mirror the documented DDL (t_asset, t_generation_request, t_task,
t_task_progress, t_result, t_engine_config) plus lightweight tables for the
prompt-template / optimize-record sub-domains (M2).

NOTE on PK types: ids use `Integer` (not BigInteger) so SQLite creates them as
`INTEGER PRIMARY KEY` (a rowid alias that auto-increments). BigInteger PKs are NOT
auto-incrementing on SQLite and `AUTOINCREMENT` is rejected for them. `Integer`
maps to INT4 on PostgreSQL, which is sufficient for these tables and keeps the
models cross-database compatible. Non-id columns (tenant_id, version, sizes) stay
BigInteger.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Asset(Base):
    __tablename__ = "t_asset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)  # image/audio/video
    mime: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="READY")
    created_at: Mapped[datetime] = mapped_column(default=func.now())


class GenerationRequest(Base):
    __tablename__ = "t_generation_request"
    __table_args__ = (
        UniqueConstraint("tenant_id", "biz_id", name="uk_tenant_biz_gen"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    template_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    template_key: Mapped[str | None] = mapped_column(String(64), default=None, nullable=True)
    optimized_prompt: Mapped[str | None] = mapped_column(Text)
    reference_asset_ids: Mapped[str | None] = mapped_column(Text)  # JSON array of ids
    video_asset_ids: Mapped[str | None] = mapped_column(Text)  # JSON array of ids
    mode: Mapped[str] = mapped_column(
        String(32), default="reference", nullable=False
    )  # reference | first_frame | dual_stage
    first_stage_resolution: Mapped[str] = mapped_column(
        String(16), default="360P", nullable=False
    )  # 360P | 540P | 720P | 1080P
    loras: Mapped[str | None] = mapped_column(Text)  # JSON array of LoRASpec
    optimize_method: Mapped[str] = mapped_column(
        String(16), default="builtin", nullable=False
    )  # builtin | third_party
    text_encoder: Mapped[str] = mapped_column(
        String(255), default="clip", nullable=False
    )  # clip | text_encoding/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
    step: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, default=-1, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=1376, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=768, nullable=False)
    duration: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    fps: Mapped[int] = mapped_column(Integer, default=24, nullable=False)
    lora_id: Mapped[str] = mapped_column(
        String(128), default="fl2v_turbo_8step_v1.0", nullable=False
    )
    use_fallback: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    biz_id: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30), default="CREATED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )


class Task(Base):
    __tablename__ = "t_task"
    __table_args__ = (
        UniqueConstraint("tenant_id", "biz_id", name="uk_tenant_biz_task"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    generation_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("t_generation_request.id"), nullable=False
    )
    engine: Mapped[str] = mapped_column(String(20), default="comfyui", nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), default="PENDING", nullable=False
    )  # PENDING/RUNNING/SUCCEEDED/FAILED/RETRYING
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    result_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(String(512))
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    force_engine: Mapped[str | None] = mapped_column(String(20))  # comfyui/minimax/None
    biz_id: Mapped[str | None] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )

    progresses: Mapped[list["TaskProgress"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    result: Mapped["Result | None"] = relationship(back_populates="task")


class TaskProgress(Base):
    __tablename__ = "t_task_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("t_task.id"), nullable=False
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stage: Mapped[str | None] = mapped_column(String(30))
    message: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    task: Mapped["Task"] = relationship(back_populates="progresses")


class Result(Base):
    __tablename__ = "t_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("t_task.id"), nullable=False
    )
    engine: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    duration: Mapped[int | None] = mapped_column(Integer)
    thumbnail: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(30), default="READY", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )

    task: Mapped["Task"] = relationship(back_populates="result")


class EngineConfig(Base):
    __tablename__ = "t_engine_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    config_key: Mapped[str] = mapped_column(String(64), nullable=False)
    config_value: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )


class PromptTemplate(Base):
    __tablename__ = "t_prompt_template"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    template_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    name_zh: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    guide_path: Mapped[str | None] = mapped_column(String(512))


class PromptOptimizeRecord(Base):
    __tablename__ = "t_prompt_optimize_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    template_id: Mapped[int] = mapped_column(Integer, default=0)
    original_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    optimized_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=func.now())
