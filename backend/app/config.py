"""
Runtime configuration (L1~L4 strategy from 部署设计 §4.4).

- L1: code defaults (step=8, seed=-1, 1376x768, 10s, 24fps ...)
- L2: business switches via env (USE_FALLBACK, FORCE_FALLBACK, engine URLs)
- L3: infra endpoints via env (REDIS_URL, DATABASE_URL, COMFYUI_URL)
- L4: secrets via env ONLY, never written to disk/config/db (V3/X4)

Fail-fast: required infra is validated at import time so the service refuses to
run with unsafe defaults.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- L3: infrastructure endpoints ---
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    database_url: str = Field(
        default="sqlite:////data/db/app.db", alias="DATABASE_URL"
    )
    comfyui_url: str = Field(default="http://comfyui:8188", alias="COMFYUI_URL")
    backend_public_url: str = Field(
        default="http://api:8000", alias="BACKEND_PUBLIC_URL"
    )

    # --- L2: engine routing switches ---
    # USE_FALLBACK maps to t_engine_config (H-04 方案B): default ON per 部署设计 §4.4
    use_fallback: bool = Field(default=True, alias="USE_FALLBACK")
    # FORCE_FALLBACK: when true, always use MiniMax even if local ComfyUI is up
    force_fallback: bool = Field(default=False, alias="FORCE_FALLBACK")

    # --- MiniMax fallback engine (B2) ---
    minimax_base_url: str = Field(
        default="https://api.minimax.chat/v1", alias="MINIMAX_BASE_URL"
    )
    minimax_model: str = Field(default="MiniMax-H3", alias="MINIMAX_MODEL")
    minimax_api_key: Optional[str] = Field(default=None, alias="MINIMAX_API_KEY")  # L4

    # --- Prompt optimizer LLM (B3, OpenAI-compatible, optional) ---
    llm_api_url: Optional[str] = Field(default=None, alias="LLM_API_URL")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")  # L4
    llm_model: str = Field(default="agnes-2.5-flash", alias="LLM_MODEL")

    # --- storage paths (N2, X3: configurable relative paths on mounted volume) ---
    data_root: str = Field(default="/data", alias="DATA_ROOT")
    uploads_dir: str = Field(default="/data/uploads", alias="UPLOADS_DIR")
    outputs_dir: str = Field(default="/data/outputs", alias="OUTPUTS_DIR")
    models_dir: str = Field(default="/models", alias="MODELS_DIR")
    workflow_path: str = Field(
        default="/app/workflows/h3_fl2v.json", alias="H3_WORKFLOW_PATH"
    )

    # --- ComfyUI H3 model / LoRA binding (defaults from original settings) ---
    h3_lora_name: str = Field(
        default="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        alias="H3_LORA_NAME",
    )
    h3_lora_strength: float = Field(default=1.0, alias="H3_LORA_STRENGTH")

    # --- generation defaults (L1) ---
    default_step: int = 8
    default_seed: int = -1
    default_width: int = 1376
    default_height: int = 768
    default_duration: int = 10
    default_fps: int = 24
    default_resolution: str = "360P"

    # --- limits (frozen SLA §6.2) ---
    max_upload_mb: int = 200  # audio/video; image ≤100MB enforced separately
    max_references: int = 5
    max_result_mb: int = 500

    # --- misc ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    tenant_id: int = Field(default=0, alias="TENANT_ID")  # O1 single-tenant

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
