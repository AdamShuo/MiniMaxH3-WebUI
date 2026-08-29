"""M1b — 本地 LoRA 文件列表.

直接扫描 models/loras 目录，返回可供前端下拉选择的文件名列表。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..config import settings

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])

_LORA_EXTS = {".safetensors", ".pt", ".ckpt", ".bin"}


@router.get("", response_model=list[str])
def list_loras() -> list[str]:
    """列出 MODELS_DIR/loras 下所有支持的 LoRA 文件名。"""
    lora_dir = Path(settings.models_dir) / "loras"
    if not lora_dir.exists():
        return []
    files = [
        f.name for f in lora_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _LORA_EXTS
    ]
    return sorted(files)
