"""M1b — 本地 LoRA 文件列表.

直接扫描 models/loras（或复数 models/lora）目录，返回可供前端下拉选择的
文件名列表。两个目录都会扫描并合并去重，以满足不同部署下的目录命名习惯。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..config import settings

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])

_LORA_EXTS = {".safetensors", ".pt", ".ckpt", ".bin"}
# 同时兼容单数/复数目录命名（models/lora 与 models/loras）
_LORA_DIRS = ["loras", "lora"]


@router.get("", response_model=list[str])
def list_loras() -> list[str]:
    """列出 MODELS_DIR 下 loras/lora 子目录中所有支持的 LoRA 文件名（合并去重）。"""
    base = Path(settings.models_dir)
    found: set[str] = set()
    for sub in _LORA_DIRS:
        lora_dir = base / sub
        if not lora_dir.exists():
            continue
        for f in lora_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _LORA_EXTS:
                found.add(f.name)
    return sorted(found)
