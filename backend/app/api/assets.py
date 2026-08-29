"""M1 — 上传与素材管理 (Asset)."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Asset
from ..schemas import AssetVO

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


_MEDIA = {"image": 100, "audio": 200, "video": 200, "lora": 500}

# 扩展名 -> 媒体类型（当 content-type 不可靠时，如 .safetensors 常为 octet-stream）
_EXT_MEDIA = {
    ".safetensors": "lora", ".pt": "lora", ".ckpt": "lora", ".bin": "lora",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".bmp": "image",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".aac": "audio", ".m4a": "audio",
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
}


@router.post("", response_model=AssetVO, status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    media_type: str | None = None,
    db: Session = Depends(get_db),
):
    # 媒体类型优先级：显式参数 > content-type > 扩展名推断 > 兜底
    media = (media_type or (file.content_type or "").split("/")[0]).lower()
    if media not in _MEDIA:
        ext = (Path(file.filename or "bin").suffix or "").lower()
        media = _EXT_MEDIA.get(ext, "image" if file.filename else "audio")
    limit = _MEDIA.get(media, 200)
    data = await file.read()
    if len(data) > limit * 1024 * 1024:
        raise HTTPException(413, f"{media} 超过上限 {limit}MB")
    if len(data) == 0:
        raise HTTPException(400, "空文件")

    ext = Path(file.filename or "bin").suffix or ".bin"
    name = f"{uuid.uuid4().hex}{ext}"
    dest = Path(settings.uploads_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    asset = Asset(
        filename=file.filename or name, media_type=media,
        mime=file.content_type, size_bytes=len(data),
        storage_path=str(dest.resolve()), status="READY",
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return AssetVO.model_validate(asset)


@router.get("/{asset_id}", response_model=AssetVO)
def get_asset(asset_id: int, db: Session = Depends(get_db)):
    a = db.get(Asset, asset_id)
    if not a:
        raise HTTPException(404, "asset not found")
    return AssetVO.model_validate(a)


@router.delete("/{asset_id}", status_code=204)
def delete_asset(asset_id: int, db: Session = Depends(get_db)):
    a = db.get(Asset, asset_id)
    if not a:
        raise HTTPException(404, "asset not found")
    db.delete(a)
    db.commit()
