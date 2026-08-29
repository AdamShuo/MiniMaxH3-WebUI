"""B2 — MiniMax 官方视频生成 API 兜底引擎 (ACL 防腐层).

Implements the documented `/v1/video_generation` submit + poll flow (系统设计 §3.2.B2).
Region: Singapore (`api.minimax.chat`). Per H-04 方案B (已裁决) media may be uploaded
to the fallback endpoint; data-export compliance must be resolved before prod egress.

NOTE: this is the *test-ready* path — works as soon as MINIMAX_API_KEY is set, with
no local GPU required.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import httpx

from .base import BaseEngine, EngineResult, GenParams, ProgressCB


class MiniMaxEngine(BaseEngine):
    name = "minimax"

    def __init__(self, base_url: str, api_key: Optional[str], model: str, outputs_dir: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.outputs_dir = Path(outputs_dir)

    async def health(self) -> bool:
        # We cannot ping without a key; treat "key present" as healthy enough.
        return bool(self.api_key)

    async def generate(
        self,
        *,
        prompt: str,
        params: GenParams,
        reference_paths: list[str],
        audio_paths: list[str],
        video_paths: Optional[list[str]] = None,
        progress_cb: ProgressCB,
        task_id: int,
    ) -> EngineResult:
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY 未配置，无法使用兜底引擎")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "duration": params.duration,
        }
        # Optional multimodal references (public URLs). Local files are not uploaded
        # here; for MVP testing the prompt-only path is sufficient.
        if reference_paths:
            body["subject"] = [{"type": "image", "url": reference_paths[0]}]

        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                f"{self.base_url}/video_generation",
                headers=headers, json=body,
            )
            r.raise_for_status()
            data = r.json()
            task_id_ext = data.get("task_id") or data.get("id") or data.get("taskId")
            if not task_id_ext:
                raise RuntimeError(f"MiniMax 未返回 task_id: {data}")

        file_url = await self._poll(task_id_ext, headers, progress_cb)
        dest = await self._download(file_url, task_id)
        return EngineResult(file_path=str(dest), engine="minimax", duration=params.duration)

    async def _poll(self, task_id_ext: str, headers: dict, progress_cb: ProgressCB) -> str:
        url = f"{self.base_url}/video_generation/{task_id_ext}"
        deadline = time.time() + 1800
        async with httpx.AsyncClient(timeout=15.0) as c:
            while time.time() < deadline:
                r = await c.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
                status = (data.get("status") or "").lower()
                if status in ("success", "succeeded", "done"):
                    await progress_cb(95, "postprocess", "下载结果")
                    file_url = (
                        data.get("file_url")
                        or data.get("video_url")
                        or (data.get("output") or {}).get("url")
                    )
                    if not file_url:
                        raise RuntimeError(f"MiniMax 成功但无 file_url: {data}")
                    return file_url
                if status in ("fail", "failed", "error"):
                    raise RuntimeError(f"MiniMax 生成失败: {data.get('msg') or data}")
                await progress_cb(
                    min(80, int((1 - (deadline - time.time()) / 1800) * 80)),
                    "generating", "云端生成中",
                )
                await asyncio.sleep(5)
        raise RuntimeError("MiniMax 任务轮询超时")

    async def _download(self, file_url: str, task_id: int) -> Path:
        dest = self.outputs_dir / f"task_{task_id}_minimax.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as c:
            async with c.stream("GET", file_url) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
        return dest
