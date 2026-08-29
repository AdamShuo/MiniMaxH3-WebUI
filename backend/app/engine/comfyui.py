"""B1 — 本地 ComfyUI 引擎适配 (ACL).

Submits a configurable H3 workflow JSON to ComfyUI's `/prompt` API, polls
`/history/{prompt_id}` for completion, then copies the produced video into the
shared outputs volume.

The H3 workflow is NOT shipped (the original `capai.exe` H3 node graph cannot be
extracted — see 系统设计 R-01/U-02). You must drop your exported ComfyUI H3
workflow at `H3_WORKFLOW_PATH` (default /app/workflows/h3_fl2v.json). The adapter
renders the following placeholders before submit:

  {{prompt}} {{seed}} {{steps}} {{width}} {{height}} {{duration}} {{fps}}
  {{lora_name}} {{lora_strength}} {{first_frame}} {{last_frame}} {{audio}}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from .base import BaseEngine, EngineResult, GenParams, ProgressCB

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class ComfyUIEngine(BaseEngine):
    name = "comfyui"

    def __init__(self, base_url: str, workflow_path: str, lora_dir: str, outputs_dir: str):
        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(workflow_path)
        self.lora_dir = Path(lora_dir)
        self.outputs_dir = Path(outputs_dir)
        self._final_path: Optional[Path] = None

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base_url}/system_stats")
                return r.status_code == 200
        except Exception:
            return False

    def _render(self, params: GenParams, prompt: str,
                reference_paths: list[str], audio_paths: list[str]) -> dict:
        if not self.workflow_path.exists():
            raise RuntimeError(
                f"H3 workflow not found at {self.workflow_path}. "
                "Export your ComfyUI H3 workflow there (see README)."
            )
        raw = self.workflow_path.read_text(encoding="utf-8")
        # ComfyUI workflow files may be a bare graph dict or {"prompt": {...}}
        wf = json.loads(raw)
        graph = wf["prompt"] if isinstance(wf, dict) and "prompt" in wf else wf

        values = {
            "prompt": prompt,
            "seed": params.seed if params.seed and params.seed > 0 else 0,
            "steps": params.step,
            "width": params.width,
            "height": params.height,
            "duration": params.duration,
            "fps": params.fps,
            "lora_name": params.lora_name,
            "lora_strength": params.lora_strength,
            "first_frame": reference_paths[0] if reference_paths else "",
            "last_frame": reference_paths[1] if len(reference_paths) > 1 else "",
            "audio": audio_paths[0] if audio_paths else "",
        }

        def sub(m: re.Match) -> str:
            return str(values.get(m.group(1), m.group(0)))

        rendered = _PLACEHOLDER.sub(sub, json.dumps(graph, ensure_ascii=False))
        return json.loads(rendered)

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
        graph = self._render(params, prompt, reference_paths, audio_paths)
        payload = {"prompt": graph, "client_id": f"minimax-h3-{task_id}"}

        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{self.base_url}/prompt", json=payload)
            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]

        await self._wait_and_collect(prompt_id, progress_cb, task_id)
        return EngineResult(file_path=str(self._final_path), engine="comfyui",
                            duration=params.duration)

    async def _wait_and_collect(self, prompt_id: str, progress_cb: ProgressCB, task_id: int):
        """Poll /history until the prompt finishes; copy output into outputs dir."""
        deadline = time.time() + 1800  # 30 min hard cap
        out_file: Optional[Path] = None
        async with httpx.AsyncClient(timeout=10.0) as c:
            while time.time() < deadline:
                r = await c.get(f"{self.base_url}/history/{prompt_id}")
                if r.status_code == 200:
                    data = r.json().get(prompt_id)
                    if data:
                        await progress_cb(60, "infer", "推理中")
                        outs = data.get("outputs", {})
                        for node in outs.values():
                            for fmeta in node.get("videos", []):
                                out_file = Path(fmeta["filename"])
                            if not out_file:
                                for fmeta in node.get("images", []):
                                    out_file = Path(fmeta["filename"])
                        if out_file:
                            await progress_cb(95, "postprocess", "落盘")
                            break
                await progress_cb(
                    min(50, int((1 - (deadline - time.time()) / 1800) * 50)),
                    "queue", "排队/推理中",
                )
                import asyncio
                await asyncio.sleep(3)

        if not out_file:
            raise RuntimeError("ComfyUI 任务超时或未完成")

        # ComfyUI writes into its own output dir; resolve via shared volume.
        src = self.outputs_dir / out_file.name
        if not src.exists():
            # fall back to a configured comfyui output location
            alt = Path("/comfyui/output") / out_file.name
            src = alt if alt.exists() else src
        dest = self.outputs_dir / f"task_{task_id}_{out_file.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(src, dest)
        self._final_path = dest
