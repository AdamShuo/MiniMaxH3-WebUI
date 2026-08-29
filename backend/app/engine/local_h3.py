"""B4 — 本地 H3 直跑引擎：子进程调用 inference/run_minimax_h3.py。

取代原 ComfyUI server 适配（comfyui.py）。不依赖任何外部 ComfyUI 服务，直接
复用 vendored ``comfy_core`` + 本地权重在 GPU 上推理。采样 / VAE 解码 / 双阶段 /
参考素材张量化等路径在 runner 中标记 [#RECONSTRUCTED]，需在真实 GPU 上验证。

工作方式：worker 把本次任务的参数、参考图/音频/视频路径、LoRA 列表写成一份
job JSON，再 ``subprocess`` 拉起 ``run_minimax_h3.py --config job.json``。每个
任务独立子进程，模型权重随进程加载/释放（实现简单、隔离性好；代价是每次生成
都重新加载权重，如需常驻可后续改为进程内调用）。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseEngine, EngineResult, GenParams, ProgressCB
from ..config import settings


class LocalH3Engine(BaseEngine):
    name = "local_h3"

    def __init__(self, runner_script: str, comfy_core_dir: str,
                 inference_python: str, outputs_dir: str,
                 attention_backend: str = "torch"):
        self.runner_script = runner_script
        self.comfy_core_dir = comfy_core_dir
        self.inference_python = inference_python
        self.outputs_dir = Path(outputs_dir)
        self.attention_backend = attention_backend

    async def health(self) -> bool:
        if not os.path.isfile(self.runner_script):
            return False
        if not os.path.isdir(self.comfy_core_dir):
            return False
        # 仅确认推理 Python 可用（不做 torch import，避免拖慢启动）。
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [self.inference_python, "-c", "import sys; sys.exit(0)"],
                capture_output=True, timeout=30,
            )
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _build_job(self, *, prompt: str, params: GenParams,
                   reference_paths: list[str], audio_paths: list[str],
                   video_paths: list[str], task_id: int):
        out_path = str(self.outputs_dir / f"task_{task_id}.mp4")
        job = {
            "mode": params.mode or "reference",
            "prompt": prompt,
            "resolution": params.first_stage_resolution or "360P",
            "aspect_ratio": "16:9",
            "width": params.width, "height": params.height,
            "seconds": float(params.duration), "fps": params.fps,
            "advanced": False, "keyframe_role": "first",
            "ref_image_size": "1k", "reference_mention_mode": "index",
            "steps": params.step, "cfg": 1.0,
            "sampler": "euler", "scheduler": "simple",
            "shift_video": 12.0, "shift_audio": 3.0,
            "seed": params.seed if params.seed and params.seed > 0 else -1,
            "attention": self.attention_backend,
            "ref_images": reference_paths or [],
            "audios": audio_paths or [],
            "videos": video_paths or [],
            "loras": params.loras or [],
            "output": out_path,
            # 模型文件名（与 run_minimax_h3.DEFAULT_CFG 对齐；云端权重置于 MODELS_DIR）
            "fl2va_model": "minimax_h3_fl2v_preview.safetensors",
            "ref2va_model": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
            "text_encoder": "minimax_h3_text_encoder_fp16.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        }
        return job, out_path

    # ------------------------------------------------------------------
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
        job, out_path = self._build_job(
            prompt=prompt, params=params, reference_paths=reference_paths,
            audio_paths=audio_paths, video_paths=video_paths or [], task_id=task_id,
        )
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, dir=str(self.outputs_dir))
        json.dump(job, tmp, ensure_ascii=False)
        tmp.close()
        tmp2_path = None
        try:
            cmd = [self.inference_python, self.runner_script,
                   "--config", tmp.name]
            if params.mode == "dual_stage":
                # 二遍放大：第二遍用更高分辨率 + 目标 1920x1088
                job2 = dict(job)
                job2["width"] = 1920
                job2["height"] = 1088
                t2 = tempfile.NamedTemporaryFile(
                    "w", suffix=".json", delete=False, dir=str(self.outputs_dir))
                json.dump(job2, t2)
                t2.close()
                tmp2_path = t2.name
                cmd += ["--pass2-config", tmp2_path]

            await progress_cb(5, "queued", "已写入任务配置，启动本地推理子进程")

            env = {
                **os.environ,
                "PYTHONPATH": self.comfy_core_dir,
                "MODELS_DIR": settings.models_dir,
            }
            proc = await asyncio.to_thread(
                subprocess.run, cmd, cwd=self.comfy_core_dir,
                capture_output=True, text=True, timeout=3600, env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"H3 runner 失败 (exit={proc.returncode}):\n"
                    f"{proc.stderr[-3000:]}"
                )
            await progress_cb(95, "postprocess", "推理完成，落盘")
            return EngineResult(
                file_path=out_path, engine="local_h3", duration=params.duration)
        finally:
            for p in (tmp.name, tmp2_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
