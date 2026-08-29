"""Engine routing — 系统设计 §3.2.M4 路由决策 + 失败切换兜底 (BD-02/BD-03).

Priority (matches the documented state machine):
  force_engine=minimax                  -> MiniMax
  force_engine=comfyui                  -> ComfyUI (fail -> BD-03 unless use_fallback)
  auto:
    ComfyUI healthy & not force_fallback -> ComfyUI
    else if use_fallback & MiniMax ready -> MiniMax   (BD-02)
    else                                   -> raise (BD-03)
"""
from __future__ import annotations

from .base import BaseEngine
from .comfyui import ComfyUIEngine
from .minimax import MiniMaxEngine
from .local_h3 import LocalH3Engine


def build_engines(**kw) -> dict[str, BaseEngine]:
    engines: dict[str, BaseEngine] = {}
    # 本地 H3 直跑引擎（默认开启，取代 ComfyUI server 依赖）
    if kw.get("local_h3_enabled"):
        engines["local_h3"] = LocalH3Engine(
            runner_script=kw["runner_script"],
            comfy_core_dir=kw["comfy_core_dir"],
            inference_python=kw["inference_python"],
            outputs_dir=kw["outputs_dir"],
            attention_backend=kw.get("h3_attention_backend", "torch"),
        )
    if kw.get("comfyui_url"):
        engines["comfyui"] = ComfyUIEngine(
            base_url=kw["comfyui_url"],
            workflow_path=kw["workflow_path"],
            lora_dir=kw["models_dir"],
            outputs_dir=kw["outputs_dir"],
        )
    if kw.get("minimax_api_key"):
        engines["minimax"] = MiniMaxEngine(
            base_url=kw["minimax_base_url"],
            api_key=kw["minimax_api_key"],
            model=kw["minimax_model"],
            outputs_dir=kw["outputs_dir"],
        )
    return engines


def decide_primary(force_engine: str | None, use_fallback: bool,
                   has_comfyui: bool, has_minimax: bool,
                   has_local_h3: bool = False) -> str:
    if force_engine == "minimax":
        if not has_minimax:
            raise RuntimeError("强制 MiniMax 兜底，但未配置 MINIMAX_API_KEY (BD-03)")
        return "minimax"
    if force_engine == "local_h3":
        if has_local_h3:
            return "local_h3"
        if use_fallback and has_minimax:
            return "minimax"
        raise RuntimeError("强制本地 H3，但本地直跑引擎不可用 (BD-03)")
    if force_engine == "comfyui":
        if has_comfyui:
            return "comfyui"
        if use_fallback and has_minimax:
            return "minimax"
        raise RuntimeError("强制本地 ComfyUI，但本地引擎不可用 (BD-03)")
    # auto：优先本地 H3 直跑，其次 ComfyUI，再次 MiniMax 兜底
    if has_local_h3:
        return "local_h3"
    if has_comfyui:
        return "comfyui"
    if use_fallback and has_minimax:
        return "minimax"
    raise RuntimeError("本地引擎不可用且兜底未开启 (BD-03)")
