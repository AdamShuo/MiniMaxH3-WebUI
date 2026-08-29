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


def build_engines(**kw) -> dict[str, BaseEngine]:
    engines: dict[str, BaseEngine] = {}
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
                   has_comfyui: bool, has_minimax: bool) -> str:
    if force_engine == "minimax":
        if not has_minimax:
            raise RuntimeError("强制 MiniMax 兜底，但未配置 MINIMAX_API_KEY (BD-03)")
        return "minimax"
    if force_engine == "comfyui":
        if has_comfyui:
            return "comfyui"
        if use_fallback and has_minimax:
            return "minimax"
        raise RuntimeError("强制本地 ComfyUI，但本地引擎不可用 (BD-03)")
    # auto
    if has_comfyui:
        return "comfyui"
    if use_fallback and has_minimax:
        return "minimax"
    raise RuntimeError("本地引擎不可用且兜底未开启 (BD-03)")
