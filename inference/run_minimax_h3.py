#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax-H3 头less 直接推理运行器（不依赖 ComfyUI server / 节点图 / 自定义节点管理器）。

把 ComfyUI 节点图拍平为线性管线（见 FLATTENED_WORKFLOWS.md），直接调用 ComfyUI 内核里
`comfy_extras.nodes_minimax_h3` 的节点 `execute()`。对应原工作流：
  - MiniMax_H3_Easy.json        -> mode=reference / image（单遍）
  - MiniMax_H3_Easy_Pass2.json  -> mode=dual_stage（双阶段放大/精修）

前置条件（详见 LINUX_PORT.md）：
  1) pip install -r requirements-linux.txt
  2) 把 ComfyUI 内核 vendored 到 ./comfy_core（comfy / comfy_api / comfy_extras/nodes_minimax_h3
     / comfy/ldm/minimax* / transformers/models/minimax），并加入 sys.path
  3) 权重放到 models/ 对应子目录（FLATTENED_WORKFLOWS.md 第四节）

本文件是“拍平后可直接调用的蓝图”。凡涉及 comfy 内核内部 API 的精确接线（model 句柄构造、
sampler 调度、VAE 解码、CreateVideo）需结合 vendored 源码完成；标注 [NEEDS_COMFY_CORE] 处即待接线点。
脚本层面（参数解析、尺寸对齐、文件落盘）已可直接用。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---- vendored ComfyUI 内核（大小写/路径按实际目录调整） ----
COMFY_CORE = Path(__file__).resolve().parent / "comfy_core"
if COMFY_CORE.exists():
    sys.path.insert(0, str(COMFY_CORE))

# 模型默认文件名（来自两个工作流 JSON 的 widgets_values，未做臆测）
MODELS = {
    "fl2va": "minimax_h3_fl2va_int8_convrot.safetensors",
    "ref2va_w4a8": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",   # 仅双阶段第二阶
    "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}
LORA_STRENGTH = 1.0
FPS = 24
ATTENTION_BACKEND = "comfy kitchen attention"  # comfy_kitchen 包提供


# =============================================================================
# 模型加载  [NEEDS_COMFY_CORE]
# =============================================================================
def load_models(models_dir: Path, use_ref2va: bool = False):
    """加载 FL2VA / (可选)REF2VA / 文本编码器 / 视频VAE / 音频VAE。

    对应工作流 S0 (MiniMaxH3EasyLoader)。实际应通过 comfy 的模型加载器加载 .safetensors，
    返回可直接喂给 nodes_minimax_h3 节点的句柄。
    """
    # [NEEDS_COMFY_CORE]: 用 comfy.model_management / comfy.utils / folder_paths 加载
    #   fl2va -> model
    #   text_encoder -> clip
    #   video_vae / audio_vae -> vae 句柄
    #   use_ref2va=True 时额外加载 ref2va_w4a8（双阶段第二阶）
    raise NotImplementedError("load_models: 需结合 vendored comfy 源码完成模型加载接线")


# =============================================================================
# LoRA + 注意力后端  [NEEDS_COMFY_CORE]
# =============================================================================
def apply_lora(model, lora_name: str, strength: float = LORA_STRENGTH):
    """对应工作流 S3 (LoraLoaderModelOnly)。"""
    # [NEEDS_COMFY_CORE]: comfy 的 LoRA 应用（model_only）
    raise NotImplementedError("apply_lora")


def set_attention_backend(model, backend: str = ATTENTION_BACKEND):
    """对应工作流 S4 (ModelAttentionBackend)。"""
    # [NEEDS_COMFY_CORE]: comfy_kitchen 的注意力后端替换
    raise NotImplementedError("set_attention_backend")


# =============================================================================
# 单遍：参考生视频 / 图生视频
# 对应工作流 S2 (MiniMaxH3Easy -> MiniMaxH3ReferenceToVideo / MiniMaxH3ImageToVideo)
# =============================================================================
def generate_single_pass(model_bundle, *, mode: str, prompt: str,
                         image=None, width: int, height: int, length: int,
                         seed: int, steps: int = 8, fps: int = FPS):
    """拍平管线 S2 + S5..S10。

    直接调用 comfy_extras.nodes_minimax_h3 的节点 execute()：
      - mode=='reference' -> MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, length, ref_image_size=...)
      - mode=='image'     -> MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length, ...)
    之后用标准 comfy 节点完成采样/解码/组装（见下）。
    """
    from comfy_extras import nodes_minimax_h3 as h3  # 真实推理入口

    clip = model_bundle["clip"]
    vae = model_bundle["video_vae"]
    audio_vae = model_bundle["audio_vae"]
    model = model_bundle["model"]

    if mode == "reference":
        h3_context = h3.MiniMaxH3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt, width, height, length,
            ref_image_size="match",  # 工作流 S2 scale=原图（不缩放）
            # 首帧优先 / 按序号 / 仅通用方案 等由节点内部按 h3_context 处理
        )
    elif mode == "image":
        h3_context = h3.MiniMaxH3ImageToVideo.execute(
            clip, vae, prompt, width, height, length,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    # --- S5 输出准备：从 h3_context 取 positive / latent / vae / fps ---
    positive, latent, latent_vae, latent_audio_vae, fps = unpack_h3_output(h3_context)

    # --- S6 采样器配置（euler / simple / 8步） ---
    noise = make_noise(seed)                       # RandomNoise
    sampler = select_sampler("euler")              # KSamplerSelect
    sigmas = basic_scheduler(model, "simple", steps, denoise=1.0)  # BasicScheduler
    guider = basic_guider(model, positive)        # BasicGuider

    # --- S7 采样（SamplerCustomAdvanced） ---
    out_latent, denoised = sample_advanced(noise, guider, sampler, sigmas, latent)

    # --- S8 解码 ---
    frames = vae_decode(out_latent, latent_vae)            # VAEDecode (video)
    audio = vae_decode_audio(denoised, latent_audio_vae)    # VAEDecodeAudio

    # --- S9 组装视频 ---
    video = create_video(frames, audio, fps)               # CreateVideo
    return video, out_latent, denoised  # 双阶段需复用 denoised 里的 audio latent


# =============================================================================
# 双阶段：放大 / 精修（对应 MiniMax_H3_Easy_Pass2.json）
# =============================================================================
def generate_dual_stage(model_bundle, *, prompt: str, image,
                        first_w=1344, first_h=768, first_len=4,
                        final_w=1920, final_h=1088,
                        seed: int, first_steps=8, second_steps=3, fps=FPS):
    """拍平管线 阶段A（低分辨率）+ 阶段B（放大精修）。"""
    # ---- 阶段 A：第一遍 ----
    video_first, lat_first, den_first = generate_single_pass(
        model_bundle, mode="image", prompt=prompt, image=image,
        width=first_w, height=first_h, length=first_len, seed=seed, steps=first_steps,
    )
    # 拆分第一遍 AV latent -> 复用 audio_latent（LTXVSeparateAVLatent）
    video_latent_a, audio_latent_a = separate_av_latent(lat_first)  # [NEEDS_COMFY_CORE]

    # ---- 阶段 B：第二遍 ----
    # B.1 ResolutionSelector -> 1920x1088；ImageResizeKJv2 放大第一遍帧
    frames_up = resize_frames(video_first, final_w, final_h, method="lanczos", divisible_by=32)
    # B.2 VAEEncode 重新编码 -> second_pass_video_latent
    sp_video_latent = vae_encode(frames_up, model_bundle["video_vae"])  # [NEEDS_COMFY_CORE]
    # B.3 UNETLoader 加载 REF2VA W4A8（第二阶模型）
    model2 = load_models(Path("models"), use_ref2va=True)["model"]
    model2 = apply_lora(model2, MODELS["lora"], LORA_STRENGTH)
    model2 = set_attention_backend(model2)
    # B.4 二阶段条件（MiniMaxH3EasySecondPassConditioning）
    sp_positive = second_pass_conditioning(model_bundle["h3_context"], sp_video_latent)  # [NEEDS_COMFY_CORE]
    # B.5 Beta scheduler 3步 denoise 0.25
    sigmas2 = basic_scheduler(model2, "beta", second_steps, denoise=0.25)
    guider2 = basic_guider(model2, sp_positive)
    # B.6 拼接 AV latent（复用 audio_latent_a）
    av_latent = concat_av_latent(sp_video_latent, audio_latent_a)  # LTXVConcatAVLatent [NEEDS_COMFY_CORE]
    # B.7 第二遍采样
    out2, _ = sample_advanced(make_noise(seed), guider2, select_sampler("euler"), sigmas2, av_latent)
    # B.8 解码 + 组装
    frames2 = vae_decode(out2, model_bundle["video_vae"])
    video_final = create_video(frames2, None, fps)  # 音频已内嵌于 av_latent
    return video_final


# =============================================================================
# 标准 comfy 节点等价辅助  [NEEDS_COMFY_CORE]（接线点）
# =============================================================================
def unpack_h3_output(h3_context):            # MiniMaxH3EasyOutput
    raise NotImplementedError("unpack_h3_output")
def make_noise(seed):                        # RandomNoise
    raise NotImplementedError("make_noise")
def select_sampler(name):                    # KSamplerSelect
    raise NotImplementedError("select_sampler")
def basic_scheduler(model, name, steps, denoise):  # BasicScheduler
    raise NotImplementedError("basic_scheduler")
def basic_guider(model, positive):           # BasicGuider
    raise NotImplementedError("basic_guider")
def sample_advanced(noise, guider, sampler, sigmas, latent):  # SamplerCustomAdvanced
    raise NotImplementedError("sample_advanced")
def vae_decode(latent, vae):                 # VAEDecode
    raise NotImplementedError("vae_decode")
def vae_decode_audio(latent, audio_vae):    # VAEDecodeAudio
    raise NotImplementedError("vae_decode_audio")
def vae_encode(frames, vae):                 # VAEEncode
    raise NotImplementedError("vae_encode")
def separate_av_latent(latent):              # LTXVSeparateAVLatent
    raise NotImplementedError("separate_av_latent")
def concat_av_latent(video_latent, audio_latent):  # LTXVConcatAVLatent
    raise NotImplementedError("concat_av_latent")
def second_pass_conditioning(h3_context, video_latent):  # MiniMaxH3EasySecondPassConditioning
    raise NotImplementedError("second_pass_conditioning")
def create_video(frames, audio, fps):        # CreateVideo (VHS)
    raise NotImplementedError("create_video")
def resize_frames(frames, w, h, method, divisible_by):  # ImageResizeKJv2
    raise NotImplementedError("resize_frames")


# =============================================================================
# 落盘
# =============================================================================
def save_video(video, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # [NEEDS_COMFY_CORE]: 用 comfy_extras.nodes_vhs 的 SaveVideo 或直接 imageio/av 写 mp4
    raise NotImplementedError("save_video")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 头less 直接推理")
    p.add_argument("--mode", choices=["reference", "image", "dual_stage"], required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--image", default=None, help="参考图/首帧 (png/jpg)")
    p.add_argument("--audio", default=None, help="可选独立音频")
    p.add_argument("--models-dir", default="models", type=Path)
    p.add_argument("--output", default="output/MiniMaxH3_Easy.mp4", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=1344)
    p.add_argument("--length", type=int, default=5, help="潜在帧数")
    p.add_argument("--fps", type=int, default=FPS)
    return p.parse_args()


def main():
    args = parse_args()
    if not COMFY_CORE.exists():
        sys.exit(f"[错误] 未找到 vendored ComfyUI 内核：{COMFY_CORE}\n"
                 f"请按 LINUX_PORT.md §3 把 comfy / comfy_api / comfy_extras/nodes_minimax_h3 放入 comfy_core/")

    bundle = load_models(args.models_dir, use_ref2va=(args.mode == "dual_stage"))

    if args.mode == "dual_stage":
        video = generate_dual_stage(
            bundle, prompt=args.prompt, image=args.image,
            seed=args.seed, first_steps=args.steps, final_w=args.width, final_h=args.height,
        )
    else:
        video, _, _ = generate_single_pass(
            bundle, mode=args.mode, prompt=args.prompt, image=args.image,
            width=args.width, height=args.height, length=args.length,
            seed=args.seed, steps=args.steps, fps=args.fps,
        )
    save_video(video, args.output)
    print(f"[完成] 视频已保存：{args.output}")


if __name__ == "__main__":
    main()
