#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax H3 — headless direct-Python runner (no ComfyUI server).

This replaces the ComfyUI graph runtime with a linear Python script. The
"surface" nodes from the original `MiniMax_H3_Easy*.json` workflows that are
*open* (model loading, conditioning, sigma-shift) are reused directly from the
vendored H3-flavored `comfy` package; the nodes that were **sealed inside
``capai.exe``** (Sampler, LoRA-apply, AttentionBackend, VAE-decode, CreateVideo,
SaveVideo) are **reconstructed here** against ComfyUI's public ``comfy.*`` APIs
using the model/VAE source that *is* available in ``comfy/ldm/minimax``.

Reconstructed (sealed-capai) stages are marked with  [#RECONSTRUCTED]。

⚠️  VALIDATION STATUS
    This script was authored on a Windows box with no GPU and no weights and
    cannot be executed here. It MUST be run on the GPU Linux instance after
    `pip install -r requirements-linux.txt` and placing weights under ./models.
    The reconstructed sampler / VAE-decode / video-mux paths are the highest
    risk and need a real run to confirm. Use `--check` first to validate that
    every import resolves in your environment.

Usage
-----
    python inference/run_minimax_h3.py --config job.json
    python inference/run_minimax_h3.py --prompt "..." --mode reference \
        --ref-image-1 a.jpg --audio-1 a.wav --output out.mp4
    python inference/run_minimax_h3.py --check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

# ---------------------------------------------------------------------------
# 0. Path bootstrap — make the vendored comfy_core importable
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
COMFY_CORE = os.path.join(REPO_ROOT, "comfy_core")
if COMFY_CORE not in sys.path:
    sys.path.insert(0, COMFY_CORE)

import torch  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402  (shim)
import node_helpers  # noqa: E402  (shim)
import nodes  # noqa: E402  (shim: MAX_RESOLUTION + loader classes)

from comfy.sd import load_lora_for_models  # noqa: E402
from comfy_extras import nodes_minimax_h3 as h3  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Import the vendored custom node (named-file load to avoid clashing with
#    our shim `nodes.py`).
# ---------------------------------------------------------------------------
_EASY_PATH = os.path.join(COMFY_CORE, "custom_nodes", "ComfyUI-MiniMaxH3-Easy", "nodes.py")
_spec = importlib.util.spec_from_file_location("minimax_h3_easy_nodes", _EASY_PATH)
easy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(easy)

MiniMaxH3EasyLoader = easy.MiniMaxH3EasyLoader
MiniMaxH3Easy = easy.MiniMaxH3Easy
MiniMaxH3EasyOutput = easy.MiniMaxH3EasyOutput
MiniMaxH3EasySecondPassConditioning = easy.MiniMaxH3EasySecondPassConditioning
MiniMaxH3ReferenceToVideo = h3.MiniMaxH3ReferenceToVideo
MiniMaxH3ImageToVideo = h3.MiniMaxH3ImageToVideo
MiniMaxH3AddGuide = h3.MiniMaxH3AddGuide
MiniMaxH3SigmaShift = h3.MiniMaxH3SigmaShift
MiniMaxH3EasyMediaBridge = easy.MiniMaxH3EasyMediaBridge

# Disable the external prompt-optimizer network call (reconstructed as a no-op
# so the run never phones home to the third-party LLM endpoint).  [#RECONSTRUCTED]
class _OptimizationStub(types.SimpleNamespace):
    marker = None

def _noop_optimize(prompt, *a, **k):
    return _OptimizationStub(prompt=prompt)

easy._optimize_prompt_on_run = _noop_optimize


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def unwrap(out):
    """Extract a value from an ``io.NodeOutput`` (or pass through)."""
    if isinstance(out, dict) and "result" in out:
        return out["result"]
    if hasattr(out, "args"):  # io.NodeOutput
        a = out.args
        return a[0] if len(a) == 1 else a
    return out


def unwrap_vae(vae):
    """Get the raw nn.Module (MiniMaxH3VideoVAE / MiniMaxH3AudioVAE) from a
    comfy VAE patcher object."""
    for attr in ("first_stage_model",):
        if hasattr(vae, attr) and getattr(vae, attr) is not None:
            return getattr(vae, attr)
    if hasattr(vae, "model") and hasattr(vae.model, "first_stage_model"):
        return vae.model.first_stage_model
    return vae


def set_attention_backend(backend: str):
    """[#RECONSTRUCTED] Select the attention implementation.

    The sealed ``MiniMaxH3EasyAttentionBackend`` node mapped a dropdown to a
    comfy attention function. We apply the same global switch comfy uses.
    backend: 'torch' | 'sageattention' | 'xformers'
    """
    import comfy.ldm.modules.attention as attn
    backend = (backend or "torch").lower()
    if backend == "sageattention" and hasattr(attn, "sage_attention"):
        attn.attention_function = attn.sage_attention
    elif backend == "xformers" and hasattr(attn, "xformers_attention"):
        attn.attention_function = attn.xformers_attention
    else:
        attn.attention_function = attn.default_attention
    print(f"[attention] backend = {backend}")


# ---------------------------------------------------------------------------
# 3. Load models (reuses open MiniMaxH3EasyLoader)
# ---------------------------------------------------------------------------
def build_bundle(cfg):
    # NOTE: MiniMaxH3EasyLoader.load is a *regular* instance method that returns
    # a ComfyUI-style single-element tuple ``(MiniMaxH3Bundle(...),)``.
    bundle = MiniMaxH3EasyLoader().load(
        cfg["fl2va_model"], cfg["ref2va_model"], cfg["text_encoder"],
        cfg["video_vae"], cfg["audio_vae"],
    )[0]
    return bundle


# ---------------------------------------------------------------------------
# 4. Conditioning + latent (reuses open MiniMaxH3Easy.generate / Output.unpack)
# ---------------------------------------------------------------------------
def prepare_conditioning(bundle, cfg, media=None):
    kwargs = {}
    if media is not None:
        kwargs["media"] = media  # image tensor / video components / audio dict
    model, context = unwrap(MiniMaxH3Easy.generate(
        bundle,
        cfg["mode"], cfg["prompt"], cfg["resolution"], cfg["aspect_ratio"],
        cfg["width"], cfg["height"], cfg["seconds"], cfg.get("advanced", False),
        cfg["fps"], cfg["keyframe_role"], cfg["ref_image_size"],
        cfg.get("reference_mention_mode", "index"),
        **kwargs,
    ))
    positive, latent, video_vae, audio_vae, fps = unwrap(MiniMaxH3EasyOutput.unpack(context))
    return model, positive, latent, video_vae, audio_vae, fps


# ---------------------------------------------------------------------------
# 5. [#RECONSTRUCTED] LoRA apply (sealed MiniMaxH3EasyLoRAApply)
# ---------------------------------------------------------------------------
def apply_lora(model, clip, lora_name, strength):
    if not lora_name or str(lora_name).lower() in ("none", "null", ""):
        return model, clip
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        lora_path = lora_name  # maybe an absolute path
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA not found: {lora_name}")
    model, clip = load_lora_for_models(model, clip, lora_path, strength, 0.0)
    print(f"[lora] applied {lora_name} (strength={strength})")
    return model, clip


# ---------------------------------------------------------------------------
# 6. [#RECONSTRUCTED] Sampler (sealed MiniMaxH3EasySampler)
# ---------------------------------------------------------------------------
def sample_h3(model, positive, latent, cfg):
    # 6a. attach ModelSamplingAV (MiniMaxH3SigmaShift is an OPEN node)
    model = unwrap(MiniMaxH3SigmaShift.execute(
        model, cfg["shift_video"], cfg["shift_audio"]))

    # 6b. build sigmas from scheduler + steps
    ms = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, cfg["scheduler"], cfg["steps"])

    device = mm.get_torch_device()
    latent_samples = latent["samples"]  # NestedTensor (video, audio)
    noise = comfy.sample.prepare_noise(latent_samples, cfg["seed"])

    # H3 is a single-conditioning flow model -> CFG=1 disables the negative
    # term, so a copy of positive as negative is harmless.
    negative = positive
    cfg_val = float(cfg.get("cfg", 1.0))

    samples = comfy.samplers.sample(
        model, noise, positive, negative, cfg_val, device,
        cfg["sampler"], sigmas,
        latent_image=latent_samples, seed=cfg["seed"],
        model_options=model.model_options,
    )
    return samples  # AV latent (NestedTensor)


# ---------------------------------------------------------------------------
# 7. [#RECONSTRUCTED] VAE decode (sealed MiniMaxH3EasyVAEDecode)
# ---------------------------------------------------------------------------
def decode_av(video_vae, audio_vae, av_samples):
    import torch

    if isinstance(av_samples, dict):
        av_samples = av_samples["samples"]
    if not hasattr(av_samples, "tensors") or len(av_samples.tensors) != 2:
        raise RuntimeError(
            "AV latent is not a 2-component NestedTensor. The reconstructed "
            "`comfy.samplers.sample` path likely collapsed the audio/video "
            "latents. Inspect the returned latent shape and compare against "
            "MiniMaxH3VideoVAE/ MiniMaxH3AudioVAE expected inputs "
            "([B,24,T,H,W] and [B,32,2,T])."
        )
    video_lat = av_samples.tensors[0]
    audio_lat = av_samples.tensors[1]

    vmodel = unwrap_vae(video_vae)
    amodel = unwrap_vae(audio_vae)

    with torch.no_grad():
        # MiniMaxH3VideoVAE.decode: [B,24,T_lat,H,W] -> [B,3,T,H,W] in [0,1]
        frames = vmodel.decode(video_lat)
        # MiniMaxH3AudioVAE.decode: [B,32,2,T] -> [B,2,L] waveform
        wav = amodel.decode(audio_lat)

    frames = frames[0].permute(1, 2, 3, 0).clamp(0.0, 1.0)  # [T,H,W,3]
    wav = wav[0]  # [2, L]
    return frames, wav


# ---------------------------------------------------------------------------
# 8. [#RECONSTRUCTED] CreateVideo + SaveVideo (sealed MiniMaxH3Easy*Video/Save)
# ---------------------------------------------------------------------------
def save_video(frames, wav, fps, out_path, audio_sr=32000):
    import numpy as np
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import subprocess

    frames_np = (frames.detach().cpu().float().numpy() * 255).astype("uint8")
    if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
        pass  # [T,H,W,3]
    else:
        raise ValueError(f"unexpected frames shape {frames_np.shape}")

    tmp_vid = out_path + ".novideo.mp4"
    writer = imageio.get_writer(
        tmp_vid, fps=fps, codec="libx264", macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p"],
    )
    for f in frames_np:
        writer.append_data(f)
    writer.close()

    if wav is not None:
        import torchaudio
        tmp_wav = out_path + ".audio.wav"
        wav_np = wav.detach().cpu().float()
        torchaudio.save(tmp_wav, wav_np, audio_sr)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-y", "-i", tmp_vid, "-i", tmp_wav,
             "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
            check=True,
        )
        os.remove(tmp_vid)
        os.remove(tmp_wav)
    else:
        os.replace(tmp_vid, out_path)
    print(f"[save] wrote {out_path}")


# ---------------------------------------------------------------------------
# 9. Single-pass orchestration
# ---------------------------------------------------------------------------
def run_single_pass(cfg):
    bundle = build_bundle(cfg)
    model, positive, latent, video_vae, audio_vae, fps = prepare_conditioning(bundle, cfg)
    if cfg.get("lora"):
        model, _ = apply_lora(model, bundle.clip, cfg["lora"], cfg.get("lora_strength", 1.0))
    set_attention_backend(cfg.get("attention", "torch"))
    samples = sample_h3(model, positive, latent, cfg)
    frames, wav = decode_av(video_vae, audio_vae, samples)
    save_video(frames, wav, fps, cfg["output"])
    return cfg["output"]


# ---------------------------------------------------------------------------
# 10. [#RECONSTRUCTED] Pass2 dual-stage (sealed MiniMax_H3_Easy_Pass2 workflow)
# ---------------------------------------------------------------------------
def run_pass2(cfg, cfg2):
    # --- pass 1: low-resolution base generation ---
    bundle = build_bundle(cfg)
    m1, p1, lat1, vvae1, avae1, fps = prepare_conditioning(bundle, cfg)
    if cfg.get("lora"):
        m1, _ = apply_lora(m1, bundle.clip, cfg["lora"], cfg.get("lora_strength", 1.0))
    set_attention_backend(cfg.get("attention", "torch"))
    s1 = sample_h3(m1, p1, lat1, cfg)
    frames1, wav1 = decode_av(vvae1, avae1, s1)  # base video + audio

    # decode the FIRST-pass video latent back to a latent at the target
    # resolution so SecondPassConditioning can rebuild keyframe conditioning.
    # The empty target-resolution AV latent is produced by the open
    # EmptyMiniMaxH3LatentAV-style path inside MiniMaxH3Easy.generate already;
    # we rebuild conditioning at that resolution here.
    _, _, lat2, vvae2, avae2, _ = prepare_conditioning(bundle, cfg2)
    # NOTE: MiniMaxH3EasySecondPassConditioning.rebuild is a classmethod that
    # returns a plain ``(conditioning,)`` tuple (not a NodeOutput), so index [0].
    second_pass_positive = MiniMaxH3EasySecondPassConditioning.rebuild(
        _context_of(cfg2, bundle), lat2)[0]

    # load the REF2VA W4A8 model for the upscaled second pass
    bundle2 = build_bundle(cfg2)
    m2, _, lat2b, vvae2b, avae2b, fps2 = prepare_conditioning(bundle2, cfg2)
    set_attention_backend(cfg2.get("attention", "torch"))
    s2 = sample_h3(m2, second_pass_positive, lat2b, cfg2)
    frames2, wav2 = decode_av(vvae2b, avae2b, s2)

    # upscale to 1920x1088 (the Pass2 target) — nearest simple bilinear.
    frames2 = _upscale_frames(frames2, cfg2.get("target_height", 1088), cfg2.get("target_width", 1920))
    save_video(frames2, wav2, fps2, cfg2["output"])
    return cfg2["output"]


def _context_of(cfg, bundle):
    """Rebuild the MiniMaxH3Context object the second-pass node expects."""
    _, context = unwrap(MiniMaxH3Easy.generate(
        bundle, cfg["mode"], cfg["prompt"], cfg["resolution"], cfg["aspect_ratio"],
        cfg["width"], cfg["height"], cfg["seconds"], cfg.get("advanced", False),
        cfg["fps"], cfg["keyframe_role"], cfg["ref_image_size"],
        cfg.get("reference_mention_mode", "index"),
    ))
    return context


def _upscale_frames(frames, h, w):
    import torch.nn.functional as F
    # frames: [T,H,W,3] -> [T,3,h,w] -> [T,h,w,3]
    x = frames.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 11. Config loading / CLI
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "mode": "reference",            # image | reference
    "prompt": "",
    "resolution": "360P",
    "aspect_ratio": "16:9",
    "width": 1376, "height": 768,
    "seconds": 10.0, "fps": 24,
    "advanced": False,
    "keyframe_role": "first",
    "ref_image_size": "1k",
    "reference_mention_mode": "index",
    "steps": 8, "cfg": 1.0,
    "sampler": "euler", "scheduler": "simple",
    "shift_video": 12.0, "shift_audio": 3.0,
    "seed": -1,
    "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "lora_strength": 1.0,
    "attention": "torch",
    # model filenames (resolved under ./models via folder_paths)
    "fl2va_model": "minimax_h3_fl2v_preview.safetensors",
    "ref2va_model": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    "text_encoder": "minimax_h3_text_encoder_fp16.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "output": "output.mp4",
}


def load_cfg(args):
    cfg = dict(DEFAULT_CFG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for k in ("prompt", "mode", "resolution", "aspect_ratio", "output",
              "sampler", "scheduler", "lora", "attention", "ref_image_size",
              "keyframe_role", "reference_mention_mode"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    for k in ("width", "height", "steps", "fps", "seed"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if cfg["seed"] is None or cfg["seed"] < 0:
        cfg["seed"] = int(torch.randint(0, 2**31 - 1, (1,)).item())
    return cfg


def main():
    ap = argparse.ArgumentParser(description="MiniMax H3 headless runner")
    ap.add_argument("--config", help="JSON job config (overrides defaults)")
    ap.add_argument("--prompt", help="generation prompt")
    ap.add_argument("--mode", choices=["image", "reference"], default=None)
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--aspect-ratio", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--scheduler", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--attention", default=None)
    ap.add_argument("--ref-image-1", default=None, help="reference image path")
    ap.add_argument("--audio-1", default=None, help="reference audio path")
    ap.add_argument("--output", default=None)
    ap.add_argument("--pass2-config", default=None, help="second-pass JSON config")
    ap.add_argument("--check", action="store_true",
                    help="only validate that all imports resolve, then exit")
    args = ap.parse_args()

    folder_paths.init_folders()

    if args.check:
        print("[check] all imports resolved OK")
        print(f"[check] comfy version path: {os.path.join(COMFY_CORE, 'comfy')}")
        print(f"[check] models dir: {folder_paths.models_dir}")
        return

    cfg = load_cfg(args)
    if args.pass2_config:
        cfg2 = dict(DEFAULT_CFG)
        with open(args.pass2_config, "r", encoding="utf-8") as f:
            cfg2.update(json.load(f))
        out = run_pass2(cfg, cfg2)
    else:
        out = run_single_pass(cfg)
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
