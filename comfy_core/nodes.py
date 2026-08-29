"""Minimal ``nodes`` shim for headless MiniMax H3 execution.

The real ComfyUI app-root ``nodes.py`` (which holds UNETLoader / CLIPLoader /
VAELoader / LoraLoader / KSampler / MAX_RESOLUTION ...) is sealed inside
``capai.exe``. The vendored H3 nodes only need:

* ``MAX_RESOLUTION``
* ``UNETLoader.load_unet(name, dtype)``
* ``CLIPLoader.load_clip(name, clip_type, device)``
* ``VAELoader.load_vae(name)``

These delegate to the vendored ``comfy.sd`` loaders (the same functions the
real nodes call internally).
"""
import os

import folder_paths
import comfy.sd
from comfy.sd import CLIPType

MAX_RESOLUTION = 16384


def _resolve(category, name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return folder_paths.get_full_path(category, name)


class UNETLoader:
    @classmethod
    def load_unet(cls, unet_name, weight_dtype="default"):
        path = _resolve("unet", unet_name) or _resolve("diffusion_models", unet_name)
        if path is None:
            raise FileNotFoundError(f"UNET model not found: {unet_name}")
        model = comfy.sd.load_unet(path)
        return (model,)


class CLIPLoader:
    @classmethod
    def load_clip(cls, clip_name, clip_type="stable_diffusion", device="default"):
        path = _resolve("clip", clip_name)
        if path is None:
            raise FileNotFoundError(f"CLIP/text-encoder model not found: {clip_name}")
        ct = getattr(CLIPType, str(clip_type).upper(), CLIPType.STABLE_DIFFUSION)
        clip = comfy.sd.load_clip([path], clip_type=ct)
        return (clip,)


class VAELoader:
    @classmethod
    def load_vae(cls, vae_name):
        path = _resolve("vae", vae_name)
        if path is None:
            raise FileNotFoundError(f"VAE model not found: {vae_name}")
        vae = comfy.sd.load_vae_patcher(path)
        return (vae,)
