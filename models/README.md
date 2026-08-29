# 模型权重目录 `models/`

本地 H3 直跑引擎（`LocalH3Engine` → `inference/run_minimax_h3.py`）通过
`comfy_core/folder_paths.py` 在 `$MODELS_DIR`（容器内为 `/models`，由 `docker-compose.yml`
把仓库根的 `./models` **只读**挂载）下，按 **ComfyUI 分类子目录** 查找权重。

**权重必须放进对应子目录，且文件名需与本文件完全一致**，否则引擎找不到文件、会静默回退到
MiniMax 官方兜底 API（参考图/音频将被发往新加坡，数据出境），或直接报错。

---

## 目录结构（请把权重放进对应子目录）

```
MiniMaxH3-WebUI/
└── models/                              # docker-compose 挂载进容器 /models:ro
    ├── clip/                            # 文本编码器 (CLIPLoader, type=minimax)
    │   └── minimax_h3_text_encoder_fp16.safetensors
    ├── vae/                             # 视频 / 音频 VAE (VAELoader)
    │   ├── minimax_h3_video_vae_fp16.safetensors
    │   └── minimax_h3_audio_vae_fp32.safetensors
    ├── diffusion_models/                # 主 / 参考 transformer（unet/ 也等价）
    │   ├── minimax_h3_fl2v_preview.safetensors
    │   └── minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
    └── loras/                           # LoRA（文件名可用 .env 的 H3_LORA_NAME 覆盖）
        └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

> 注：`diffusion_models/` 与 `unet/` 在 `folder_paths.py` 中映射到同一对目录，
> 两个名字都可放；本仓库约定用 `diffusion_models/`。

---

## 权重文件名 → 分类映射（来自 `inference/run_minimax_h3.py` 的 `DEFAULT_CFG`）

| 配置键 | 文件名 | 子目录（folder_paths 分类） |
|---|---|---|
| `text_encoder` | `minimax_h3_text_encoder_fp16.safetensors` | `models/clip/` |
| `video_vae` | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| `audio_vae` | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| `fl2va_model` | `minimax_h3_fl2v_preview.safetensors` | `models/diffusion_models/`（或 `unet/`） |
| `ref2va_model` | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | `models/diffusion_models/`（或 `unet/`） |
| `lora` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/loras/` |

---

## 重要说明

1. **核心 5 个文件名是锁死的**：`text_encoder` / `video_vae` / `audio_vae` / `fl2va_model` /
   `ref2va_model` 没有 CLI 参数、也没有 `.env` 开关，必须把你手上的权重**重命名**成上表名字。
   唯一的便利开关是 LoRA —— `.env` 里的 `H3_LORA_NAME` 可改 LoRA 文件名。
2. 若你的权重文件名和上面不同，有两个解法：
   (a) 直接把文件**重命名**成上表名字（最简单）；
   (b) 让 runner 支持用环境变量覆盖这 5 个核心路径（类似 `H3_LORA_NAME`，如
   `H3_FL2VA_MODEL` / `H3_REF2VA_MODEL` / `H3_TEXT_ENCODER` / `H3_VIDEO_VAE` / `H3_AUDIO_VAE`）。
3. **参考素材（图片/音频/视频）不放这里** —— 它们由 WebUI 上传到 `data/uploads/`，
   推理时由 `build_media_bundle` 解码后传入，与 `models/` 无关。
4. `models/` 下的 `.safetensors/.ckpt/.pt/...` 等大模型文件已被 `.gitignore` 忽略，
   本目录只保留结构占位（`.gitkeep`）与本文档，**不会**把权重提交进 git。
5. 挂载关系：`docker-compose.yml` 中 `api` / `worker` 服务把宿主 `./models` 以只读方式
   挂到容器 `/models`；改完文件名后无需重新 build 镜像，重启容器即可（`docker compose restart worker`）。
