# 工作流拍平说明（Flattened Workflows）

> 目的：把 ComfyUI 节点图（有向图）拍平为**线性、可直接用 Python 调用的管线**，作为
> `inference/run_minimax_h3.py` 头less 运行器的蓝图。所有参数/模型文件名都来自原始工作流
> JSON（`MiniMax_H3_Easy.json` / `MiniMax_H3_Easy_Pass2.json`），未做任何臆测。
>
> 关键结论（见 `LINUX_PORT.md`）：H3 的真实推理代码在 **ComfyUI 内核**里
> （`comfy/ldm/minimax`、`comfy/ldm/minimax_music`、`comfy_extras/nodes_minimax_h3.py`、
> `transformers/models/minimax`），自定义节点 `ComfyUI-MiniMaxH3-Easy` 只是薄封装。
> 因此"直接用 py 跑" = vendor ComfyUI 内核 + `nodes_minimax_h3`，直接调节点的 `execute()`，
> 绕开 server / 节点图 / 自定义节点管理器。

---

## 一、主工作流 `MiniMax_H3_Easy.json`（单遍 · 参考生视频）

对应节点调用链（线性顺序）：

### S0 · 加载 H3 模型组合 — `MiniMaxH3EasyLoader`
| 槽位 | 值 |
|---|---|
| fl2va_model | `minimax_h3_fl2va_int8_convrot.safetensors` |
| ref2va_model | `无`（none，单遍模式不加载） |
| text_encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| video_vae | `minimax_h3_video_vae_fp16.safetensors` |
| audio_vae | `minimax_h3_audio_vae_fp32.safetensors` |

→ 产出 `h3_bundle`（模型 + 文本编码器 + 视频/音频 VAE 句柄）。

### S1 · 参考媒体 — `LoadImage` → Media 端口
- 参考图：`ComfyUI_temp_anoxe_00001_.png`（单张，对应 `<Picture 1>`）
- 通过 `MiniMaxH3Easy` 的 `media` 输入（节点属性 `minimax_h3_virtual_media_links` 指向该图）。

### S2 · H3 核心 — `MiniMaxH3Easy`  → 内部调用 `MiniMaxH3ReferenceToVideo.execute(...)`
| 参数（widgets_values 顺序） | 值 | 含义 |
|---|---|---|
| mode | `参考生视频` | reference-to-video |
| prompt | `\n`（占位，运行时由用户填） | 文本提示词（支持 `<Picture N>` / `<d>...</d>` 结构化标签） |
| — | `null` | 外部文本（未接） |
| resolution | `360P` | 分辨率档位 |
| aspect_ratio | `9:16` | 画幅 |
| width | `1344` | 画布宽（内部对齐用） |
| height | `1344` | 画布高 |
| num_frames | `5` | 潜在帧数（length） |
| flag | `false` | （首尾帧开关，关闭） |
| fps | `24` | 帧率 |
| frame_priority | `首帧优先` | first-frame priority |
| scale | `原图（不缩放）` | keep original, no upscale |
| order | `按序号` | by sequence |
| general_only | `仅通用方案` | general-only scheme |

→ 产出 `model`（已接好文本编码器/VAE 的 H3 模型）+ `h3_context`（条件上下文）。

- **提示词优化（可选，非推理）**：节点属性 `minimax_h3_prompt_optimizer` = OpenAI 兼容 API
  `https://zz.211b.site`，model `gpt-5.6-terra`，`read_media=true`。这是外部 LLM 改写提示词，
  **与 H3 推理无关**，可在脚本里设为可选或去掉。

### S3 · LoRA — `LoraLoaderModelOnly`
- lora：`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`
- strength：`1.0`
→ 叠加到 model。

### S4 · 注意力后端 — `ModelAttentionBackend`
- backend：`comfy kitchen attention`（来自 `comfy_kitchen` 包）
→ 替换 model 的注意力实现。

### S5 · 输出准备 — `MiniMaxH3EasyOutput`
消费 `h3_context` → 产出：
- `positive`（CONDITIONING，给采样器）
- `latent`（空 AV latent，尺寸按画布=width×height×length）
- `video_vae`、`audio_vae`（回传供解码）
- `fps` = `24`

### S6 · 采样器配置
- `RandomNoise`：seed = `283248527611089`（`randomize` 时可随机）
- `KSamplerSelect`：sampler = `euler`
- `BasicScheduler`：scheduler = `simple`，steps = `8`，denoise = `1`
- `BasicGuider`：model + positive

### S7 · 采样 — `SamplerCustomAdvanced`
输入：noise + guider + sampler(euler) + sigmas(8步) + latent_image(空 AV latent)
→ 输出 `output`(视频潜在) + `denoised_output`(含音频潜在)。

### S8 · 解码
- `VAEDecode`：output latent → IMAGE（视频帧），用 `video_vae`
- `VAEDecodeAudio`：denoised latent → AUDIO，用 `audio_vae`

### S9 · 组装视频 — `CreateVideo`
输入：images + audio + fps(24) → VIDEO。

### S10 · 保存 — `SaveVideo`
- 路径：`video/MiniMaxH3_Easy`
- （`RAMCleanup` 为显存回收，脚本里用 `torch.cuda.empty_cache()` 替代）

---

## 二、双阶段工作流 `MiniMax_H3_Easy_Pass2.json`（DualStage 放大/精修）

第一遍低分辨率生成 → 解码；第二遍加载 REF2VA W4A8 模型、复用第一遍音频 latent、做二阶段
条件再采样并放大到 1920×1088。

### 阶段 A · 第一遍（低分辨率生成）
等价于主工作流，但参数不同：

| 项 | 值 |
|---|---|
| mode | `图生或首尾帧` → `MiniMaxH3ImageToVideo` |
| 参考图 | `be5db0607cf5a9bd98675031f3982f82.png` |
| resolution | `360P` |
| aspect_ratio | `2:3`（注意：不是 9:16） |
| width / height | `1344` / `768` |
| num_frames | `4` |
| flag | `true` |
| fps | `24` |
| frame_priority | `首帧优先` |
| scale | `1.5K 面积（约2.25MP）` |
| order | `按序号` |
| general_only | `仅通用方案` |
| LoRA | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` strength 1.0 |
| Attention | `comfy kitchen attention` |
| Scheduler | `beta`，steps `8`，denoise `1` |
| Sampler | `euler` |

第一遍解码后：
- `VAEDecode` → 第一遍帧；`VAEDecodeAudio` → 第一遍音频
- `LTXVSeparateAVLatent`：拆分第一遍 AV latent → `video_latent` + `audio_latent`
  （**audio_latent 在阶段 B 复用**）
- `CreateVideo`(第一遍) → `SaveVideo`(`video/MiniMaxH3_Easy_DualStage_Standard_FirstPass`)

### 阶段 B · 第二遍（放大 / 精修）
1. `ResolutionSelector`：aspect_ratio=`9:16`，megapixels=`1`，multiple=`32` → 目标 `1920 × 1088`
2. `ImageResizeKJv2`：第一遍帧放大到 `1920×1088`（lanczos, crop, divisible_by=32, cpu）
3. `VAEEncode`：用 `video_vae` 重新编码放大帧 → `second_pass_video_latent`
4. `UNETLoader`：加载第二阶模型
   - `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`，weight_dtype=`default`
   - （即 REF2VA W4A8 剪枝模型）
   - 叠加 LoRA（同上 turbo 8step bf16，strength 1.0）、`comfy kitchen attention`
5. `MiniMaxH3EasySecondPassConditioning`：`h3_context` + `second_pass_video_latent`
   → `second_pass_positive`（二阶段条件）
6. `BasicScheduler`：`beta`，steps `3`，denoise `0.25`
7. `BasicGuider`：第二阶 model + `second_pass_positive`
8. `LTXVConcatAVLatent`：`video_latent`(第二遍) + `audio_latent`(第一遍复用) → `av_latent`
9. `SamplerCustomAdvanced`(第二遍)：noise(可与第一遍共享) + guider + euler + sigmas(3步) + `av_latent`
10. `VAEDecode`(第二遍) → 帧
11. `CreateVideo`(第二遍) → `SaveVideo`(`video/MiniMaxH3_Easy_DualStage_Standard_Final`)

> 附：megapixels→分辨率参考表（来自工作流内 `MarkdownNote`，16:9 示例）：
> 0.98MP→1344×768，1.0MP→1376×768，1.2MP→1504×832，1.5MP→1664×928，2.0MP→1920×1088。

---

## 三、拍平后可直接调用的核心入口（来自 `comfy_extras/nodes_minimax_h3.py`）

| 类 / 函数 | 在拍平管线中的位置 |
|---|---|
| `MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, length, ref_image_size=...)` | 主工作流 S2 |
| `MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length, ...)` | Pass2 阶段 A |
| `MiniMaxH3AddGuide.execute(positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None)` | 首尾帧引导 |
| `MiniMaxH3SecondPassConditioning`(等价逻辑) | Pass2 阶段 B.5 |
| `MiniMaxH3SigmaShift.execute(model, shift_video, shift_audio)` | scheduler shift |
| `EmptyMiniMaxH3LatentAV.execute(width, height, length)` | 空 AV latent |
| 辅助：`_empty_av_latent` / `_resize` / `adapt_canvas` / `video_latent_t` / `temporal_shape` / `align_frame_count` | 尺寸/潜在对齐 |

> 标准 ComfyUI 节点（采样器/VAE/创建视频）在头less 运行器里直接用 `comfy.samplers` /
> `comfy.latent` / `comfy_extras.nodes_vhs`(CreateVideo) 等对应函数替代，见 `run_minimax_h3.py`。

---

## 四、模型文件清单（需放到 Linux 的 `models/` 对应子目录）

| 文件 | 用途 | 来源 |
|---|---|---|
| `minimax_h3_fl2va_int8_convrot.safetensors` | FL2VA 主模型（int8） | README 资源盘 / HuggingFace `Kijai/MiniMax-H3-experimental` |
| `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | REF2VA 第二阶（W4A8 剪枝） | 同上 |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 文本编码器（Qwen3-VL 32B, NVFP4 AWQ） | README 资源盘 |
| `minimax_h3_video_vae_fp16.safetensors` | 视频 VAE | 资源盘 |
| `minimax_h3_audio_vae_fp32.safetensors` | 音频 VAE | 资源盘 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | LoRA（turbo 8步, bf16） | HuggingFace `lightx2v/Minimax-h3-Turbo` |
