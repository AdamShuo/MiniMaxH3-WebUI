# Linux 适配方案（Dependency & Porting）

> 目标：把原项目 `F:\ProgramData\AI文字图片生成带音频的视频（MiniMax-H3）工具V1.4\capai\_internal`
> 这套 **Windows 嵌入式 Python 3.10 环境** 改造为可在云端 Linux（GPU）实例上**直接用 Python 跑 H3**
> 的环境，彻底摆脱 ComfyUI server / 节点图 / 自定义节点管理器的"不确定因素"。

---

## 1. 源环境画像（已实测）

- **类型**：Windows 嵌入式 Python 3.10（`cp310-win_amd64`），全部依赖以 `.pyd`/`.dll` 形式预编译。
- **总包数**：101 个 `dist-info`。
- **关键 ML 栈（Linux 需 1:1 复刻版本）**：

| 包 | 版本 | 备注 |
|---|---|---|
| torch | `2.10.0+cu130` | CUDA 13.0 构建 |
| torchvision | `0.25.0+cu130` | |
| torchaudio | `2.10.0` | |
| transformers | `4.57.6` | 含 `transformers/models/minimax` 自定义模型 |
| diffusers | `0.37.1` | |
| accelerate | `1.12.0` | |
| sageattention | `2.2.0+cu130torch2.9.0andhigher.post4` | 注意力加速 |
| nunchaku | `1.3.0.dev20260213+cu13.0torch2.10` | W4A8/FP4 推理 |
| bitsandbytes | `0.49.1` | |
| torchao | `0.13.0` | |
| peft | `0.17.1` | LoRA |
| rotary_embedding_torch | `0.8.9` | |
| insightface | `0.7.3` | 人脸（可选） |
| opencv_python | `4.10.0.84` | |
| librosa | `0.10.2` / pyloudnorm / av `16.0.1` | 音频处理 |
| comfy_kitchen | `0.2.31` | 即工作流里 "comfy kitchen attention" 后端 |
| comfy_aimdo | `0.4.13` | ComfyUI 内部 |
| huggingface_hub | `0.34.4` | |

- **真实模型代码位置（在 ComfyUI 内核里，不在自定义节点）**：
  - `comfy/ldm/minimax/`：`audio_vae.py` / `model.py` / `vae.py`
  - `comfy/ldm/minimax_music/`（音频分支）
  - `comfy_extras/nodes_minimax_h3.py`：把上述模型封装为 ComfyUI 原生节点的 `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` / `MiniMaxH3AddGuide` / `MiniMaxH3SigmaShift` / `EmptyMiniMaxH3LatentAV` 等。
  - `transformers/models/minimax`：文本编码器（Qwen3-VL 变体）。
  - `custom_nodes/ComfyUI-MiniMaxH3-Easy/nodes.py`：薄封装（Loader / Easy / Output / MediaBridge / SecondPassConditioning）。

> 结论：**"直接用 py 跑" 不是从零重写，而是把 ComfyUI 内核（`comfy` 包 + `comfy_api` 包）
> 作为库 vendored 进来，直接调用 `nodes_minimax_h3` 里的节点 `execute()`**。模型代码本身
> 是跨平台的纯 torch；Windows 耦合仅来自预编译二进制和少数 Windows 专有包。

---

## 2. Windows → Linux 必须处理的差异

### 2.1 二进制扩展（自动解决）
源环境的 `.pyd` 是 Windows 专有，Linux 上通过 `pip install` 同名包的 **Linux wheel** 自动替换。
不要在 Linux 复制任何 `.pyd` / `.dll`。

### 2.2 Windows 专有包 → 替换 / 删除
| 源（Windows） | Linux 处理 |
|---|---|
| `triton_windows 3.7.1.post27` | 替换为 `triton`（与 torch 2.10 匹配的版本，通常 pip 会自动带） |
| `pyreadline3 3.5.4` | **删除**（仅 Windows 交互式 REPL 用，脚本无需） |
| `api-ms-win-*.dll` / `python3*.dll` | 由 Linux Python 运行时提供，忽略 |

### 2.3 Python 代码层（已核验：无硬编码 Windows 路径）
对 `nodes.py` 与 `nodes_minimax_h3.py` 做了 `C:\`、`os.sep`、`win32`、`ctypes.windll`、`.pyd`、
`ProgramData` 等模式扫描：**零命中**。代码使用 `os` / `folder_paths` / `pathlib` 等跨平台写法，
**无需改源码即可在 Linux 跑**。唯一要注意的是模型/输出路径在 Linux 用正斜杠与绝对路径。

### 2.4 CUDA / 驱动对齐
源是 `cu130`（CUDA 13.0）。云端 Linux GPU 实例常见为 `cu124` / `cu128`。
- 推荐：**torch 与实例驱动对齐**（如实例是 CUDA 12.4 → 装 `torch==2.10.0+cu124`）。
- 版本号（2.10.0）尽量保持一致，构建标签（cuXXX）按实例实际驱动选。
- 同理 `sageattention` / `nunchaku` 也选对应 CUDA 构建。

### 2.5 模型权重
权重文件（`.safetensors`，见 `FLATTENED_WORKFLOWS.md` 第四节）从 README 资源盘 / HuggingFace 下载，
放到 Linux 的 `models/` 对应子目录（FL2VA / REF2VA / text_encoder / vae / lora），**不要带入 git**。

---

## 3. 头less 运行器策略（推荐落地方式）

> 实际落地结构（已生成，见仓库 `comfy_core/`）：

```
inference/
├── run_minimax_h3.py        # 头less 运行器：复用开放节点 + 重构密封节点
├── run.sh                   # Linux 启动脚本（设 PYTHONPATH/comfy_core、建模型目录、转发参数）
├── comfy_core/              # vendored ComfyUI 内核（仅保留推理所需，来自 F 盘只读源逐字拷贝）
│   ├── comfy/               # model_management / model_sampling / nested_tensor / utils /
│   │                       #   ldm/minimax (model.py/vae.py/audio_vae.py) / ldm/minimax_music
│   ├── comfy_api/           # io.ComfyNode / ComfyExtension / NodeOutput 框架
│   ├── comfy_extras/
│   │   └── nodes_minimax_h3.py        # 开放节点：Reference/ImageToVideo/AddGuide/SigmaShift
│   ├── custom_nodes/
│   │   └── ComfyUI-MiniMaxH3-Easy/    # 薄封装：Loader / Easy / EasyOutput /
│   │                               #   SecondPassConditioning / MediaBridge（开放）
│   ├── folder_paths.py       # SHIM：get_filename_list / folder_names_and_paths / *_directory
│   ├── node_helpers.py       # SHIM：conditioning_set_values
│   └── nodes.py              # SHIM：MAX_RESOLUTION + UNETLoader/CLIPLoader/VAELoader
├── requirements-linux.txt
├── FLATTENED_WORKFLOWS.md
└── LINUX_PORT.md
```

- **做法**：`comfy/` / `comfy_api/` / `comfy_extras/nodes_minimax_h3.py` 是从
  `F:\...\capai\_internal\comfy` **逐字拷贝**的 H3 定制 fork（非全新 clone ComfyUI），因为
  真实推理代码在该 fork 的 `comfy/ldm/minimax*` 里；`custom_nodes/ComfyUI-MiniMaxH3-Easy/nodes.py`
  是开放的薄封装。三个 `SHIM` 文件（`folder_paths.py` / `node_helpers.py` / `nodes.py`）是
  **新增**的——因为原 app-root 的这三个模块被封进 `capai.exe`，本仓库以最小 API 面重新实现。
- **运行器** `run_minimax_h3.py` 通过 `comfy_core` 的 `sys.path` 引导直接 `import comfy.*`、
  `from comfy_extras import nodes_minimax_h3 as h3`、并以命名文件加载方式 `importlib` 加载自定义节点
  （避免与 shim `nodes.py` 重名冲突），按 `FLATTENED_WORKFLOWS.md` 的 S0–S10 顺序调用，绕过
  server / UI / 节点图。
- **收益**：消除 ComfyUI server 启动、节点图执行引擎、自定义节点版本漂移、UI 相关不确定性；
  单进程、可断点、可批处理、可嵌入 WebUI 后端。

> 注意：`comfy` 内部并非稳定公开 API，vendoring 锁定 fork 版本是关键，升级需回归测试。

### 3.1 开放节点（直接复用） vs 密封节点（已重构）

原工作流里的节点分两类——**开放**节点（源码在 vendored `comfy`/`custom_nodes` 里，直接调用）
与**密封**节点（源码编译进 `capai.exe`，不可得，已在 `run_minimax_h3.py` 中以
`comfy.*` 公共 API **重构**，标注 `[#RECONSTRUCTED]`）：

| 工作流节点 | 类别 | 运行器中的实现 |
|---|---|---|
| `MiniMaxH3EasyLoader.load` | 开放 | `build_bundle()` |
| `MiniMaxH3Easy.generate` / `MiniMaxH3EasyOutput.unpack` | 开放 | `prepare_conditioning()` |
| `MiniMaxH3SigmaShift.execute` | 开放 | `sample_h3()` 内调用 |
| `MiniMaxH3EasySecondPassConditioning.rebuild` | 开放 | `run_pass2()` 内调用 |
| `MiniMaxH3EasyMediaBridge` | 开放 | （媒体桥接，参考图/音频入参用） |
| `MiniMaxH3EasySampler`（采样） | **密封→重构** | `sample_h3()`：`comfy.samplers.sample` + `calculate_sigmas` + `prepare_noise` |
| `MiniMaxH3EasyLoRAApply` | **密封→重构** | `apply_lora()`：`comfy.sd.load_lora_for_models` |
| `MiniMaxH3EasyAttentionBackend` | **密封→重构** | `set_attention_backend()`：切换 `comfy.ldm.modules.attention` |
| `MiniMaxH3EasyVAEDecode` | **密封→重构** | `decode_av()`：直调 `MiniMaxH3VideoVAE`/`MiniMaxH3AudioVAE.decode` |
| `MiniMaxH3EasyCreateVideo` / `SaveVideo` | **密封→重构** | `save_video()`：imageio + ffmpeg mux 音视频 |
| 提示词优化 API | **关闭** | `_noop_optimize` 桩（去掉第三方 LLM 出网） |

> 密封节点重构路径（采样 / VAE 解码 / 双阶段二遍条件）是**最高风险**部分，
> 必须在云端 GPU 实例实跑验证（先 `--check` 校验 import，再真正跑一条视频）。

---

## 4. 实施步骤（到云端实例上执行）

> `comfy_core/`（含 vendored `comfy`/`comfy_api`/`comfy_extras`/`custom_nodes` 与三个 SHIM）
> 已随仓库提供，**无需再 clone ComfyUI**。只需准备依赖 + 权重，然后跑 `run.sh`。

1. 准备 Linux Python 3.10 venv/conda（`python3 --version` 应为 3.10.x）。
2. 安装依赖：`cd <repo> && PIP_INSTALL=1 ./inference/run.sh --check`
   （`run.sh` 会 `pip install -r inference/requirements-linux.txt`；按需把 torch 系列
   的 `+cu130` 改成实例实际 CUDA 的 `+cu124`/`+cu128`，见 `requirements-linux.txt` 顶部注释）。
3. 下载 6 个权重到 `models/` 对应子目录（`unet/` 放 FL2VA/REF2VA，`clip/` 放文本编码器，
   `vae/` 放视频/音频 VAE，`loras/` 放 turbo LoRA）。**不要带入 git**（`.gitignore` 已排除 `models/`）。
4. 先校验 import 链路：`python inference/run_minimax_h3.py --check`
   （该命令只验证 `comfy`/`comfy_api`/自定义节点/SHIM 全部可解析，不加载权重、不占显存）。
5. 单遍跑通：`python inference/run_minimax_h3.py --mode reference --prompt "..." \
   --ref-image-1 ref.png --output out.mp4`（或 `./inference/run.sh --config job.json`）。
6. 双阶段：准备 `pass2.json`（第二阶参数 + 目标分辨率 1920×1088），用
   `python inference/run_minimax_h3.py --config job.json --pass2-config pass2.json`。

---

## 5. 风险提示
- `comfy` 内部 API 随版本变动；vendoring 锁定 fork 版本是关键，升级需回归测试。
- `nunchaku`/`sageattention` 为预发布/特定 CUDA 构建，若实例 CUDA 不符需重新编译或换等价实现。
- **密封节点重构路径（采样 / VAE 解码 / 双阶段二遍采样）未经真实 GPU 运行验证**：
  - 采样器 `comfy.samplers.sample` 是否能原样保留 AV 潜在（video+audio 的 NestedTensor）需实跑确认；
    `decode_av()` 已对"非 2 分量 NestedTensor"给出明确报错，便于在 GPU 上快速定位。
  - 双阶段 `run_pass2()` 当前为**简化重建**：第一遍解码出的视频帧未回灌第二遍
    （原工作流用 `LTXVConcatAVLatent` 复用第一遍音频潜在），第二遍条件仅由
    `MiniMaxH3EasySecondPassConditioning.rebuild` 在目标分辨率重建。若需严格对齐原工作流，
    需补"第一遍音频潜在 → 第二遍 concat"这一段（见 `FLATTENED_WORKFLOWS.md` §二 阶段 B.8）。
- 提示词优化 API（`zz.211b.site`/`gpt-5.6-terra`）已**关闭**（改为 `_noop_optimize` 桩），
  去掉了第三方 LLM 出网依赖；如需提示词改写，可在 `run_minimax_h3.py` 中接自有端点。
