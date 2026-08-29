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

```
inference/
├── run_minimax_h3.py        # 头less 运行器：直接调用 nodes_minimax_h3 节点 execute()
├── comfy_core/              # vendored ComfyUI 内核（仅保留推理所需）
│   ├── comfy/               # model_management / model_sampling / nested_tensor / utils / ldm/minimax / ldm/minimax_music
│   ├── comfy_api/           # io.ComfyNode / ComfyExtension 框架
│   ├── comfy_extras/
│   │   └── nodes_minimax_h3.py
│   └── transformers/models/minimax/
├── requirements-linux.txt
├── FLATTENED_WORKFLOWS.md
└── LINUX_PORT.md
```

- **做法**：克隆/拷贝 ComfyUI 仓库到 `comfy_core/`（锁定一个 commit，保证 `comfy` API 稳定），
  把 `nodes_minimax_h3.py` 与 `comfy/ldm/minimax*`、`transformers/models/minimax` 一并放进去。
- **运行器** `run_minimax_h3.py` 直接 `from comfy_extras import nodes_minimax_h3 as h3`，
  按 `FLATTENED_WORKFLOWS.md` 的 S0–S10 顺序构造输入并调用 `execute()`，绕过 server/UI/节点图。
- **收益**：消除 ComfyUI server 启动、节点图执行引擎、自定义节点版本漂移、UI 相关不确定性；
  单进程、可断点、可批处理、可嵌入 WebUI 后端。

> 注意：`comfy` 内部并非稳定公开 API，vendoring 锁定 commit 是关键，升级需回归测试。

---

## 4. 实施步骤（到云端实例上执行）

1. 准备 Linux Python 3.10 venv/conda。
2. `pip install -r requirements-linux.txt`（按实例 CUDA 调整 torch 构建标签）。
3. 克隆 ComfyUI 到 `comfy_core/`（锁定 commit），保留上述子模块；其余 server/frontend 代码可删。
4. 放入 `nodes_minimax_h3.py` 与 `comfy/ldm/minimax*`、`transformers/models/minimax`。
5. 下载 6 个权重到 `models/` 对应子目录。
6. `python inference/run_minimax_h3.py --mode reference --prompt "..." --image ref.png`
   先单遍跑通，再试 `--mode dual_stage` 双阶段。

---

## 5. 风险提示
- `comfy` 内部 API 随版本变动；锁定 commit 并写好冒烟测试。
- `nunchaku`/`sageattention` 为预发布/特定 CUDA 构建，若实例 CUDA 不符需重新编译或换等价实现。
- 提示词优化 API（`zz.211b.site`/`gpt-5.6-terra`）是第三方外部服务，**与 H3 推理解耦**，
  建议改为可配置的自有 LLM 端点或去掉，避免数据出网与依赖不确定性。
