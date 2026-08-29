# 部署手册（DEPLOY.md）

云端 GPU 推理的通用方式：**后端启动后只暴露一个端口（本工程为 `7860`），用户直接访问 `实例地址:7860` 即可打开 WebUI**。UI（Gradio）与 API（FastAPI）同进程同源，UI 通过相对路径调用 `/api/v1/*`，无需第二端口、无需 CORS。

> 本手册与 `README.md` 互补：README 讲架构与功能，本文件讲「怎么把它跑起来」。
> 权重放哪见 [`models/README.md`](models/README.md)；云端冒烟测试见 [`scripts/smoke_test.sh`](scripts/smoke_test.sh)。

---

## 1. 部署拓扑

```
                        实例公网 IP:7860
                               │
                 ┌─────────────┴──────────────┐
                 │   api 容器（单端口 7860）    │
                 │  FastAPI + 挂载的 Gradio UI │
                 └───────┬──────────────┬──────┘
                         │ 入队/出图      │ HTTP（兜底时才出网）
                         ▼               ▼
                   redis 队列      MiniMax 官方 API（可选兜底）
                         │
                         ▼
                 ┌──────────────────────┐
                 │  worker 容器（有 GPU） │  ← 真正的推理发生在这里
                 │  LocalH3Engine 子进程  │
                 │  → run_minimax_h3.py  │
                 └──────────────────────┘
```

要点：
- **7860 一个端口**里同时跑着 FastAPI 和挂载进去的 Gradio UI。
- 真正吃 GPU 的推理在 **worker 容器**（LocalH3Engine 子进程调 `inference/run_minimax_h3.py`）。
- api 容器只负责收请求、入队、出图，**不需要 GPU**。

---

## 2. 前置条件

| 项 | 要求 |
|---|---|
| 操作系统 | Linux（Ubuntu 22.04+ 推荐） |
| Docker | Docker Engine 24+ 与 Docker Compose v2（`docker compose` 子命令） |
| NVIDIA Container Toolkit | 必须。让容器能见到宿主 GPU（否则 worker 内 `nvidia-smi` 为空） |
| NVIDIA 驱动 | **驱动支持的 CUDA 版本需与镜像一致**（见坑①） |
| 权重 | 把 H3 模型/VAE/文本编码器/LoRA 放进仓库根 `./models/`（见 `models/README.md`） |

安装 Container Toolkit 后验证：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# 能打印出显卡信息即 OK
```

---

## 3. 快速开始

### 3.1 零 GPU 快速体验（只看 UI / 走 MiniMax 兜底）
不需要显卡，但需要 `MINIMAX_API_KEY`：
```bash
git clone https://github.com/AdamShuo/MiniMaxH3-WebUI.git
cd MiniMaxH3-WebUI
cp .env.example .env
# 编辑 .env：填入 MINIMAX_API_KEY（LOCAL_H3_ENABLED 留默认 true 也无妨，失败会兜底）
docker compose up -d
curl -f http://localhost:7860/healthz     # -> {"status":"ok"}
# 浏览器打开 http://<实例公网IP>:7860
```
> ⚠️ 兜底路径下，参考图/音频会传至 MiniMax 新加坡节点（数据出境，见 `.env` 中 H-04 标注）。

### 3.2 全功能 GPU 部署（本地 H3 直跑，默认主引擎）
```bash
git clone https://github.com/AdamShuo/MiniMaxH3-WebUI.git
cd MiniMaxH3-WebUI

# 1) 放权重：把模型文件按 models/README.md 的子目录结构上传到 ./models/
#    （参考图/音频不出本机，无需 MINIMAX_API_KEY）

# 2) 准备环境变量
cp .env.example .env
#   默认 LOCAL_H3_ENABLED=true 即开本地引擎；MINIMAX_API_KEY 可留空（仅兜底用）

# 3) 构建并启动（首次会拉重推理栈，几分钟到十几分钟）
docker compose up -d --build

# 4) 看状态
docker compose ps
curl -f http://localhost:7860/healthz     # -> {"status":"ok"}

# 5) 实跑冒烟（见第 5 节）
docker compose exec worker bash /app/scripts/smoke_test.sh --generate
```

---

## 4. 配置 .env（关键项）

| 变量 | 说明 |
|---|---|
| `LOCAL_H3_ENABLED` | 本地 H3 直跑引擎开关，默认 `true`。关掉则只用 MiniMax 兜底。 |
| `H3_ATTENTION_BACKEND` | 注意力后端：`torch`（默认）/ `comfy_kitchen_int8` / `sageattention` / `xformers`。 |
| `MINIMAX_API_KEY` | 兜底引擎 Key，可选。留空时本地引擎失败不再二次兜底（直接报错）。 |
| `BACKEND_PUBLIC_URL` | 已修正为 `http://api:7860`，须与 api 暴露端口一致（原 `8000` 是笔误）。 |
| `H3_LORA_NAME` | LoRA 文件名覆盖（唯一有便利开关的模型文件）。 |
| `MODELS_DIR` | 容器内权重目录，默认 `/models`（由宿主机 `./models` 只读挂入）。 |

---

## 5. 验证部署

### 5.1 健康检查
```bash
curl -f http://localhost:7860/healthz    # 期望返回 {"status":"ok"}
# Swagger 文档：http://<实例>:7860/docs
```

### 5.2 GPU 冒烟测试（推荐首次必跑）
脚本三阶段：import 解析 → 权重清点 → 最小生成（可选 `--generate`）。
```bash
# 仅检查 import + 权重清单（不生成，秒级）
docker compose exec worker bash /app/scripts/smoke_test.sh

# 跑一次最小生成（验证 权重加载→VAE→采样→输出 全链路）
docker compose exec worker bash /app/scripts/smoke_test.sh --generate

# 实测 comfy_kitchen_int8 注意力分支
docker compose exec worker bash /app/scripts/smoke_test.sh --generate --attention comfy_kitchen_int8

# 双阶段回灌 / 多 LoRA 路径
docker compose exec worker bash /app/scripts/smoke_test.sh --generate --dual-stage --loras mylora.safetensors:0.8
```
脚本退出码非零表示有阶段失败，请按输出的 `[FAIL]` 行排查。

---

## 6. 常见坑（部署前必读）

### 坑① CUDA 版本不匹配（最常见，首次 build 就报）
推理栈锁死具体 CUDA 标签：`inference/requirements-linux.txt` 中
`torch==2.10.0+cu130`、`sageattention==...+cu130`、`nunchaku==...+cu13.0`，
`Dockerfile.worker` 也用 `--extra-index-url .../cu130` 安装。

- 若云实例驱动支持的是 **CUDA 12.4 / 12.8**（多数常见镜像），直接 build 会装不上或运行时报错。
- **解决**：把 `requirements-linux.txt` 里的 `+cu130` 改成 `+cu124` / `+cu128`，并把 `Dockerfile.worker` 的 extra-index-url 改成对应目录（`.../cu124` 或 `.../cu128`），再重建 worker 镜像。

### 坑② worker 必须分到 GPU（本仓库已修复）
本地 H3 直跑引擎跑在 **worker 容器**。`docker-compose.yml` 的 `worker` 服务已声明
`deploy.resources.reservations.devices` 把宿主 GPU 透传进去；普通 `docker compose up`
**不会**自动透传，不声明则容器内 `nvidia-smi` 为空、引擎失败或回退 MiniMax。
（可选 `comfyui` 服务仍需 `--profile gpu` 才启用，与本修复无关。）

### 坑③ 5 个核心权重文件名是锁死的
`text_encoder` / `video_vae` / `audio_vae` / `fl2va_model` / `ref2va_model` 没有 CLI 或
`.env` 开关，文件名硬编码在 `inference/run_minimax_h3.py` 的 `DEFAULT_CFG`。
你的真实权重**必须重命名为** `models/README.md` 列出的名字，否则引擎找不到文件会
静默回退 MiniMax（数据出境）。唯一有便利开关的是 LoRA（`H3_LORA_NAME`）。

### 坑④ 干净 clone 后 worker 可能因缺文件崩溃
`comfy_core/comfy/ldm/models/autoencoder.py` 已补提交（见提交 `bc802b8`）。
若你从旧提交部署，worker import 会失败——请用最新 `master`。

---

## 7. 安全（不要裸暴露）

MVP 按设计（O1）**应用层无鉴权**，靠网关/反向代理层做认证。公网直接开 7860 有风险，建议：
- 用 **Caddy / Nginx** 反代 `:7860`，加 Basic Auth + TLS；或
- 用 **Cloudflare Tunnel / 内网穿透** 不暴露端口；
- 云安全组只放行 443，7860 不对外。

---

## 8. 日常运维

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose restart worker        # 改了权重文件名后重启即可，无需重建镜像
docker compose up -d --build          # 改了代码/依赖后重建
```

---

## 9. 故障排查速查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `curl /healthz` 失败 | api 未起 / 构建失败 | `docker compose logs api` |
| worker 日志 `nvidia-smi: command not found` 或无 GPU | GPU 未透传 | 确认 `worker.deploy.resources` 已声明；宿主装好 Container Toolkit |
| 引擎回退到 MiniMax（出网） | 权重文件名不符 / 缺失 | 按 `models/README.md` 核对文件名与子目录；跑冒烟 `-generate` 看 `[FAIL]` |
| build 时 torch 装不上 | CUDA 标签不匹配 | 见坑①，改 cu 版本 |
| 生成报缺模块 | 旧提交缺 autoencoder.py | 拉最新 `master`（含 `bc802b8`） |
