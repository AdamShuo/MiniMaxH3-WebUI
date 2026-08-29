# MiniMax-H3 WebUI（文字 / 图片 + 音频 → 带音频视频）

一套**可部署**的全栈工程：把原 Windows 桌面应用（MiniMax-H3 工具 V1.4）的能力，重做成
**Linux（含 GPU）云端 WebUI**。架构、接口契约、安全与部署策略均来自已冻结的设计文档
（高层架构 / 系统设计 / UserStory / 部署设计 / 安全设计，G0~G6 评审 64/64 通过）。

- **后端**：FastAPI（M1~M5 + B1~B5），异步任务队列 RQ + Redis
- **前端**：Gradio MVP（生成 Tab + 画廊 Tab）
- **引擎**：本地 ComfyUI（GPU，主）+ MiniMax 官方 API（兜底，H-04 方案B，**默认开启**）
- **存储**：默认 SQLite（零配置即可 `docker compose up`），生产可切 PostgreSQL

> 原 Windows 应用包（`F:/ProgramData/AI文字图片生成带音频的视频（MiniMax-H3）工具V1.4/`）
> 为**只读参考**，本仓库不修改、不打包它；其 `prompt_guides/` 已**只读复制**进本仓库的
> `prompt_guides/`，原路径零改动。

---

## 1. 目录结构

```
minimax-h3-webui/
├── backend/
│   ├── app/
│   │   ├── api/         # M1 assets / M2 templates / M3 generations / M4 tasks / M5 results
│   │   ├── engine/      # B1 comfyui / B2 minimax / router / base
│   │   ├── config.py    # L1~L4 配置分层（env 驱动，fail-fast）
│   │   ├── db.py        # SQLAlchemy 2.0（SQLite/Postgres 兼容）
│   │   ├── models.py    # t_* ORM
│   │   ├── schemas.py   # Pydantic DTO
│   │   ├── prompt_llm.py# B3 提示词优化（OpenAI 兼容，可降级）
│   │   ├── tasks_queue.py# RQ 队列
│   │   ├── worker.py    # 生成状态机 + 失败切兜底（BD-02）
│   │   └── main.py      # FastAPI 入口 + /healthz /readyz
│   ├── workflows/h3_fl2v.json   # H3 工作流【模板】，需替换为你的真实导出
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py          # Gradio MVP
│   ├── requirements.txt
│   └── Dockerfile
├── prompt_guides/      # 9 类场景指南（只读复制自原应用，用于模板灌库）
├── reference/example_settings.json  # 原 MiniMaxh3_settings.json 脱敏副本
├── models/             # 【不入库】LoRA / 权重，运行实例上传到此
├── data/               # 【不入库】运行时 uploads / outputs / db
├── docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml
```

---

## 2. 快速开始（零 GPU，MiniMax 兜底）

只要有 `MINIMAX_API_KEY` 即可在任意机器跑通端到端流程（无需显卡）：

```bash
cp .env.example .env
# 编辑 .env，填入 MINIMAX_API_KEY（区域默认新加坡，见 H-04 合规提示）

docker compose up -d
#   单端口：后端 API + Gradio UI 同进程，统一暴露 7860
#   -> http://localhost:7860  （WebUI，Swagger 文档在 /docs）
#   worker -> 后台消费队列（GPU 实例上调用 inference/run_minimax_h3.py 直跑）
```

打开 `http://localhost:7860`：填提示词 → 选「提示词优化方式」（同实例 H3 文本编码
或第三方 API）→ 选参数/参考图/音频/视频 →「开始生成」→ 轮询进度 → 视频直接播放，
画廊可回看下载。

---

## 3. GPU 全功能（本地 H3 直跑引擎，默认主引擎）

本仓库**不再依赖 ComfyUI server**。后端 worker 通过子进程直接调用
`inference/run_minimax_h3.py`（复用 vendored `comfy_core/` 内核 + 本地权重），
在 GPU 实例上生成带音频视频。ComfyUI 仅作为可选兜底存在。

在**含 GPU 的 Linux 实例**上：

1. **放置权重**：把 H3 模型 / VAE / 文本编码器 / LoRA 放到 `./models/`
   （默认文件名见 `inference/run_minimax_h3.py` 的 `DEFAULT_CFG`，LoRA 可用
   `H3_LORA_NAME` 覆盖；`MODELS_DIR` 通过环境变量指向权重目录）。
2. **开启本地引擎**：`.env` 中 `LOCAL_H3_ENABLED=true`（默认值即开）。
   可设 `RUNNER_SCRIPT` / `COMFY_CORE_DIR` / `INFERENCE_PYTHON` / `H3_ATTENTION_BACKEND`
   调整运行方式（注意力后端支持 `torch` / `sageattention` / `xformers` /
   `comfy_kitchen_int8`）。
3. **（可选）启用独立 ComfyUI 兜底**：如需，`.env` 设 `ENABLE_COMFYUI=true` 并准备
   自带 H3 节点包的 ComfyUI 镜像（通过 `COMFYUI_IMAGE` 指定）。
4. **启动**：
   ```bash
   docker compose --profile gpu up -d
   ```
   路由逻辑：ComfyUI 健康 → 用本地引擎；本地失败且 `USE_FALLBACK=true` → 自动切 MiniMax（BD-02）。

---

## 4. 接口契约（M1~M5）

| 模块 | Method | Path | 说明 |
|------|--------|------|------|
| M1 素材 | POST | `/api/v1/assets` | 上传参考图/音频，返回 `id` |
| M2 模板 | GET | `/api/v1/prompt-templates` | 列出 9 类场景模板（启动时从 `prompt_guides/manifest.json` 灌库） |
| M2 优化 | POST | `/api/v1/prompt-templates/{id}/optimize` | LLM 优化提示词（无 key 时静默降级 BD-01） |
| M3 生成 | POST | `/api/v1/generations` | 创建生成请求，返回 `id` |
| M3 提交 | POST | `/api/v1/generations/{id}/submit` | 派生 Task 并入队，返回 `task id` |
| M4 任务 | GET | `/api/v1/tasks/{id}` | 轮询进度 `progress` / `status` |
| M4 重试 | POST | `/api/v1/tasks/{id}/retry` | 失败任务重试（可指定 `force_engine`） |
| M5 结果 | GET | `/api/v1/results?page_size=50` | 画廊列表 |
| M5 下载 | GET | `/api/v1/results/{id}/download` | 视频文件 |
| Ops | GET | `/healthz` `/readyz` | 健康检查 / DB 探活 |

前端完整对齐上述契约（`frontend/app.py`）。

---

## 5. 默认生成参数（与原应用一致）

`step=8, seed=-1, w=1376, h=768, duration=10s, fps=24, resolution=360P`；
LoRA 默认 `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`，strength=1.0。
提示词优化 LLM 端点占位 `LLM_API_URL` / `LLM_MODEL=agnes-2.5-flash`（密钥用 env，已脱敏）。

---

## 6. ⚠️ 数据出境合规（H-04 方案B）

MiniMax 兜底引擎区域为**新加坡**（`api.minimax.chat`）。当主引擎不可用、走兜底路径时，
参考图 / 音频可能上传至 MiniMax。**上线前请完成数据出境合规评估**。
Gen 参数与提示词默认不离开本地；仅兜底链路涉及出境。生产可在 `FORCE_FALLBACK=false` 且
`USE_FALLBACK=false` 下禁用兜底以彻底避免出境（但会牺牲可用性）。

---

## 7. 切换到 PostgreSQL（生产）

1. 在 `.env` 设 `DATABASE_URL=postgresql+psycopg2://user:pass@db:5432/minimaxh3`
2. 在 `docker-compose.yml` 的 `api` / `worker` 增加 `depends_on: [db]` 与 `db` 服务
   （Postgres 16.4 镜像）。
3. Model 层已用 SQLAlchemy 2.0 + `Text` 兼容 JSONB，无需改代码。

---

## 8. 本地开发（不依赖 Docker）

```bash
# 后端 + 挂载式 Gradio UI（单端口 7860）
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 7860
# 浏览器打开 http://localhost:7860 即 WebUI（UI 通过同源 /api/v1 调用后端）

# worker（另一终端，GPU 机器需额外装推理栈：pip install -r inference/requirements-linux.txt）
rq worker --url redis://localhost:6379/0 default

# 仅前端独立开发（不挂载进后端，需单独跑 API 在 8000）：
#   pip install -r frontend/requirements.txt
#   BACKEND_URL=http://localhost:8000 python frontend/app.py
```

---

## 9. CI/CD

`.github/workflows/ci.yml` 实现设计文档 §4.2 的八阶段流水线：
检出 → 依赖安装 → 代码静态检查 → 后端 import/启动冒烟 → 构建后端镜像 →
构建前端镜像 → 镜像扫描（Trivy）→ 汇总。默认在 `push` / `PR` 到 `main` 时触发。

---

## 10. 已知约束与后续

- H3 工作流须用户自行导出（R-01/U-02），仓库仅提供占位模板。
- MiniMax 兜底为无 GPU 即可测试的默认路径；生产需评估数据出境（H-04）。
- MVP 鉴权采用方案B（单租户、网关/反向代理层鉴权，O1），应用层不做多租户隔离。
- 性能 SLA 采用方案A（固定 8 step / 360P 基线），已在代码中硬编码为默认值（§6.2）。
