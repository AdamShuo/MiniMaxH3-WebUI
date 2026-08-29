#!/usr/bin/env bash
# ============================================================================
# MiniMax H3 — 云端 GPU 冒烟测试脚本
# ----------------------------------------------------------------------------
# 用途：在 GPU 实例上快速验证之前未在真实 GPU 上跑过的代码路径是否已经可用。
#   阶段 1：import 解析 + 打印权重目录（对应 run_minimax_h3.py --check）
#   阶段 2：清点 models/ 下 4 类权重的实际文件清单
#   阶段 3（可选 --generate）：用一张 stdlib 生成的极小测试图跑一次最小生成，
#           验证 权重加载→VAE→采样→输出 全链路（含指定注意力后端）
#
# 设计原则：
#   - 不依赖 Pillow / torchvision：测试图用 Python 标准库（zlib）凭空画一张。
#   - 默认不碰第三代/多 LoRA/双阶段，除非显式传参（--loras / --dual-stage）。
#   - 任何阶段失败都打印 FAIL 并继续，最后给出总览，退出码=非零当有失败。
#
# 运行方式（容器内，推荐）：
#   docker compose exec worker bash /app/scripts/smoke_test.sh --generate
#   docker compose exec worker bash /app/scripts/smoke_test.sh --generate --attention comfy_kitchen_int8
#   docker compose exec worker bash /app/scripts/smoke_test.sh --generate --dual-stage --loras mylora.safetensors:0.8
#
# 运行方式（宿主机，已装好推理栈并设好 env）：
#   MODELS_DIR=./models COMFY_CORE_DIR=./comfy_core \
#     bash scripts/smoke_test.sh --generate
# ============================================================================
set -u

# ---- 环境（容器内由 Dockerfile.worker 注入；宿主机可手动覆盖） ----
RUNNER_SCRIPT="${RUNNER_SCRIPT:-/app/inference/run_minimax_h3.py}"
COMFY_CORE_DIR="${COMFY_CORE_DIR:-/app/comfy_core}"
MODELS_DIR="${MODELS_DIR:-/models}"
PY="${INFERENCE_PYTHON:-python}"

# 若容器内 env 缺失，退回到仓库相对路径（宿主机直接跑）
if [ ! -f "$RUNNER_SCRIPT" ]; then
  _repo="$(cd "$(dirname "$0")/.." && pwd)"
  RUNNER_SCRIPT="${_repo}/inference/run_minimax_h3.py"
  COMFY_CORE_DIR="${_repo}/comfy_core"
  MODELS_DIR="${_repo}/models"
  PY="${PY:-python}"
fi

# ---- 参数解析 ----
DO_GENERATE=0
ATTENTION="torch"
DUAL_STAGE=0
LORA_SPECS=""
PROMPT="smoke test, a calm ocean at sunset"
WIDTH=256
HEIGHT=256
STEPS=1
SECONDS_DUR=2
FPS=8

while [ $# -gt 0 ]; do
  case "$1" in
    --generate)   DO_GENERATE=1 ;;
    --attention)  ATTENTION="$2"; shift ;;
    --dual-stage) DUAL_STAGE=1 ;;
    --loras)      LORA_SPECS="$2"; shift ;;
    --prompt)     PROMPT="$2"; shift ;;
    --width)      WIDTH="$2"; shift ;;
    --height)     HEIGHT="$2"; shift ;;
    --steps)      STEPS="$2"; shift ;;
    --seconds)    SECONDS_DUR="$2"; shift ;;
    --fps)        FPS="$2"; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done

PASS=0
FAIL=0
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
ok()   { echo "  [OK]   $1"; PASS=$((PASS+1)); }

echo "============================================================"
echo " MiniMax H3 冒烟测试"
echo " RUNNER_SCRIPT  = $RUNNER_SCRIPT"
echo " COMFY_CORE_DIR = $COMFY_CORE_DIR"
echo " MODELS_DIR     = $MODELS_DIR"
echo " PY             = $PY"
echo "============================================================"

# ---- 阶段 1：import 解析 + 权重目录 ----
echo
echo ">> 阶段 1/3: import 解析 (--check)"
if [ ! -f "$RUNNER_SCRIPT" ]; then
  fail "找不到 runner 脚本: $RUNNER_SCRIPT"
else
  OUT=$("$PY" "$RUNNER_SCRIPT" --check 2>&1)
  RC=$?
  echo "$OUT" | sed 's/^/    /'
  if [ $RC -eq 0 ] && echo "$OUT" | grep -q "all imports resolved OK"; then
    ok "所有 import 解析成功，models dir 已定位"
  else
    fail "import 解析失败 (rc=$RC)，见上方输出"
  fi
fi

# ---- 阶段 2：清点权重文件 ----
echo
echo ">> 阶段 2/3: models/ 权重清点"
INV=$("$PY" - "$MODELS_DIR" <<'PYEOF' 2>&1
import os, sys
md = sys.argv[1]
# 复用 runner 自带的 folder_paths shim
repo = os.path.dirname(os.path.dirname(os.path.abspath("$RUNNER_SCRIPT")))
cc = os.environ.get("COMFY_CORE_DIR", os.path.join(repo, "comfy_core"))
if cc not in sys.path:
    sys.path.insert(0, cc)
import folder_paths as fp
fp.MODELS_DIR = os.path.abspath(md)
for cat in ("clip", "vae", "diffusion_models", "loras"):
    files = fp.get_filename_list(cat)
    print(f"{cat}: {len(files)} 个 -> {files}")
PYEOF
)
echo "$INV" | sed 's/^/    /'
if echo "$INV" | grep -q "clip: 0"; then
  fail "clip 类权重为空（需上传 text_encoder）"
else
  ok "clip 类权重已就位"
fi
if echo "$INV" | grep -q "diffusion_models: 0"; then
  fail "diffusion_models 类权重为空（需上传 fl2va/ref2va transformer）"
else
  ok "diffusion_models 类权重已就位"
fi
if echo "$INV" | grep -q "vae: 0"; then
  fail "vae 类权重为空（需上传 video/audio VAE）"
else
  ok "vae 类权重已就位"
fi

# ---- 阶段 3：最小生成（可选） ----
if [ $DO_GENERATE -eq 1 ]; then
  echo
  echo ">> 阶段 3/3: 最小生成 (attention=$ATTENTION, ${WIDTH}x${HEIGHT}, steps=$STEPS)"
  REF_PNG="/tmp/smoke_ref.png"
  # 用标准库凭空生成 64x64 测试图（避免依赖 Pillow）
  "$PY" - "$REF_PNG" <<'PYEOF'
import zlib, struct, sys
path = sys.argv[1]; W=H=64
raw=bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        raw += bytes([(x*4)&255,(y*4)&255,((x+y)*2)&255])
def chunk(t,d):
    return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
png=b"\x89PNG\r\n\x1a\n"
png+=chunk(b"IHDR",struct.pack(">IIBBBBB",W,H,8,2,0,0,0))
png+=chunk(b"IDAT",zlib.compress(bytes(raw)))
png+=chunk(b"IEND",b"")
open(path,"wb").write(png)
print("测试图已生成:", path)
PYEOF

  OUT_MP4="/tmp/smoke_out.mp4"
  CMD=("$PY" "$RUNNER_SCRIPT" --prompt "$PROMPT" --mode reference \
       --ref-images "$REF_PNG" --width "$WIDTH" --height "$HEIGHT" \
       --seconds "$SECONDS_DUR" --fps "$FPS" --steps "$STEPS" \
       --attention "$ATTENTION" --output "$OUT_MP4")
  # 双阶段：写一个最小 pass2 配置，走 --pass2-config 分支
  if [ $DUAL_STAGE -eq 1 ]; then
    PASS2="/tmp/smoke_pass2.json"
    cat > "$PASS2" <<JSON
{
  "width": $((WIDTH*2)), "height": $((HEIGHT*2)),
  "steps": 1, "seconds": 2, "fps": $FPS,
  "mode": "reference"
}
JSON
    CMD+=("--pass2-config" "$PASS2")
    echo "    双阶段: 已写入 $PASS2，第二遍 ${WIDTH*2}x$((HEIGHT*2))"
  fi
  # 多 LoRA
  if [ -n "$LORA_SPECS" ]; then
    CMD+=("--loras" "$LORA_SPECS")
    echo "    多 LoRA: $LORA_SPECS"
  fi

  echo "    命令: ${CMD[*]}"
  GEN_OUT=$("${CMD[@]}" 2>&1)
  GRC=$?
  echo "$GEN_OUT" | tail -n 20 | sed 's/^/    /'
  if [ $GRC -eq 0 ] && [ -s "$OUT_MP4" ]; then
    ok "生成成功，输出: $OUT_MP4 ($(stat -c%s "$OUT_MP4" 2>/dev/null || echo '?') bytes)"
  else
    fail "生成失败 (rc=$GRC) 或输出为空；检查上方错误（多为权重未上传/文件名不符）"
  fi
else
  echo
  echo ">> 阶段 3/3: 跳过（未传 --generate）。要实跑全链路请加 --generate。"
fi

echo
echo "============================================================"
echo " 结果: $PASS 通过, $FAIL 失败"
echo "============================================================"
[ $FAIL -eq 0 ]
