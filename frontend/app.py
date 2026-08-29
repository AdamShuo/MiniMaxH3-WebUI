"""MVP WebUI (Gradio) for MiniMax-H3 video generation.

Talks to the backend REST API (M1~M5). Supports: structured prompt + template,
reference images / audio / video (9+3+3 with per-slot quote), multi-LoRA with
per-LoRA strength from models/loras, generation mode, a prompt optimizer toggle
(built-in CLIP vs third-party API with settings dialog), async submit with live
progress, and a results gallery.

Serving modes
-------------
- Mounted into the FastAPI backend at "/" (single-port cloud deploy): leave
  ``BACKEND_URL`` empty (default) so every API call uses a same-origin *relative*
  URL — the browser hits ``<instance>:7860/api/...`` and no second port/CORS is
  needed.
- Standalone dev: ``BACKEND_URL=http://localhost:8000 python app.py`` launches a
  separate Gradio server on 7860 that talks to a separately-running API.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import requests

# 空字符串 = 同源相对路径（挂载到后端 / 时使用，云端单端口访问）
BACKEND = os.getenv("BACKEND_URL", "").rstrip("/")
POLL_INTERVAL = 3

MODES = ["reference", "first_frame", "dual_stage"]
RESOLUTIONS = ["360P", "540P", "720P", "1080P"]
OPT_METHODS = [("内置clip优化", "builtin"), ("第三方API优化", "third_party")]
# 内置文本编码器：CLIP 或 text_encoding 下的 Qwen3VL 32B int8 convrot 权重
TEXT_ENCODERS = [
    ("CLIP", "clip"),
    ("Qwen3VL 32B (int8 convrot)",
     "text_encoding/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
]
# 8 个官方风格提示词模板（来自原版 BatchGenerator 内置灵感库）；选中后填入提示词框
STYLE_TEMPLATES = [
    ("柯基犬樱花", "一只可爱的柯基犬在樱花树下奔跑,阳光明媚,高清摄影,8K"),
    ("日落海景", "美丽的日落风景,金色阳光洒在海面上,波光粼粼"),
    ("高山湖泊", "高山湖泊,倒影清晰,蓝天白云,雪山环绕"),
    ("赛博朋克夜景", "城市夜景,霓虹闪烁,车流如织,赛博朋克风格"),
    ("森林小径", "森林小径,阳光斑驳,秋叶纷飞,宁静祥和"),
    ("星空银河", "梦幻的星空银河,宇宙深处,星云璀璨"),
    ("热带雨林", "热带雨林,瀑布飞流,绿色植被,生机勃勃"),
    ("雪山云海", "雪山之巅,云海翻腾,日出金光,壮丽景色"),
]
STYLE_PROMPT_MAP = {f"style_{i+1}": p for i, (_, p) in enumerate(STYLE_TEMPLATES)}
MAX_LORA = 6
MAX_IMAGE = 9
MAX_AUDIO = 3
MAX_VIDEO = 3

# 默认参数（对应原 Windows 界面默认值）
DEFAULTS = dict(
    step=8, seed=-1, width=1376, height=768, duration=10, fps=24,
    mode="dual_stage", resolution="360P", optimize_method="builtin",
)


def _get(path: str):
    return requests.get(f"{BACKEND}{path}", timeout=30)


def _post(path: str, json=None, files=None, data=None, params=None):
    return requests.post(f"{BACKEND}{path}", json=json, files=files,
                         data=data, params=params, timeout=60)


def _put(path: str, json=None):
    return requests.put(f"{BACKEND}{path}", json=json, timeout=30)


def upload_one(path: str, media_type: str | None = None) -> int | None:
    """上传单个文件到 /api/v1/assets，返回 asset id；失败返回 None。"""
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh, "application/octet-stream")}
            params = {"media_type": media_type} if media_type else None
            r = _post("/api/v1/assets", files=files, params=params)
            if r.ok:
                return r.json().get("id")
    except Exception:
        pass
    return None


def load_templates() -> list[dict]:
    """加载提示词模板列表，返回 [{label, key, is_scene, style}]。

    包含后端提供的 H3 通用 + 8 个官方场景指南，以及前端内置的 8 个官方风格
    提示词模板（选中后直接填入提示词框）。
    """
    items = []
    try:
        r = _get("/api/v1/prompt-templates")
        if r.ok:
            for t in r.json():
                key = t["template_key"]
                if key in ("none", "h3_general"):
                    continue
                label = f"{t['name_zh'] or t['name']} ({key})"
                items.append({
                    "label": label,
                    "key": key,
                    "is_scene": True,
                    "style": False,
                })
    except Exception:
        pass
    # 始终保留 H3 通用作为默认
    items.insert(0, {
        "label": "H3 通用 (h3_general)",
        "key": "h3_general",
        "is_scene": False,
        "style": False,
    })
    # 官方风格提示词模板（前端内置）
    for i, (name, prompt) in enumerate(STYLE_TEMPLATES, 1):
        items.append({
            "label": f"风格·{name}",
            "key": f"style_{i}",
            "is_scene": False,
            "style": True,
        })
    return items


def load_loras() -> list[str]:
    """加载 models/loras 下的可用 LoRA 文件名。"""
    try:
        r = _get("/api/v1/loras")
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []


def load_optimizer_settings():
    """获取已保存的第三方优化器设置。"""
    try:
        r = _get("/api/v1/settings/optimizer")
        if r.ok:
            s = r.json()
            return (
                s.get("api_format", "openai"),
                s.get("api_url", ""),
                s.get("api_key", ""),
                s.get("api_model", ""),
                s.get("scene_guide", "none"),
            )
    except Exception:
        pass
    return ("openai", "", "", "", "none")


def save_optimizer_settings(api_format, api_url, api_key, api_model, scene_guide):
    """保存第三方优化器设置。"""
    try:
        r = _put("/api/v1/settings/optimizer", json={
            "api_format": api_format,
            "api_url": api_url,
            "api_key": api_key,
            "api_model": api_model,
            "scene_guide": scene_guide,
        })
        msg = "✅ 已保存" if r.ok else f"❌ 保存失败: {r.text}"
    except Exception as e:
        msg = f"❌ 保存失败: {e}"
    return gr.update(value=msg, visible=True)


def test_optimizer_settings(api_url, api_key, api_model):
    """测试第三方 API 连通性。"""
    if not api_url or not api_key:
        msg = "⚠️ 请先填写 API 地址和 API Key"
    else:
        try:
            r = _post("/api/v1/settings/optimizer/test", json={
                "api_url": api_url,
                "api_key": api_key,
                "api_model": api_model,
            })
            if r.ok:
                data = r.json()
                if data.get("ok"):
                    msg = f"✅ 连接成功（{data.get('model') or api_model}）"
                else:
                    msg = f"❌ 连接失败: {data.get('error')}"
            else:
                msg = f"❌ 测试失败: {r.text}"
        except Exception as e:
            msg = f"❌ 测试失败: {e}"
    return gr.update(value=msg, visible=True)


def optimize(prompt, template_key, method):
    """点击「优化提示词」后的处理：第三方走后端 optimize，内置暂时原样返回。"""
    if not prompt or not prompt.strip():
        return prompt
    # 风格模板只负责填入提示词，不作为场景指南传给优化器
    scene_key = None if (template_key and template_key.startswith("style_")) else template_key
    if method != "third_party":
        # 内置 CLIP 优化：当前尚未接入真实 H3/CLIP 文本编码优化，原样返回。
        return prompt
    try:
        r = _post(f"/api/v1/prompt-templates/0/optimize",
                  json={"prompt": prompt, "template_key": scene_key})
        if r.ok:
            return r.json().get("optimized_prompt", prompt)
    except Exception:
        pass
    return prompt


def submit(prompt, template_key, image_paths, audio_paths, video_paths,
           mode, resolution, optimize_method, text_encoder,
           lora_names, lora_strengths,
           step, seed, width, height, duration, fps,
           progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return "⚠️ 请填写提示词", None, gr.update()

    # 风格模板只负责填入提示词，不作为场景指南传给后端
    backend_template_key = "none" if (template_key and template_key.startswith("style_")) else (template_key or "none")

    # 1) 上传参考素材（按小窗顺序）
    ref_ids, vid_ids = [], []
    for p in (image_paths or []):
        aid = upload_one(p, "image")
        if aid is not None:
            ref_ids.append(aid)
    for p in (audio_paths or []):
        aid = upload_one(p, "audio")
        if aid is not None:
            ref_ids.append(aid)
    for p in (video_paths or []):
        aid = upload_one(p, "video")
        if aid is not None:
            vid_ids.append(aid)

    # 2) 收集 LoRA（直接从模型目录下拉选择，按文件名传递）
    loras = []
    for name, strength in zip(lora_names, lora_strengths):
        if name:
            loras.append({"name": str(name), "strength": float(strength or 1.0)})
    if len(loras) > MAX_LORA:
        return f"⚠️ LoRA 最多 {MAX_LORA} 个", None, gr.update()

    # 3) 创建生成请求
    body = {
        "prompt": prompt,
        "template_key": backend_template_key,
        "reference_asset_ids": ref_ids,
        "video_asset_ids": vid_ids,
        "mode": mode,
        "first_stage_resolution": resolution,
        "optimize_method": optimize_method or "builtin",
        "text_encoder": text_encoder or "clip",
        "loras": loras,
        "step": int(step), "seed": int(seed), "width": int(width),
        "height": int(height), "duration": int(duration), "fps": int(fps),
    }
    r = _post("/api/v1/generations", json=body)
    if not r.ok:
        return f"❌ 创建失败: {r.text}", None, gr.update()
    gen_id = r.json()["id"]

    # 4) 提交 / 入队
    r = _post(f"/api/v1/generations/{gen_id}/submit")
    if not r.ok:
        return f"❌ 提交失败: {r.text}", None, gr.update()
    task_id = r.json()["id"]

    # 5) 轮询
    video_url = None
    status_msg = ""
    for _ in range(600):
        r = _get(f"/api/v1/tasks/{task_id}")
        if not r.ok:
            time.sleep(POLL_INTERVAL)
            continue
        t = r.json()
        pct = t.get("progress", 0)
        progress(pct / 100, f"{t['status']} {pct}%")
        status_msg = f"状态: {t['status']} ({pct}%)"
        if t["status"] == "SUCCEEDED":
            rid = t.get("result_id")
            if rid:
                video_url = f"{BACKEND}/api/v1/results/{rid}/download"
            break
        if t["status"] == "FAILED":
            status_msg = f"❌ 失败: {t.get('error_msg')}"
            break
        time.sleep(POLL_INTERVAL)
    return status_msg, video_url, gr.update(value=video_url) if video_url else gr.update()


def submit_from_ui(*args, progress=gr.Progress()):
    """将 UI 上分散的参考素材/LoRA 控件打包后调用 submit。"""
    prompt, tmpl = args[0], args[1]
    image_paths = [p for p in args[2:2 + MAX_IMAGE] if p]
    audio_paths = [p for p in args[2 + MAX_IMAGE:2 + MAX_IMAGE + MAX_AUDIO] if p]
    video_paths = [p for p in args[2 + MAX_IMAGE + MAX_AUDIO:2 + MAX_IMAGE + MAX_AUDIO + MAX_VIDEO] if p]
    base = 2 + MAX_IMAGE + MAX_AUDIO + MAX_VIDEO
    mode, resolution, optimize_method, text_encoder = (
        args[base], args[base + 1], args[base + 2], args[base + 3])
    lora_names = list(args[base + 4:base + 4 + MAX_LORA])
    lora_strengths = list(args[base + 4 + MAX_LORA:base + 4 + 2 * MAX_LORA])
    step, seed, width, height, duration, fps = args[base + 4 + 2 * MAX_LORA:base + 10 + 2 * MAX_LORA]
    return submit(prompt, tmpl, image_paths, audio_paths, video_paths,
                  mode, resolution, optimize_method, text_encoder,
                  lora_names, lora_strengths,
                  step, seed, width, height, duration, fps, progress)


def refresh_gallery():
    try:
        r = _get("/api/v1/results?page_size=50")
        if r.ok:
            items = r.json().get("items", [])
            return [f"{BACKEND}/api/v1/results/{it['id']}/download" for it in items]
    except Exception:
        pass
    return []


def reset_defaults():
    """恢复默认：返回所有参数控件默认值。"""
    return (
        DEFAULTS["mode"], DEFAULTS["resolution"], DEFAULTS["optimize_method"], "clip",
        *([None] * MAX_LORA), *([1.0] * MAX_LORA),
        DEFAULTS["step"], DEFAULTS["seed"], DEFAULTS["width"],
        DEFAULTS["height"], DEFAULTS["duration"], DEFAULTS["fps"],
    )


def _opt_label(method):
    return "第三方 API" if method == "third_party" else "内置 CLIP"


# ---------------------------------------------------------------------------
# 自定义 JS：在提示词输入框当前光标位置插入标记，替换选中文本
# ---------------------------------------------------------------------------
_INSERT_JS = """
<script>
function insertPromptMarker(marker) {
    const box = document.querySelector('#prompt-box textarea, #prompt-box input');
    if (!box) return marker;
    const start = box.selectionStart ?? 0;
    const end = box.selectionEnd ?? 0;
    const value = box.value || '';
    const newValue = value.substring(0, start) + marker + value.substring(end);
    box.value = newValue;
    const pos = start + marker.length;
    box.setSelectionRange(pos, pos);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    return newValue;
}
</script>
"""


def build_ui():
    templates = load_templates()
    tmpl_choices = [(t["label"], t["key"]) for t in templates]
    scene_choices = [("仅通用方案", "none")] + [
        (t["label"], t["key"]) for t in templates if t["is_scene"]
    ]
    loras = load_loras()

    with gr.Blocks(title="MiniMax-H3 WebUI") as demo:
        gr.HTML(_INSERT_JS)
        gr.Markdown("# MiniMax-H3 视频生成 WebUI\n文字 / 图片 + 音频 + 视频 → 带音频视频")

        with gr.Tabs():
            with gr.Tab("生成"):
                # ========== 上方：左输入 / 右参数 ==========
                with gr.Row():
                    with gr.Column(scale=2, elem_id="left-col"):
                        prompt = gr.Textbox(
                            label="正描述（支持 Ref2VA / FL2VA 结构）", lines=8,
                            placeholder=(
                                "subject_definitions: ...\n"
                                "summary: ...\n"
                                "retention_analysis: ...\n"
                                "detailed_description: ..."
                            ),
                            elem_id="prompt-box",
                        )

                        tmpl = gr.Dropdown(
                            label="提示词模板（可选）",
                            choices=tmpl_choices,
                            value="h3_general",
                            allow_custom_value=False,
                        )

                        with gr.Row():
                            optimize_method = gr.Dropdown(
                                label="提示词优化方式",
                                choices=OPT_METHODS,
                                value=DEFAULTS["optimize_method"],
                                allow_custom_value=False,
                            )
                            text_encoder = gr.Dropdown(
                                label="内置文本编码器",
                                choices=TEXT_ENCODERS,
                                value="clip",
                                allow_custom_value=False,
                                visible=(DEFAULTS["optimize_method"] == "builtin"),
                            )
                            api_settings_btn = gr.Button(
                                "第三方API设置", size="sm", visible=False)
                            optimize_btn = gr.Button("✨ 优化提示词")
                            submit_btn = gr.Button("🚀 开始生成", variant="primary")
                            reset_btn = gr.Button("↺ 恢复默认")

                        optimize_note = gr.Markdown(
                            "当前：内置 CLIP 优化（数据不出本机；未接入真实编码器前为原样返回）")

                        # 第三方 API 设置面板（默认隐藏，仅点击按钮或选第三方API时弹出，
                        # 切回内置优化即自动收起，避免 API 密钥常驻主界面导致外泄）
                        with gr.Column(visible=False, elem_id="api-settings-panel") as api_settings_panel:
                            gr.Markdown("### 提示词优化器设置")
                            api_format = gr.Dropdown(
                                label="API格式", choices=["openai"], value="openai")
                            api_url = gr.Textbox(label="API地址")
                            api_key = gr.Textbox(label="API Key", type="password")
                            api_model = gr.Textbox(label="模型名称")
                            api_scene_guide = gr.Dropdown(
                                label="场景指南", choices=scene_choices, value="none")
                            api_hint = gr.Markdown(
                                "提示：优化器会根据 MiniMax H3 提示词指南自动改写您的提示词，"
                                "使其更适合 H3 模型的输入格式，提升生成质量。")
                            with gr.Row():
                                api_test_btn = gr.Button("测试连接")
                                api_save_btn = gr.Button("保存", variant="primary")
                                api_cancel_btn = gr.Button("取消")
                            api_settings_msg = gr.Markdown(visible=False)

                        with gr.Accordion("生成模式与分辨率", open=True):
                            mode = gr.Dropdown(
                                label="优化及生成模式", choices=MODES,
                                value=DEFAULTS["mode"], allow_custom_value=False)
                            resolution = gr.Dropdown(
                                label="一阶段分辨率", choices=RESOLUTIONS,
                                value=DEFAULTS["resolution"], allow_custom_value=False)

                        with gr.Accordion("参考素材", open=True):
                            gr.Markdown("参考图（最多 9 张）")
                            image_refs = []
                            for row in range(3):
                                with gr.Row():
                                    for col in range(3):
                                        i = row * 3 + col
                                        with gr.Column(min_width=80):
                                            im = gr.Image(
                                                label=f"图{i+1}", height=80,
                                                type="filepath", sources=["upload"])
                                            btn = gr.Button("引用", size="sm")
                                            image_refs.append((im, btn))

                            gr.Markdown("参考音频（最多 3 个）")
                            audio_refs = []
                            with gr.Row():
                                for i in range(MAX_AUDIO):
                                    with gr.Column(min_width=120):
                                        au = gr.Audio(
                                            label=f"音频{i+1}",
                                            sources=["upload"], type="filepath")
                                        btn = gr.Button("引用", size="sm")
                                        audio_refs.append((au, btn))

                            gr.Markdown(f"参考视频（最多 {MAX_VIDEO} 个）")
                            video_refs = []
                            with gr.Row():
                                for i in range(MAX_VIDEO):
                                    with gr.Column(min_width=120):
                                        vd = gr.Video(
                                            label=f"视频{i+1}",
                                            sources=["upload"], height=100)
                                        btn = gr.Button("引用", size="sm")
                                        video_refs.append((vd, btn))

                    with gr.Column(scale=1, elem_id="right-col"):
                        with gr.Accordion(f"LoRA（最多 {MAX_LORA} 个）", open=True):
                            lora_rows = []
                            for i in range(MAX_LORA):
                                with gr.Row():
                                    ld = gr.Dropdown(
                                        label=f"LoRA {i+1}", choices=loras,
                                        value=None, allow_custom_value=True)
                                    ls = gr.Slider(
                                        0.0, 2.0, value=1.0, step=0.05,
                                        label=f"强度")
                                    lora_rows.append((ld, ls))

                        with gr.Accordion("参数", open=True):
                            step = gr.Number(
                                value=DEFAULTS["step"], label="步数", precision=0)
                            seed = gr.Number(
                                value=DEFAULTS["seed"], label="种子", precision=0)
                            width = gr.Number(
                                value=DEFAULTS["width"], label="宽", precision=0)
                            height = gr.Number(
                                value=DEFAULTS["height"], label="高", precision=0)
                            duration = gr.Number(
                                value=DEFAULTS["duration"], label="生成时长(s)",
                                precision=0)
                            fps = gr.Number(
                                value=DEFAULTS["fps"], label="帧率", precision=0)

                # ========== 下方：全横幅状态 + 生成结果 ==========
                with gr.Row():
                    with gr.Column():
                        status = gr.Textbox(
                            label="状态", interactive=False, lines=1)
                        video_out = gr.Video(label="生成结果")

                # ========== 事件绑定 ==========
                optimize_method.change(
                    lambda m: (
                        gr.update(visible=(m == "third_party")),  # 第三方API设置按钮
                        gr.update(visible=(m == "builtin")),      # 内置文本编码器
                        gr.update(visible=False) if m != "third_party" else gr.skip(),  # 切回内置时收起面板
                        f"当前：{_opt_label(m)} 优化"
                        + ("（未配置 API 时静默原样返回）" if m == "third_party" else "")
                    ),
                    [optimize_method],
                    [api_settings_btn, text_encoder, api_settings_panel, optimize_note],
                )

                # 选中官方风格提示词模板时，将对应提示词填入提示词框
                def _on_template_change(key):
                    if key and key.startswith("style_"):
                        return gr.update(value=STYLE_PROMPT_MAP.get(key, ""))
                    return gr.skip()
                tmpl.change(_on_template_change, [tmpl], [prompt])

                api_settings_btn.click(
                    lambda: gr.update(visible=True),
                    None,
                    api_settings_panel,
                )
                api_cancel_btn.click(
                    lambda: (gr.update(visible=False), gr.update(visible=False)),
                    None,
                    [api_settings_panel, api_settings_msg],
                )
                api_save_btn.click(
                    save_optimizer_settings,
                    [api_format, api_url, api_key, api_model, api_scene_guide],
                    api_settings_msg,
                )
                api_test_btn.click(
                    test_optimizer_settings,
                    [api_url, api_key, api_model],
                    api_settings_msg,
                )

                # 小窗「引用」按钮：通过 JS 在提示词框光标处插入标记
                for i, (im, btn) in enumerate(image_refs, 1):
                    btn.click(
                        fn=lambda: None,
                        inputs=None,
                        outputs=prompt,
                        js=f"() => insertPromptMarker('<Picture {i}>')",
                    )
                for i, (au, btn) in enumerate(audio_refs, 1):
                    btn.click(
                        fn=lambda: None,
                        inputs=None,
                        outputs=prompt,
                        js=f"() => insertPromptMarker('<Audio {i}>')",
                    )
                for i, (vd, btn) in enumerate(video_refs, 1):
                    btn.click(
                        fn=lambda: None,
                        inputs=None,
                        outputs=prompt,
                        js=f"() => insertPromptMarker('<Video {i}>')",
                    )

                optimize_btn.click(
                    optimize,
                    [prompt, tmpl, optimize_method],
                    [prompt],
                )

                submit_btn.click(
                    submit_from_ui,
                    [
                        prompt, tmpl,
                        *[c for c, _ in image_refs],
                        *[c for c, _ in audio_refs],
                        *[c for c, _ in video_refs],
                        mode, resolution, optimize_method, text_encoder,
                        *[c for c, _ in lora_rows],
                        *[c for _, c in lora_rows],
                        step, seed, width, height, duration, fps,
                    ],
                    [status, video_out, video_out],
                )

                reset_btn.click(
                    reset_defaults,
                    None,
                    [mode, resolution, optimize_method, text_encoder,
                     *[c for c, _ in lora_rows], *[c for _, c in lora_rows],
                     step, seed, width, height, duration, fps],
                )

                # 页面加载时刷新下拉选项/设置
                demo.load(
                    lambda: (
                        gr.update(choices=tmpl_choices),
                        *[gr.update(choices=loras) for _ in range(MAX_LORA)],
                    ),
                    None,
                    [tmpl, *[c for c, _ in lora_rows]],
                )
                demo.load(
                    load_optimizer_settings,
                    None,
                    [api_format, api_url, api_key, api_model, api_scene_guide],
                )

            with gr.Tab("画廊"):
                gallery = gr.Gallery(label="历史结果", columns=3, height="auto")
                refresh = gr.Button("🔄 刷新")
                refresh.click(refresh_gallery, None, gallery)
                demo.load(refresh_gallery, None, gallery)

    return demo


if __name__ == "__main__":
    # 独立开发时用：BACKEND_URL=http://localhost:8000 python app.py
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
