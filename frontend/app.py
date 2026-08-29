"""MVP WebUI (Gradio) for MiniMax-H3 video generation.

Talks to the backend REST API (M1~M5). Supports: structured prompt + template,
reference images / audio / video, multi-LoRA with per-LoRA strength, generation
mode (reference / first_frame / dual_stage), first-stage resolution, a prompt
optimization toggle (同实例 H3 文本编码 vs 第三方 API), async submit with live
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
OPT_METHODS = ["builtin", "third_party"]
MAX_LORA = 6
MAX_VIDEO = 3

# 默认参数（对应原 Windows 界面默认值）
DEFAULTS = dict(
    step=8, seed=-1, width=1376, height=768, duration=10, fps=24,
    mode="reference", resolution="360P", optimize_method="builtin",
)

# LoRA 控件引用（运行时填充，便于 submit 收集与恢复默认）
_LORA_FILES: list[gr.File] = []
_LORA_STRENGTHS: list[gr.Slider] = []


def _get(path: str):
    return requests.get(f"{BACKEND}{path}", timeout=30)


def _post(path: str, json=None, files=None, data=None, params=None):
    return requests.post(f"{BACKEND}{path}", json=json, files=files,
                         data=data, params=params, timeout=60)


def upload_one(path: str, media_type: str | None = None) -> int | None:
    """上传单个文件到 /api/v1/assets，返回 asset id；失败返回 None。"""
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


def load_templates():
    try:
        r = _get("/api/v1/prompt-templates")
        if r.ok:
            return {f"{t['name_zh'] or t['name']} ({t['template_key']})": t["id"]
                    for t in r.json()}
    except Exception:
        pass
    return {"（不使用模板）": 0}


def optimize(prompt, template_id):
    if not prompt.strip():
        return prompt
    try:
        r = _post(f"/api/v1/prompt-templates/{template_id or 0}/optimize",
                  json={"prompt": prompt})
        if r.ok:
            return r.json().get("optimized_prompt", prompt)
    except Exception:
        pass
    return prompt


def submit(prompt, images, audio, video, mode, resolution, optimize_method,
           lf1, ls1, lf2, ls2, lf3, ls3, lf4, ls4, lf5, ls5, lf6, ls6,
           template_label, step, seed, width, height, duration, fps,
           progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return "⚠️ 请填写提示词", None, gr.update()

    lora_pairs = [(lf1, ls1), (lf2, ls2), (lf3, ls3),
                  (lf4, ls4), (lf5, ls5), (lf6, ls6)]

    # 1) 上传参考素材
    ref_ids, vid_ids, loras = [], [], []
    for im in (images or []):
        aid = upload_one(im.name, "image")
        if aid is not None:
            ref_ids.append(aid)
    for au in (audio or []):
        aid = upload_one(au.name, "audio")
        if aid is not None:
            ref_ids.append(aid)
    for vd in (video or []):
        aid = upload_one(vd.name, "video")
        if aid is not None:
            vid_ids.append(aid)
    if len(vid_ids) > MAX_VIDEO:
        return f"⚠️ 参考视频最多 {MAX_VIDEO} 个", None, gr.update()

    for f, s in lora_pairs:
        if f is not None:
            aid = upload_one(f.name, "lora")
            if aid is not None:
                loras.append({"asset_id": aid, "strength": float(s or 1.0)})
    if len(loras) > MAX_LORA:
        return f"⚠️ LoRA 最多 {MAX_LORA} 个", None, gr.update()

    # 2) 创建生成请求（含提示词优化方式选择）
    body = {
        "prompt": prompt,
        "reference_asset_ids": ref_ids,
        "video_asset_ids": vid_ids,
        "mode": mode,
        "first_stage_resolution": resolution,
        "optimize_method": optimize_method or "builtin",
        "loras": loras,
        "step": int(step), "seed": int(seed), "width": int(width),
        "height": int(height), "duration": int(duration), "fps": int(fps),
    }
    r = _post("/api/v1/generations", json=body)
    if not r.ok:
        return f"❌ 创建失败: {r.text}", None, gr.update()
    gen_id = r.json()["id"]

    # 3) 提交 / 入队
    r = _post(f"/api/v1/generations/{gen_id}/submit")
    if not r.ok:
        return f"❌ 提交失败: {r.text}", None, gr.update()
    task_id = r.json()["id"]

    # 4) 轮询
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


def refresh_gallery():
    try:
        r = _get("/api/v1/results?page_size=50")
        if r.ok:
            items = r.json().get("items", [])
            return [f"{BACKEND}/api/v1/results/{it['id']}/download" for it in items]
    except Exception:
        pass
    return []


# 恢复默认：返回所有参数控件默认值
def reset_defaults():
    return (DEFAULTS["mode"], DEFAULTS["resolution"], DEFAULTS["optimize_method"],
            *([None] * MAX_LORA), *([1.0] * MAX_LORA),
            DEFAULTS["step"], DEFAULTS["seed"], DEFAULTS["width"],
            DEFAULTS["height"], DEFAULTS["duration"], DEFAULTS["fps"])


def _on_opt_method(method):
    if method == "third_party":
        return (gr.update(interactive=True),
                "使用第三方 LLM API 优化提示词（需配置 LLM_API_URL / LLM_API_KEY；"
                "未配置时该按钮静默降级为原样返回）")
    return (gr.update(interactive=False),
            "使用同实例 H3 文本编码模型优化（不经第三方，数据不出本机）")


def build_ui():
    with gr.Blocks(title="MiniMax-H3 WebUI") as demo:
        gr.Markdown("# MiniMax-H3 视频生成 WebUI\n文字 / 图片 + 音频 + 视频 → 带音频视频")
        with gr.Tabs():
            with gr.Tab("生成"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="正描述（支持 Ref2VA / FL2VA 结构）", lines=8,
                            placeholder="subject_definitions: ...\nsummary: ...\nretention_analysis: ...\ndetailed_description: ...")
                        tmpl = gr.Dropdown(label="场景模板（可选）", choices=[],
                                           value=None, allow_custom_value=True)

                        with gr.Row():
                            optimize_method = gr.Dropdown(
                                label="提示词优化方式", choices=OPT_METHODS,
                                value=DEFAULTS["optimize_method"], allow_custom_value=False)
                            optimize_btn = gr.Button("✨ 优化提示词")
                            submit_btn = gr.Button("🚀 开始生成", variant="primary")
                            reset_btn = gr.Button("↺ 恢复默认")
                        optimize_note = gr.Markdown(
                            "当前：使用同实例 H3 文本编码模型（不经第三方，数据不出本机）")
                        optimize_method.change(
                            _on_opt_method, [optimize_method],
                            [optimize_btn, optimize_note])

                        with gr.Accordion("生成模式与分辨率", open=True):
                            mode = gr.Dropdown(label="优化及生成模式", choices=MODES,
                                               value=DEFAULTS["mode"], allow_custom_value=False)
                            resolution = gr.Dropdown(label="一阶段分辨率", choices=RESOLUTIONS,
                                                     value=DEFAULTS["resolution"], allow_custom_value=False)

                        with gr.Accordion("参考素材", open=True):
                            images = gr.File(label="参考图（最多 9 张）",
                                             file_types=["image"], file_count="multiple")
                            audio = gr.File(label="参考音频（可选）",
                                            file_types=["audio"], file_count="multiple")
                            video = gr.File(label=f"参考视频（最多 {MAX_VIDEO} 个）",
                                            file_types=["video"], file_count="multiple")

                        with gr.Accordion(f"LoRA（最多 {MAX_LORA} 个，带强度）", open=False):
                            for i in range(MAX_LORA):
                                with gr.Row():
                                    lf = gr.File(label=f"LoRA {i+1}",
                                                 file_types=[".safetensors", ".pt", ".ckpt", ".bin"],
                                                 file_count="single")
                                    ls = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                                   label=f"强度 {i+1}")
                                    _LORA_FILES.append(lf)
                                    _LORA_STRENGTHS.append(ls)

                        with gr.Accordion("参数", open=False):
                            step = gr.Number(value=DEFAULTS["step"], label="步数", precision=0)
                            seed = gr.Number(value=DEFAULTS["seed"], label="种子", precision=0)
                            width = gr.Number(value=DEFAULTS["width"], label="宽", precision=0)
                            height = gr.Number(value=DEFAULTS["height"], label="高", precision=0)
                            duration = gr.Number(value=DEFAULTS["duration"], label="生成时长(s)", precision=0)
                            fps = gr.Number(value=DEFAULTS["fps"], label="帧率", precision=0)

                    with gr.Column():
                        status = gr.Textbox(label="状态", interactive=False)
                        video_out = gr.Video(label="生成结果")

                tmpl_js = gr.JSON(visible=False)
                demo.load(lambda: gr.update(choices=list(load_templates().keys())),
                          outputs=tmpl)
                tmpl.change(lambda x: x, tmpl, tmpl_js, queue=False)
                optimize_btn.click(lambda p: optimize(p, 0), [prompt], [prompt])

                submit_btn.click(
                    submit,
                    [prompt, images, audio, video, mode, resolution, optimize_method,
                     *[c for pair in zip(_LORA_FILES, _LORA_STRENGTHS) for c in pair],
                     tmpl, step, seed, width, height, duration, fps],
                    [status, video_out, video_out],
                )
                reset_btn.click(
                    reset_defaults,
                    None,
                    [mode, resolution, optimize_method,
                     *_LORA_FILES, *_LORA_STRENGTHS,
                     step, seed, width, height, duration, fps],
                )

            with gr.Tab("画廊"):
                gallery = gr.Gallery(label="历史结果", columns=3, height="auto")
                refresh = gr.Button("🔄 刷新")
                refresh.click(refresh_gallery, None, gallery)
                demo.load(refresh_gallery, None, gallery)

    return demo


demo = build_ui()

if __name__ == "__main__":
    # 独立开发时用：BACKEND_URL=http://localhost:8000 python app.py
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
