"""MVP WebUI (Gradio) for MiniMax-H3 video generation.

Talks to the backend REST API (M1~M5). Supports: upload references, prompt +
template, optional LLM prompt optimization, async submit with live progress, and a
results gallery with playback/download.
"""
from __future__ import annotations

import os
import time

import gradio as gr
import requests

BACKEND = os.getenv("BACKEND_URL", "http://api:8000").rstrip("/")
POLL_INTERVAL = 3


def _get(path: str):
    return requests.get(f"{BACKEND}{path}", timeout=30)


def _post(path: str, json=None, files=None, data=None):
    return requests.post(f"{BACKEND}{path}", json=json, files=files, data=data, timeout=60)


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


def submit(prompt, images, audio, template_label, step, seed, width, height,
           duration, fps, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        return "⚠️ 请填写提示词", None, gr.update()

    # 1) upload references
    ref_ids = []
    files = []
    if images:
        for im in (images if isinstance(images, list) else [images]):
            files.append(("file", (im.name, open(im.name, "rb"), "image/*")))
    if audio:
        for au in (audio if isinstance(audio, list) else [audio]):
            files.append(("file", (au.name, open(au.name, "rb"), "audio/*")))
    for f in files:
        r = requests.post(f"{BACKEND}/api/v1/assets", files={"file": f[1]}, timeout=60)
        if r.ok:
            ref_ids.append(r.json()["id"])

    # 2) create generation request
    body = {
        "prompt": prompt, "reference_asset_ids": ref_ids,
        "step": int(step), "seed": int(seed), "width": int(width),
        "height": int(height), "duration": int(duration), "fps": int(fps),
    }
    r = _post("/api/v1/generations", json=body)
    if not r.ok:
        return f"❌ 创建失败: {r.text}", None, gr.update()
    gen_id = r.json()["id"]

    # 3) submit / enqueue
    r = _post(f"/api/v1/generations/{gen_id}/submit")
    if not r.ok:
        return f"❌ 提交失败: {r.text}", None, gr.update()
    task_id = r.json()["id"]

    # 4) poll
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


with gr.Blocks(title="MiniMax-H3 WebUI") as demo:
    gr.Markdown("# MiniMax-H3 视频生成 WebUI\n文字 / 图片 + 音频 → 带音频视频")
    with gr.Tabs():
        with gr.Tab("生成"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="提示词 (支持 Ref2VA / FL2VA 结构)", lines=8,
                                         placeholder="subject_definitions: ...\nsummary: ...")
                    tmpl = gr.Dropdown(label="场景模板（可选）", choices=[],
                                      value=None, allow_custom_value=True)
                    with gr.Row():
                        optimize_btn = gr.Button("✨ 优化提示词")
                        submit_btn = gr.Button("🚀 提交生成", variant="primary")
                    with gr.Accordion("参数", open=False):
                        step = gr.Number(value=8, label="step", precision=0)
                        seed = gr.Number(value=-1, label="seed", precision=0)
                        width = gr.Number(value=1376, label="width", precision=0)
                        height = gr.Number(value=768, label="height", precision=0)
                        duration = gr.Number(value=10, label="duration(s)", precision=0)
                        fps = gr.Number(value=24, label="fps", precision=0)
                    images = gr.File(label="参考图（首帧/末帧等，最多 5 张）",
                                    file_types=["image"], file_count="multiple")
                    audio = gr.File(label="参考音频（可选）", file_types=["audio"],
                                    file_count="multiple")
                with gr.Column():
                    status = gr.Textbox(label="状态", interactive=False)
                    video = gr.Video(label="生成结果")
            tmpl_js = gr.JSON(visible=False)
            demo.load(lambda: gr.update(choices=list(load_templates().keys())),
                      outputs=tmpl)
            tmpl.change(lambda x: x, tmpl, tmpl_js, queue=False)
            optimize_btn.click(lambda p: optimize(p, 0), [prompt], [prompt])
            submit_btn.click(
                submit,
                [prompt, images, audio, tmpl, step, seed, width, height, duration, fps],
                [status, video, video],
            )
        with gr.Tab("画廊"):
            gallery = gr.Gallery(label="历史结果", columns=3, height="auto")
            refresh = gr.Button("🔄 刷新")
            refresh.click(refresh_gallery, None, gallery)
            demo.load(refresh_gallery, None, gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
