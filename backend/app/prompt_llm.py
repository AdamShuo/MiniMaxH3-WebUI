"""B3 — 提示词优化 LLM 适配 (Conformist, OpenAI-compatible).

Optimizes/rewrites the user prompt into a full H3 prompt. Strictly follows the
OpenAI chat/completions schema. On any failure it degrades gracefully (BD-01):
the caller keeps the original prompt.
"""
from __future__ import annotations

import httpx

from .config import settings


SYSTEM_PROMPT = (
    "You are an expert prompt engineer for the MiniMax-H3 video generation model. "
    "Rewrite the user's brief into a detailed, cinematic H3 prompt. Preserve the "
    "subject, motion, camera, lighting, soundscape and any reference anchors. "
    "Output only the final prompt text."
)


async def optimize_prompt(prompt: str, model: str | None = None,
                          overrides: dict | None = None) -> str:
    """优化提示词；overrides 优先于环境变量设置。"""
    ov = overrides or {}
    api_url = ov.get("api_url") or settings.llm_api_url
    api_key = ov.get("api_key") or settings.llm_api_key
    model = model or ov.get("api_model") or settings.llm_model
    if not api_url or not api_key:
        return prompt  # B3 optional — degrade
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers, json=body,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return prompt  # BD-01 degrade
