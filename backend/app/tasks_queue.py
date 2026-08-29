"""RQ task queue integration (M4 — 前后端分离 + 队列/Worker)."""
from __future__ import annotations

from redis import Redis
from rq import Queue

from .config import settings
from .worker import run_generation

_q: Queue | None = None


def get_queue() -> Queue:
    global _q
    if _q is None:
        conn = Redis.from_url(settings.redis_url, decode_responses=False)
        _q = Queue("default", connection=conn)
    return _q


def enqueue_generation(task_id: int) -> None:
    get_queue().enqueue(run_generation, task_id, job_timeout=1800)
