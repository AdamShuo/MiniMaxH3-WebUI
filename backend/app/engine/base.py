"""Engine adapter base — B1 (ComfyUI) / B2 (MiniMax) share this protocol.

An Engine turns a (prompt + params + reference media) request into a local video
file under the shared outputs volume, reporting progress via a callback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

ProgressCB = Callable[[int, Optional[str], Optional[str]], Awaitable[None]]


@dataclass
class GenParams:
    step: int = 8
    seed: int = -1
    width: int = 1376
    height: int = 768
    duration: int = 10
    fps: int = 24
    lora_name: str = ""
    lora_strength: float = 1.0
    resolution: str = "360P"


class EngineResult(BaseModel):
    file_path: str  # absolute path on the shared outputs volume
    engine: str
    duration: Optional[int] = None
    thumbnail: Optional[str] = None


class BaseEngine(ABC):
    name: str = "base"

    @abstractmethod
    async def health(self) -> bool:
        ...

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        params: GenParams,
        reference_paths: list[str],
        audio_paths: list[str],
        progress_cb: ProgressCB,
        task_id: int,
    ) -> EngineResult:
        ...
