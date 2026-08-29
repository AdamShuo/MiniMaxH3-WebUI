from .assets import router as assets_router
from .generations import router as generations_router
from .loras import router as loras_router
from .results import router as results_router
from .settings import router as settings_router
from .tasks import router as tasks_router
from .templates import router as templates_router

__all__ = [
    "assets_router",
    "templates_router",
    "generations_router",
    "tasks_router",
    "results_router",
    "loras_router",
    "settings_router",
]
