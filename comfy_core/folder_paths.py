"""Minimal folder_paths shim for headless MiniMax H3 execution.

The app-root ``folder_paths`` module is sealed inside ``capai.exe``; this
reimplements only the surface the vendored H3 nodes actually touch
(``get_filename_list``, ``folder_names_and_paths``, ``get_input/output/temp_directory``).

Model categories are resolved under ``$MODELS_DIR`` (default: ``<repo>/models``).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models"))
MODELS_DIR = os.path.abspath(MODELS_DIR)

DATA_DIR = os.path.join(REPO_ROOT, "data")
INPUT_DIR = os.environ.get("INPUT_DIR", os.path.join(DATA_DIR, "uploads"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(DATA_DIR, "outputs"))
TEMP_DIR = os.environ.get("TEMP_DIR", os.path.join(DATA_DIR, "temp"))

SUPPORTED_TYPES = [".pt", ".pth", ".bin", ".safetensors", ".gguf", ".ckpt"]

# category -> (list of directories, list of accepted extensions)
folder_names_and_paths = {
    "unet": ([os.path.join(MODELS_DIR, "unet"), os.path.join(MODELS_DIR, "diffusion_models")], SUPPORTED_TYPES),
    "diffusion_models": ([os.path.join(MODELS_DIR, "unet"), os.path.join(MODELS_DIR, "diffusion_models")], SUPPORTED_TYPES),
    "vae": ([os.path.join(MODELS_DIR, "vae")], SUPPORTED_TYPES),
    "clip": ([os.path.join(MODELS_DIR, "clip")], SUPPORTED_TYPES),
    "loras": ([os.path.join(MODELS_DIR, "loras")], SUPPORTED_TYPES),
    "input": ([INPUT_DIR], [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".mp4", ".mov", ".wav", ".mp3", ".flac"]),
    "output": ([OUTPUT_DIR], []),
    "temp": ([TEMP_DIR], []),
}

supported_types = set(SUPPORTED_TYPES)


def get_folder_paths(folder_name):
    paths, _exts = folder_names_and_paths.get(folder_name, ([], []))
    return paths


def get_filename_list(folder_name):
    folders, exts = folder_names_and_paths.get(folder_name, ([], []))
    out = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            full = os.path.join(folder, fn)
            if os.path.isfile(full) and (not exts or any(fn.lower().endswith(e) for e in exts)):
                out.append(fn)
    return sorted(out)


def get_full_path(folder_name, filename):
    """Resolve a model name to an absolute path under the category's folders."""
    folders, _exts = folder_names_and_paths.get(folder_name, ([], []))
    for folder in folders:
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate
    # already an absolute / relative path on disk?
    if os.path.isfile(filename):
        return os.path.abspath(filename)
    return None


def get_input_directory():
    return INPUT_DIR


def get_output_directory():
    return OUTPUT_DIR


def get_temp_directory():
    return TEMP_DIR


def init_folders():
    """No-op for headless: directories are created on demand by the runner."""
    for d in (INPUT_DIR, OUTPUT_DIR, TEMP_DIR,
              os.path.join(MODELS_DIR, "unet"),
              os.path.join(MODELS_DIR, "vae"),
              os.path.join(MODELS_DIR, "clip"),
              os.path.join(MODELS_DIR, "loras")):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


# some comfy modules read these module-level globals directly
models_dir = MODELS_DIR
supported_suffixes = SUPPORTED_TYPES[:]
