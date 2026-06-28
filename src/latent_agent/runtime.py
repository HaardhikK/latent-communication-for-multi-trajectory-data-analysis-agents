from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .cloud import detect_cloud_platform

DEFAULT_RUNTIME_ROOT = detect_cloud_platform().runtime_root


def runtime_root() -> Path:
    return Path(os.environ.get("LATENT_AGENT_RUNTIME", str(DEFAULT_RUNTIME_ROOT))).expanduser()


def configure_runtime(create: bool = True) -> Path:
    platform_defaults = detect_cloud_platform()
    root = runtime_root()
    os.environ.setdefault("LATENT_AGENT_RUNTIME", str(root))
    hf_home = Path(os.environ.get("HF_HOME", str(platform_defaults.hf_cache_root))).expanduser()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _import_windows_hf_token_if_needed()

    if create:
        for subdir in (
            root,
            hf_home,
            Path(os.environ["HF_HUB_CACHE"]),
            Path(os.environ["HF_DATASETS_CACHE"]),
            root / "datasets",
            root / "runs",
            root / "logs",
            root / "tmp",
            root / "exports",
        ):
            subdir.mkdir(parents=True, exist_ok=True)
    return root


def _import_windows_hf_token_if_needed() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not powershell.exists():
        return
    try:
        completed = subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-Command",
                "[Environment]::GetEnvironmentVariable('HF_TOKEN','User')",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return
    token = completed.stdout.strip()
    if token:
        os.environ["HF_TOKEN"] = token


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def runtime_path(*parts: str) -> Path:
    return runtime_root().joinpath(*parts)


def path_is_onedrive(path: Path) -> bool:
    return "onedrive" in str(path).lower()
