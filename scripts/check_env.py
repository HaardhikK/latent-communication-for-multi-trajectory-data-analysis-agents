from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.cloud import detect_cloud_platform  # noqa: E402
from latent_agent.runtime import configure_runtime, path_is_onedrive  # noqa: E402


def main() -> int:
    runtime = configure_runtime(create=True)
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "pandas",
        "numpy",
        "sklearn",
        "psutil",
        "pynvml",
        "pytest",
        "bitsandbytes",
    ]
    platform_defaults = detect_cloud_platform()
    info: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "detected_platform": platform_defaults.name,
        "runtime_root": str(runtime),
        "runtime_on_onedrive": path_is_onedrive(runtime),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "packages": {},
    }

    ok = True
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            info["packages"][package] = {"ok": True, "version": version}
        except Exception as exc:
            if package != "bitsandbytes":
                ok = False
            info["packages"][package] = {"ok": False, "error": repr(exc)}

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info["torch_cuda_available"] = cuda_available
        if cuda_available:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_vram_mb"] = int(props.total_memory / (1024 * 1024))
        else:
            ok = False
    except Exception as exc:
        ok = False
        info["torch_cuda_error"] = repr(exc)

    if info["runtime_on_onedrive"]:
        ok = False

    print(json.dumps(info, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
