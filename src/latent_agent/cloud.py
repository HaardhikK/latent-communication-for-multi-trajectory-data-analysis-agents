from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLOUD_DEPENDENCY_PINS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
    "transformers": "4.53.3",
    "accelerate": "1.8.1",
    "bitsandbytes": "0.46.1",
}

TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"


@dataclass(frozen=True)
class CloudPlatform:
    name: str
    runtime_root: Path
    hf_cache_root: Path


def detect_cloud_platform() -> CloudPlatform:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        runtime = Path("/kaggle/working/latent-agent-runtime")
        hf_cache = Path("/kaggle/temp/hf-cache") if Path("/kaggle/temp").exists() else runtime / "hf-cache"
        return CloudPlatform("kaggle", runtime, hf_cache)
    if os.environ.get("COLAB_GPU") or Path("/content").exists():
        return CloudPlatform("colab", Path("/content/latent-agent-runtime"), Path("/content/hf-cache"))
    if Path("/mnt/c").exists():
        runtime = Path.home() / "latent-agent-runtime"
        return CloudPlatform("wsl", runtime, runtime / "hf-cache")
    return CloudPlatform("generic", Path.home() / "latent-agent-runtime", Path.home() / "latent-agent-runtime" / "hf-cache")


def force_single_gpu_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def version_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    for package in CLOUD_DEPENDENCY_PINS:
        try:
            version = importlib.metadata.version(package)
            importlib.import_module(package)
            report["packages"][package] = {"ok": True, "version": version, "required": CLOUD_DEPENDENCY_PINS[package]}
        except Exception as exc:
            report["packages"][package] = {"ok": False, "version": None, "required": CLOUD_DEPENDENCY_PINS[package], "error": repr(exc)}
    return report


def assert_cloud_dependencies(*, strict_versions: bool = True) -> dict[str, Any]:
    report = version_report()
    failures = []
    for package, info in report["packages"].items():
        if not info["ok"]:
            failures.append(f"{package}: import failed ({info.get('error')})")
            continue
        installed = str(info["version"]).split("+", 1)[0]
        required = str(info["required"])
        if strict_versions and installed != required:
            failures.append(f"{package}: installed {info['version']}, required {required}")
    report["ok"] = not failures
    report["failures"] = failures
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Cloud dependency guard failed:\n{details}\nResolved versions: {report['packages']}")
    return report


def gpu_report(torch_module: Any | None = None) -> dict[str, Any]:
    torch = torch_module or importlib.import_module("torch")
    report: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [],
        "ok": False,
    }
    if not report["cuda_available"]:
        return report
    for index in range(report["visible_device_count"]):
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        report["devices"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": [int(capability[0]), int(capability[1])],
                "total_vram_mb": int(props.total_memory / (1024 * 1024)),
            }
        )
    report["ok"] = len(report["devices"]) == 1 and gpu_is_supported(report["devices"][0]["name"], tuple(report["devices"][0]["capability"]))
    return report


def gpu_is_supported(name: str, capability: tuple[int, int]) -> bool:
    lowered = name.lower()
    major, minor = capability
    return "t4" in lowered or major >= 8


def assert_supported_single_gpu(torch_module: Any | None = None) -> dict[str, Any]:
    report = gpu_report(torch_module)
    if not report["cuda_available"]:
        raise RuntimeError(f"CUDA is not available. GPU report: {report}")
    if report["visible_device_count"] != 1:
        raise RuntimeError(
            "Tier 2 requires exactly one visible GPU. "
            "Set CUDA_VISIBLE_DEVICES=0 before importing torch/model code. "
            f"GPU report: {report}"
        )
    device = report["devices"][0]
    if not gpu_is_supported(device["name"], tuple(device["capability"])):
        raise RuntimeError(
            "Unsupported GPU for this 4-bit hidden-state gate. "
            "Use NVIDIA T4 or Ampere+; P100/Pascal-class GPUs are rejected because bnb 4-bit kernels are unreliable here. "
            f"GPU report: {report}"
        )
    return report


def analyze_latent_signal(
    *,
    normal_text: str,
    ablated_text: str,
    normal_top_token_id: int,
    ablated_top_token_id: int,
    logit_max_abs_diff: float,
    logit_mean_abs_diff: float,
    threshold: float,
) -> dict[str, Any]:
    text_diverged = normal_text.strip() != ablated_text.strip()
    top_token_diverged = int(normal_top_token_id) != int(ablated_top_token_id)
    logits_diverged = float(logit_max_abs_diff) >= float(threshold)
    ok = bool(logits_diverged and (text_diverged or top_token_diverged or float(logit_mean_abs_diff) >= float(threshold)))
    return {
        "ok": ok,
        "threshold": float(threshold),
        "text_diverged": text_diverged,
        "top_token_diverged": top_token_diverged,
        "logits_diverged": logits_diverged,
        "normal_top_token_id": int(normal_top_token_id),
        "ablated_top_token_id": int(ablated_top_token_id),
        "logit_max_abs_diff": float(logit_max_abs_diff),
        "logit_mean_abs_diff": float(logit_mean_abs_diff),
    }


def summarize_gate(records: list[Any], *, tolerance: float = 0.15) -> dict[str, Any]:
    by_mode: dict[str, list[Any]] = {}
    by_family_mode: dict[tuple[str, str], list[Any]] = {}
    for record in records:
        by_mode.setdefault(record.mode, []).append(record)
        by_family_mode.setdefault((record.task_family, record.mode), []).append(record)

    def rate(items: list[Any]) -> float:
        return sum(bool(item.passed) for item in items) / len(items) if items else 0.0

    a = by_mode.get("A_single", [])
    b = by_mode.get("B_textmas", [])
    c = by_mode.get("C_latentmas", [])
    a_rate = rate(a)
    b_rate = rate(b)
    c_rate = rate(c)
    a_b_gap = abs(a_rate - b_rate)
    valid = bool(a and b and b_rate > 0 and a_b_gap <= tolerance)
    family_rows = []
    for family in sorted({getattr(record, "task_family", "") for record in records}):
        for mode in ("A_single", "B_textmas", "C_latentmas"):
            items = by_family_mode.get((family, mode), [])
            family_rows.append(
                {
                    "family": family,
                    "mode": mode,
                    "runs": len(items),
                    "passes": sum(bool(item.passed) for item in items),
                    "pass_rate": rate(items),
                }
            )
    return {
        "valid_text_baseline": valid,
        "tolerance": float(tolerance),
        "a_runs": len(a),
        "b_runs": len(b),
        "c_runs": len(c),
        "a_pass_rate": a_rate,
        "b_pass_rate": b_rate,
        "c_pass_rate": c_rate,
        "a_b_gap": a_b_gap,
        "family_rows": family_rows,
        "coarse_screen_note": "15 runs/mode gives pass-rate resolution of about 0.067; this is a go/no-go screen, not final accuracy proof.",
    }
