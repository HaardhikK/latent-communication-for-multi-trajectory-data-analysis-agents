from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.cloud import assert_cloud_dependencies, assert_supported_single_gpu, force_single_gpu_env, gpu_report  # noqa: E402
from latent_agent.metrics import write_json  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tier 2 Phase 4A-lite forensic pilot in resumable chunks.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--variants", default="C1_phase3_exact,C2_dedup,C3_no_latent,C5_anchor")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--experiment-part", default="session2")
    parser.add_argument("--families", default="orders_kpi,sensor_quality,campaign_roi")
    parser.add_argument("--horizons", default="long")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--base-generation-seed", type=int, default=17)
    parser.add_argument("--schedule-seed", type=int, default=1701)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--fallback-latent-steps", type=int, default=2)
    parser.add_argument("--latent-observation-steps", type=int, default=1)
    parser.add_argument("--latent-repair-strategy", choices=["text_reset", "latent"], default="text_reset")
    parser.add_argument("--max-new-rows", type=int, default=18)
    parser.add_argument("--resume-zip", default="")
    parser.add_argument("--result-zip", default="")
    parser.add_argument("--output-prefix", default="tier2_phase4_lite")
    parser.add_argument("--strict-cloud-deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-dependency-guard", action="store_true")
    parser.add_argument("--skip-gpu-guard", action="store_true")
    parser.add_argument("--skip-plumbing-gates", action="store_true")
    parser.add_argument("--strict-reuse", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_single_gpu_env()
    runtime = configure_runtime(create=True)
    if args.resume_zip:
        restored = restore_checkpoint(args.resume_zip)
        print(f"PHASE4_RESTORED_CHECKPOINT {restored}", flush=True)

    import torch

    dependency_report = None
    if not args.skip_dependency_guard:
        dependency_report = assert_cloud_dependencies(strict_versions=args.strict_cloud_deps)
    gpu = gpu_report(torch)
    if not args.skip_gpu_guard and args.quantization == "4bit":
        gpu = assert_supported_single_gpu(torch)
    write_json(
        project_path("exports", "tier2_phase4_environment.json"),
        {
            "runtime_root": str(runtime),
            "dependency_report": dependency_report,
            "gpu_report": gpu,
            "pip_freeze_path": str(write_pip_freeze()),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    effective_latent_steps = args.latent_steps
    if not args.skip_plumbing_gates:
        effective_latent_steps = run_plumbing_gates(args)

    resume_csv = latest_partial_or_metrics(args.output_prefix)
    command = [
        sys.executable,
        "scripts/run_phase4.py",
        "--model",
        args.model,
        "--quantization",
        args.quantization,
        "--variants",
        args.variants,
        "--families",
        args.families,
        "--horizons",
        args.horizons,
        "--repeat",
        str(args.repeat),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--base-generation-seed",
        str(args.base_generation_seed),
        "--schedule-seed",
        str(args.schedule_seed),
        "--latent-steps",
        str(effective_latent_steps),
        "--fallback-latent-steps",
        str(args.fallback_latent_steps),
        "--latent-observation-steps",
        str(args.latent_observation_steps),
        "--latent-repair-strategy",
        args.latent_repair_strategy,
        "--experiment-part",
        args.experiment_part,
        "--max-new-rows",
        str(args.max_new_rows),
        "--output-prefix",
        args.output_prefix,
    ]
    if args.include_baselines:
        command.append("--include-baselines")
    if not args.strict_reuse:
        command.append("--no-strict-reuse")
    if resume_csv:
        command.extend(["--resume-from-partial", str(resume_csv)])
    run_command(command)

    manifest_path = project_path("exports", f"{args.output_prefix}_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    checkpoint = Path(args.result_zip) if args.result_zip else default_result_zip(complete=False)
    zip_results(checkpoint, args.output_prefix)
    print(f"PHASE4_CHECKPOINT_ZIP {checkpoint}", flush=True)
    if manifest.get("complete"):
        result_zip = Path(args.result_zip) if args.result_zip else default_result_zip(complete=True)
        zip_results(result_zip, args.output_prefix)
        print(f"PHASE4_RESULT_ZIP {result_zip}", flush=True)
    summary_path = project_path("reports", f"{args.output_prefix}_summary.md")
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    return 0


def run_plumbing_gates(args: argparse.Namespace) -> int:
    hidden_out = project_path("exports", "tier2_phase4_latent_hidden_smoke.json")
    run_command(
        [
            sys.executable,
            "scripts/latent_hidden_smoke.py",
            "--model",
            args.model,
            "--quantization",
            args.quantization,
            "--latent-steps",
            str(args.latent_steps),
            "--out",
            str(hidden_out),
        ]
        + (["--skip-gpu-guard"] if args.skip_gpu_guard else [])
    )
    hidden = json.loads(hidden_out.read_text(encoding="utf-8"))
    if not (hidden.get("ok") and hidden.get("signal_check", {}).get("ok")):
        raise RuntimeError(f"Hidden-signal smoke failed: {hidden}")

    roundtrip_out = project_path("exports", "tier2_phase4_latent_tool_roundtrip.json")
    effective_steps = args.latent_steps
    try:
        run_tool_roundtrip(args, roundtrip_out, effective_steps)
    except subprocess.CalledProcessError as exc:
        if "out of memory" not in str(exc).lower() or effective_steps == args.fallback_latent_steps:
            raise
        effective_steps = args.fallback_latent_steps
        run_tool_roundtrip(args, roundtrip_out, effective_steps)
    roundtrip = json.loads(roundtrip_out.read_text(encoding="utf-8"))
    if not roundtrip.get("ok"):
        raise RuntimeError(f"Latent tool-roundtrip failed: {roundtrip}")
    return effective_steps


def write_pip_freeze() -> Path:
    path = project_path("exports", "tier2_phase4_pip_freeze.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=False)
    path.write_text(completed.stdout, encoding="utf-8")
    return path


def run_tool_roundtrip(args: argparse.Namespace, out: Path, latent_steps: int) -> None:
    run_command(
        [
            sys.executable,
            "scripts/latent_tool_roundtrip.py",
            "--model",
            args.model,
            "--quantization",
            args.quantization,
            "--latent-steps",
            str(latent_steps),
            "--fallback-latent-steps",
            str(args.fallback_latent_steps),
            "--out",
            str(out),
        ]
    )


def latest_partial_or_metrics(prefix: str) -> Path | None:
    partial = project_path("exports", f"{prefix}_metrics.partial.csv")
    metrics = project_path("exports", f"{prefix}_metrics.csv")
    if partial.exists():
        return partial
    if metrics.exists():
        return metrics
    return None


def restore_checkpoint(source: str) -> Path:
    archive = fetch_resume_zip(source)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            parts = name.parts
            if not parts or ".." in parts:
                raise RuntimeError(f"Unsafe checkpoint member: {info.filename}")
            if parts[0] == "project":
                target = PROJECT_ROOT.joinpath(*parts[1:])
            elif parts[0] == "runtime":
                target = runtime_path().joinpath(*parts[1:])
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return archive


def fetch_resume_zip(source: str) -> Path:
    if source.startswith(("http://", "https://")):
        target = runtime_path("tmp", "tier2_phase4_resume.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source, target)
        return target
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Resume zip not found: {source}")
    return path


def zip_results(path: Path, output_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    for pattern in (
        "exports/tier2_phase4*",
        f"exports/{output_prefix}*",
        "reports/tier2_phase4*",
        f"reports/{output_prefix}*",
        "README.md",
    ):
        candidates.extend(project_path().glob(pattern))
    manifest = project_path("exports", f"{output_prefix}_manifest.json")
    if manifest.exists():
        batch_id = json.loads(manifest.read_text(encoding="utf-8")).get("batch_id")
        if batch_id:
            candidates.extend(runtime_path("runs").glob(f"{batch_id}*"))
    manifest_path = project_path("exports", f"{output_prefix}_zip_manifest.json")
    files = [
        arcname(candidate)
        for candidate in sorted(set(candidates))
        if candidate.exists() and candidate.is_file() and not forbidden_result_file(candidate)
    ]
    for candidate in sorted(set(candidates)):
        if candidate.exists() and candidate.is_dir():
            for file in candidate.rglob("*"):
                if file.is_file() and not forbidden_result_file(file):
                    files.append(arcname(file))
    write_json(
        manifest_path,
        {
            "zip_path": str(path),
            "file_count": len(sorted(set(files))),
            "files": sorted(set(files)),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    candidates.append(manifest_path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for candidate in sorted(set(candidates)):
            if not candidate.exists():
                continue
            if candidate.is_file() and not forbidden_result_file(candidate):
                zf.write(candidate, arcname(candidate))
            elif candidate.is_dir():
                for file in candidate.rglob("*"):
                    if file.is_file() and not forbidden_result_file(file):
                        zf.write(file, arcname(file))


def forbidden_result_file(path: Path) -> bool:
    lowered = str(path).lower()
    return any(part in lowered for part in ("hf-cache", "huggingface", "model.safetensors", ".bin", "hf_token", "secret"))


def arcname(path: Path) -> str:
    try:
        return str(Path("project") / path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path("runtime") / path.relative_to(runtime_path()))


def default_result_zip(*, complete: bool) -> Path:
    name = "tier2_phase4_lite_results.zip" if complete else "tier2_phase4_lite_checkpoint.zip"
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / name
    if Path("/content").exists():
        return Path("/content") / name
    return project_path("dist", name)


def run_command(command: list[str]) -> None:
    print("RUN_CMD", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
