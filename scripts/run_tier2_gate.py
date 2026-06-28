from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_phase3  # noqa: E402
from latent_agent.cloud import (  # noqa: E402
    assert_cloud_dependencies,
    assert_supported_single_gpu,
    force_single_gpu_env,
    gpu_report,
    summarize_gate,
)
from latent_agent.metrics import RunRecord, write_json  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tier 2 8B/4-bit short-horizon gate.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--horizons", default="short")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--base-generation-seed", type=int, default=17)
    parser.add_argument("--schedule-seed", type=int, default=1701)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--fallback-latent-steps", type=int, default=2)
    parser.add_argument("--latent-observation-steps", type=int, default=1)
    parser.add_argument("--latent-repair-strategy", choices=["text_reset"], default="text_reset")
    parser.add_argument("--output-prefix", default="tier2_short_gate")
    parser.add_argument("--strict-cloud-deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-dependency-guard", action="store_true")
    parser.add_argument("--skip-gpu-guard", action="store_true")
    parser.add_argument("--result-zip", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_single_gpu_env()
    runtime = configure_runtime(create=True)

    import torch

    dependency_report = None
    if not args.skip_dependency_guard:
        dependency_report = assert_cloud_dependencies(strict_versions=args.strict_cloud_deps)
    gpu = gpu_report(torch)
    if not args.skip_gpu_guard and args.quantization == "4bit":
        gpu = assert_supported_single_gpu(torch)
    write_json(
        project_path("exports", "tier2_environment.json"),
        {
            "runtime_root": str(runtime),
            "dependency_report": dependency_report,
            "gpu_report": gpu,
        },
    )

    hidden_out = project_path("exports", "tier2_latent_hidden_smoke.json")
    _run(
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
    hidden_result = json.loads(hidden_out.read_text(encoding="utf-8"))
    effective_latent_steps = args.latent_steps
    oom_fallback_used = False

    roundtrip_out = project_path("exports", "tier2_latent_tool_roundtrip.json")
    try:
        _run(
            [
                sys.executable,
                "scripts/latent_tool_roundtrip.py",
                "--model",
                args.model,
                "--quantization",
                args.quantization,
                "--latent-steps",
                str(effective_latent_steps),
                "--fallback-latent-steps",
                str(args.fallback_latent_steps),
                "--out",
                str(roundtrip_out),
            ]
        )
    except subprocess.CalledProcessError as exc:
        if "out of memory" not in str(exc).lower() or effective_latent_steps == args.fallback_latent_steps:
            raise
        effective_latent_steps = args.fallback_latent_steps
        oom_fallback_used = True
        _run(
            [
                sys.executable,
                "scripts/latent_tool_roundtrip.py",
                "--model",
                args.model,
                "--quantization",
                args.quantization,
                "--latent-steps",
                str(effective_latent_steps),
                "--fallback-latent-steps",
                str(args.fallback_latent_steps),
                "--out",
                str(roundtrip_out),
            ]
        )

    partial_csv = project_path("exports", f"{args.output_prefix}_metrics.partial.csv")
    phase3_cmd = [
        sys.executable,
        "scripts/run_phase3.py",
        "--model",
        args.model,
        "--quantization",
        args.quantization,
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
        "--output-prefix",
        args.output_prefix,
    ]
    if partial_csv.exists():
        phase3_cmd.extend(["--resume-from-partial", str(partial_csv)])
    _run(phase3_cmd)

    metrics_path = project_path("exports", f"{args.output_prefix}_metrics.csv")
    records = _load_records(metrics_path)
    gate = summarize_gate(records)
    gate.update(
        {
            "model": args.model,
            "quantization": args.quantization,
            "effective_latent_steps": effective_latent_steps,
            "oom_fallback_used": oom_fallback_used,
            "hidden_smoke": hidden_result,
            "full_sweep_deferred": True,
        }
    )
    write_json(project_path("exports", f"{args.output_prefix}_gate.json"), gate)
    summary = _tier2_summary(gate)
    project_path("reports", f"{args.output_prefix}_summary.md").write_text(summary, encoding="utf-8")
    if not gate["valid_text_baseline"]:
        project_path("reports", "tier2_gate_diagnosis.md").write_text(_tier2_diagnosis(records, gate), encoding="utf-8")

    zip_path = Path(args.result_zip) if args.result_zip else _default_result_zip()
    _zip_results(zip_path, args.output_prefix)
    print(summary)
    print(f"RESULT_ZIP {zip_path}")
    return 0


def _run(command: list[str]) -> None:
    print("RUN_CMD", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_records(path: Path) -> list[RunRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [run_phase3._run_record_from_flat_row(row) for row in csv.DictReader(handle)]


def _tier2_summary(gate: dict[str, Any]) -> str:
    status = "PASSED" if gate["valid_text_baseline"] else "FAILED"
    full_sweep_command = (
        "python scripts/run_phase3.py --model Qwen/Qwen3-8B --quantization 4bit --repeat 5 "
        "--temperature 0.2 --top-p 0.95 --base-generation-seed 17 --schedule-seed 1701 "
        "--latent-repair-strategy text_reset --output-prefix tier2_phase3_full"
    )
    repair_ablation = (
        "Deferred repair ablation after full-sweep approval: rerun Mode C with "
        "`--mode C --latent-repair-strategy latent` and compare against the default "
        "`text_reset` C rows."
    )
    lines = [
        "# Tier 2 Short Gate Summary",
        "",
        f"- Gate status: **{status}**",
        f"- Model: `{gate['model']}`",
        f"- Quantization: `{gate['quantization']}`",
        f"- A pass rate: `{gate['a_pass_rate']:.3f}` over `{gate['a_runs']}` runs",
        f"- B pass rate: `{gate['b_pass_rate']:.3f}` over `{gate['b_runs']}` runs",
        f"- C pass rate: `{gate['c_pass_rate']:.3f}` over `{gate['c_runs']}` runs",
        f"- A-B gap: `{gate['a_b_gap']:.3f}`; tolerance `{gate['tolerance']:.3f}`",
        f"- Effective latent steps: `{gate['effective_latent_steps']}`; OOM fallback used: `{gate['oom_fallback_used']}`",
        "",
        "This is a coarse screen: `15` runs per mode gives pass-rate resolution of about `0.067`. "
        "The `0.15` gate decides whether the full sweep is worth running; it is not final accuracy proof.",
        "",
        "## Family Pass Rates",
        "",
        "| Family | Mode | Runs | Passes | Pass rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in gate["family_rows"]:
        lines.append(f"| {row['family']} | {row['mode']} | {row['runs']} | {row['passes']} | {row['pass_rate']:.3f} |")
    lines.extend(
        [
            "",
            "## Deferred Full Sweep",
            "",
            "Do not run this until the short gate is reviewed and approved.",
            "",
            "```bash",
            full_sweep_command,
            "```",
            "",
            repair_ablation,
            "",
        ]
    )
    return "\n".join(lines)


def _tier2_diagnosis(records: list[RunRecord], gate: dict[str, Any]) -> str:
    lines = [
        "# Tier 2 Gate Diagnosis",
        "",
        "The short-gate text baseline is not valid yet. Stop before the full 135-row sweep.",
        "",
        f"- A pass rate: `{gate['a_pass_rate']:.3f}`",
        f"- B pass rate: `{gate['b_pass_rate']:.3f}`",
        f"- A-B gap: `{gate['a_b_gap']:.3f}`",
        "",
        "## A-pass / B-fail Pairs",
        "",
        "| Task | Repeat | A run | B run | B message | B prompt/code artifacts |",
        "|---|---:|---|---|---|---|",
    ]
    by_key = {(record.task_id, record.repeat, record.mode): record for record in records}
    for record in records:
        if record.mode != "A_single" or not record.passed:
            continue
        b = by_key.get((record.task_id, record.repeat, "B_textmas"))
        if b and not b.passed:
            lines.append(
                f"| {record.task_id} | {record.repeat} | `{record.run_id}` | `{b.run_id}` | "
                f"{_md(b.message)} | `{b.run_dir}` |"
            )
    lines.extend(
        [
            "",
            "Likely interpretations to inspect manually: task-family brittleness, remaining planner/coder decomposition weakness, or strict-scorer mismatch. "
            "Prompt audits are stored in each B/C raw run directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")[:200]


def _default_result_zip() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/tier2_short_gate_results.zip")
    if Path("/content").exists():
        return Path("/content/tier2_short_gate_results.zip")
    return project_path("dist", "tier2_short_gate_results.zip")


def _zip_results(path: Path, output_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = project_path("exports", f"{output_prefix}_run_manifest.json")
    batch_id = None
    if manifest_path.exists():
        batch_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("batch_id")
    candidates: list[Path] = []
    for pattern in (
        "exports/tier2*",
        f"exports/{output_prefix}*",
        "reports/tier2*",
        f"reports/{output_prefix}*",
    ):
        candidates.extend(project_path().glob(pattern))
    if batch_id:
        candidates.extend(runtime_path("runs").glob(f"{batch_id}*"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for candidate in sorted(set(candidates)):
            if not candidate.exists():
                continue
            if candidate.is_file():
                zf.write(candidate, _arcname(candidate))
            else:
                for file in candidate.rglob("*"):
                    if file.is_file():
                        zf.write(file, _arcname(file))


def _arcname(path: Path) -> str:
    try:
        return str(Path("project") / path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path("runtime") / path.relative_to(runtime_path()))


if __name__ == "__main__":
    raise SystemExit(main())
