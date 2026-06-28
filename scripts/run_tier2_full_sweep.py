from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
import sys
import urllib.request
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
)
from latent_agent.metrics import RunRecord, write_json  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402


HORIZONS = ["short", "medium", "long"]
PRIMARY_PREFIX = "tier2_phase3_full_primary"
ABLATION_PREFIX = "tier2_phase3_c_latent_repair"
COMBINED_PREFIX = "tier2_phase3_full_combined"
SUMMARY_PATH = project_path("reports", "tier2_phase3_full_summary.md")
PLOT_DIR = project_path("reports", "tier2_phase3_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tier 2 8B/4-bit full horizon sweep in resumable chunks.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--base-generation-seed", type=int, default=17)
    parser.add_argument("--schedule-seed", type=int, default=1701)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--fallback-latent-steps", type=int, default=2)
    parser.add_argument("--latent-observation-steps", type=int, default=1)
    parser.add_argument("--max-new-rows", type=int, default=45)
    parser.add_argument("--resume-zip", default="", help="Optional prior checkpoint/result zip path or URL.")
    parser.add_argument("--result-zip", default="", help="Optional final/checkpoint zip path.")
    parser.add_argument("--strict-cloud-deps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-dependency-guard", action="store_true")
    parser.add_argument("--skip-gpu-guard", action="store_true")
    parser.add_argument("--skip-plumbing-gates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_single_gpu_env()
    runtime = configure_runtime(create=True)
    if args.resume_zip:
        restored = restore_checkpoint(args.resume_zip)
        print(f"RESTORED_CHECKPOINT {restored}", flush=True)

    import torch

    dependency_report = None
    if not args.skip_dependency_guard:
        dependency_report = assert_cloud_dependencies(strict_versions=args.strict_cloud_deps)
    gpu = gpu_report(torch)
    if not args.skip_gpu_guard and args.quantization == "4bit":
        gpu = assert_supported_single_gpu(torch)
    write_json(
        project_path("exports", "tier2_phase3_full_environment.json"),
        {
            "runtime_root": str(runtime),
            "dependency_report": dependency_report,
            "gpu_report": gpu,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )

    effective_latent_steps = args.latent_steps
    oom_fallback_used = False
    if not args.skip_plumbing_gates:
        effective_latent_steps, oom_fallback_used = run_plumbing_gates(args)

    max_new_rows = max(0, int(args.max_new_rows))
    rows_used = 0
    primary_records = load_phase3_records(prefix_path(PRIMARY_PREFIX))
    ablation_records = load_phase3_records(prefix_path(ABLATION_PREFIX))

    primary_expected = expected_primary_rows(args.repeat)
    ablation_expected = expected_ablation_rows(args.repeat)
    print(
        json.dumps(
            {
                "stage": "before_chunk",
                "primary_rows": len(primary_records),
                "primary_expected": primary_expected,
                "ablation_rows": len(ablation_records),
                "ablation_expected": ablation_expected,
                "max_new_rows": max_new_rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if len(primary_records) < primary_expected and (max_new_rows == 0 or rows_used < max_new_rows):
        allowance = 0 if max_new_rows == 0 else max_new_rows - rows_used
        before = len(primary_records)
        run_phase3_chunk(
            args,
            prefix=PRIMARY_PREFIX,
            mode="all",
            horizons="short,medium,long",
            latent_repair_strategy="text_reset",
            max_new_rows=allowance,
            effective_latent_steps=effective_latent_steps,
        )
        primary_records = load_phase3_records(prefix_path(PRIMARY_PREFIX))
        rows_used += max(0, len(primary_records) - before)

    if len(primary_records) >= primary_expected and len(ablation_records) < ablation_expected and (max_new_rows == 0 or rows_used < max_new_rows):
        allowance = 0 if max_new_rows == 0 else max_new_rows - rows_used
        before = len(ablation_records)
        run_phase3_chunk(
            args,
            prefix=ABLATION_PREFIX,
            mode="C",
            horizons="medium,long",
            latent_repair_strategy="latent",
            max_new_rows=allowance,
            effective_latent_steps=effective_latent_steps,
        )
        ablation_records = load_phase3_records(prefix_path(ABLATION_PREFIX))
        rows_used += max(0, len(ablation_records) - before)

    combined_rows = write_combined_outputs(primary_records, ablation_records)
    complete = len(primary_records) >= primary_expected and len(ablation_records) >= ablation_expected
    manifest = {
        "model": args.model,
        "quantization": args.quantization,
        "repeat": args.repeat,
        "base_generation_seed": args.base_generation_seed,
        "schedule_seed": args.schedule_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "requested_latent_steps": args.latent_steps,
        "effective_latent_steps": effective_latent_steps,
        "oom_fallback_used": oom_fallback_used,
        "primary_rows": len(primary_records),
        "primary_expected_rows": primary_expected,
        "ablation_rows": len(ablation_records),
        "ablation_expected_rows": ablation_expected,
        "combined_rows": len(combined_rows),
        "combined_expected_rows": primary_expected + ablation_expected,
        "rows_run_this_chunk": rows_used,
        "chunk_complete": rows_used < max_new_rows if max_new_rows else complete,
        "full_complete": complete,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(project_path("exports", "tier2_phase3_full_manifest.json"), manifest)

    checkpoint = Path(args.result_zip) if args.result_zip else default_checkpoint_zip(complete=False)
    zip_results(checkpoint)
    print(f"CHECKPOINT_ZIP {checkpoint}", flush=True)
    if complete:
        result_zip = default_checkpoint_zip(complete=True)
        if args.result_zip and Path(args.result_zip).name != result_zip.name:
            result_zip = Path(args.result_zip)
        zip_results(result_zip)
        print(f"RESULT_ZIP {result_zip}", flush=True)
    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    return 0


def run_plumbing_gates(args: argparse.Namespace) -> tuple[int, bool]:
    hidden_out = project_path("exports", "tier2_phase3_latent_hidden_smoke.json")
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

    effective_latent_steps = args.latent_steps
    oom_fallback_used = False
    roundtrip_out = project_path("exports", "tier2_phase3_latent_tool_roundtrip.json")
    try:
        run_command(
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
        run_command(
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
    roundtrip = json.loads(roundtrip_out.read_text(encoding="utf-8"))
    if not roundtrip.get("ok"):
        raise RuntimeError(f"Latent tool-roundtrip failed: {roundtrip}")
    return effective_latent_steps, oom_fallback_used


def run_phase3_chunk(
    args: argparse.Namespace,
    *,
    prefix: str,
    mode: str,
    horizons: str,
    latent_repair_strategy: str,
    max_new_rows: int,
    effective_latent_steps: int,
) -> None:
    partial = project_path("exports", f"{prefix}_metrics.partial.csv")
    metrics = project_path("exports", f"{prefix}_metrics.csv")
    resume_path = partial if partial.exists() else metrics if metrics.exists() else None
    command = [
        sys.executable,
        "scripts/run_phase3.py",
        "--model",
        args.model,
        "--quantization",
        args.quantization,
        "--mode",
        mode,
        "--horizons",
        horizons,
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
        latent_repair_strategy,
        "--output-prefix",
        prefix,
        "--skip-a-qualification",
        "--skip-smoke-matrix",
        "--max-new-rows",
        str(max_new_rows),
    ]
    if resume_path:
        command.extend(["--resume-from-partial", str(resume_path)])
    run_command(command)


def write_combined_outputs(primary_records: list[RunRecord], ablation_records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in primary_records:
        mode = "C_text_reset" if record.mode == "C_latentmas" else record.mode
        rows.append(_combined_row(record, analysis_mode=mode, experiment_part="primary"))
    for record in ablation_records:
        rows.append(_combined_row(record, analysis_mode="C_pure_latent", experiment_part="repair_ablation"))
    write_combined_csv(project_path("exports", f"{COMBINED_PREFIX}_metrics.csv"), rows)
    write_full_summary(rows)
    return rows


def write_full_summary(rows: list[dict[str, Any]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pass_rows = pass_rate_rows(rows)
    avs_rows = a_vs_multi_rows(pass_rows)
    efficiency_rows = efficiency_gap_rows(rows)
    coord_rows = coordination_fraction_rows(rows)
    vram_rows = peak_vram_rows(rows)
    ablation_rows = repair_ablation_rows(rows)
    plots = write_plots(pass_rows, efficiency_rows, vram_rows, ablation_rows)
    fallback_fired = any(_coerce_bool(row.get("oom_fallback_used", False)) for row in rows)
    effective_steps = sorted({int(float(row.get("effective_latent_steps") or 0)) for row in rows if str(row.get("analysis_mode", "")).startswith("C_")})

    lines = [
        "# Tier 2 Phase 3 Full Horizon Summary",
        "",
        "This is the full 8B free-cloud horizon sweep for a **multi-stage planning-coordination horizon** with **one code execution at the end**. "
        "Medium/long task failures are retained as signal; the task specs are frozen.",
        "",
        f"- Combined rows present: `{len(rows)}` of expected `165`.",
        f"- Primary rows expected: `135`; pure-latent repair ablation rows expected: `30`.",
        f"- C effective latent steps observed: `{effective_steps}`; OOM fallback fired: `{fallback_fired}`.",
        "",
        "## Research Readout",
        "",
        research_readout(pass_rows, efficiency_rows),
        "",
        "## Pass Rate Vs Horizon",
        "",
        "| Mode | Horizon | Runs | Passes | Pass rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in pass_rows:
        lines.append(f"| {row['mode']} | {row['horizon']} | {row['runs']} | {row['passes']} | {row['pass_rate']:.3f} |")

    lines.extend(["", "## A Vs Multi-Agent", "", "| Horizon | Mode | A passes/runs | Mode passes/runs | Signed delta | Relation |", "|---|---|---:|---:|---:|---|"])
    for row in avs_rows:
        lines.append(
            f"| {row['horizon']} | {row['mode']} | {row['a_passes']}/{row['a_runs']} | "
            f"{row['mode_passes']}/{row['mode_runs']} | {row['signed_delta']:+.3f} | {row['relation']} |"
        )

    lines.extend(["", "## B-C Efficiency Gap Vs Horizon", "", "| Horizon | B coord med | C coord med | B-C coord gap | B model ms med | C model ms med | B-C model gap ms | B forward med | C forward med | B-C forward gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in efficiency_rows:
        lines.append(
            f"| {row['horizon']} | {row['b_coord']:.0f} | {row['c_coord']:.0f} | {row['coord_gap']:.0f} | "
            f"{row['b_model_ms']:.1f} | {row['c_model_ms']:.1f} | {row['model_gap_ms']:.1f} | "
            f"{row['b_forward']:.0f} | {row['c_forward']:.0f} | {row['forward_gap']:.0f} |"
        )

    lines.extend(["", "## Coordination Fraction", "", "| Mode | Horizon | Runs | Median coordination fraction | IQR |", "|---|---|---:|---:|---:|"])
    for row in coord_rows:
        lines.append(f"| {row['mode']} | {row['horizon']} | {row['runs']} | {row['median']:.3f} | {row['iqr']:.3f} |")

    lines.extend(["", "## Peak VRAM", "", "| Mode | Horizon | Runs | Median peak VRAM MB | IQR |", "|---|---|---:|---:|---:|"])
    for row in vram_rows:
        lines.append(f"| {row['mode']} | {row['horizon']} | {row['runs']} | {row['median']:.1f} | {row['iqr']:.1f} |")

    long_c_rows = [
        row
        for row in rows
        if row.get("horizon_level") == "long" and str(row.get("analysis_mode", "")).startswith("C_")
    ]
    lines.extend(["", "### Long-Horizon C VRAM Rows", "", "| Mode | Task | Repeat | Passed | Effective latent steps | Peak VRAM MB | Run dir |", "|---|---|---:|---:|---:|---:|---|"])
    for row in sorted(long_c_rows, key=lambda item: (str(item["analysis_mode"]), str(item["task_id"]), int(item["repeat"]))):
        lines.append(
            f"| {row['analysis_mode']} | {row['task_id']} | {row['repeat']} | {row['passed']} | "
            f"{row.get('effective_latent_steps', '')} | {float(row.get('peak_vram_mb') or 0.0):.1f} | `{row.get('run_dir', '')}` |"
        )

    lines.extend(["", "## C Repair Ablation", "", "| Horizon | Metric | C_text_reset | C_pure_latent | Delta pure-latent minus text-reset |", "|---|---|---:|---:|---:|"])
    for row in ablation_rows:
        lines.append(
            f"| {row['horizon']} | {row['metric']} | {row['text_reset']:.3f} | {row['pure_latent']:.3f} | {row['delta']:+.3f} |"
        )

    lines.extend(["", "## Plots", ""])
    for label, path in plots.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def pass_rate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    modes = ["A_single", "B_textmas", "C_text_reset", "C_pure_latent"]
    for mode in modes:
        for horizon in HORIZONS:
            subset = [row for row in rows if row["analysis_mode"] == mode and row["horizon_level"] == horizon]
            if not subset and mode == "C_pure_latent" and horizon == "short":
                continue
            passes = sum(_coerce_bool(row["passed"]) for row in subset)
            out.append({"mode": mode, "horizon": horizon, "runs": len(subset), "passes": passes, "pass_rate": passes / len(subset) if subset else 0.0})
    return out


def a_vs_multi_rows(pass_rows_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["mode"], row["horizon"]): row for row in pass_rows_}
    rows = []
    for horizon in HORIZONS:
        a = lookup.get(("A_single", horizon), {"runs": 0, "passes": 0, "pass_rate": 0.0})
        for mode in ("B_textmas", "C_text_reset", "C_pure_latent"):
            current = lookup.get((mode, horizon))
            if not current or not current["runs"] or not a["runs"]:
                continue
            delta = float(current["pass_rate"]) - float(a["pass_rate"])
            if current["passes"] > a["passes"]:
                relation = "beats"
            elif current["passes"] == a["passes"]:
                relation = "matches"
            else:
                relation = "trails"
            rows.append(
                {
                    "horizon": horizon,
                    "mode": mode,
                    "a_runs": a["runs"],
                    "a_passes": a["passes"],
                    "mode_runs": current["runs"],
                    "mode_passes": current["passes"],
                    "signed_delta": delta,
                    "relation": relation,
                }
            )
    return rows


def efficiency_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for horizon in HORIZONS:
        b = [row for row in rows if row["analysis_mode"] == "B_textmas" and row["horizon_level"] == horizon]
        c = [row for row in rows if row["analysis_mode"] == "C_text_reset" and row["horizon_level"] == horizon]
        b_coord, _ = median_iqr([float(row.get("coordination_tokens") or 0) for row in b])
        c_coord, _ = median_iqr([float(row.get("coordination_tokens") or 0) for row in c])
        b_model, _ = median_iqr([float(row.get("model_latency_ms") or 0) for row in b])
        c_model, _ = median_iqr([float(row.get("model_latency_ms") or 0) for row in c])
        b_forward, _ = median_iqr([float(row.get("forward_passes") or 0) for row in b])
        c_forward, _ = median_iqr([float(row.get("forward_passes") or 0) for row in c])
        out.append(
            {
                "horizon": horizon,
                "b_coord": b_coord,
                "c_coord": c_coord,
                "coord_gap": b_coord - c_coord,
                "b_model_ms": b_model,
                "c_model_ms": c_model,
                "model_gap_ms": b_model - c_model,
                "b_forward": b_forward,
                "c_forward": c_forward,
                "forward_gap": b_forward - c_forward,
            }
        )
    return out


def coordination_fraction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for mode in ["A_single", "B_textmas", "C_text_reset", "C_pure_latent"]:
        for horizon in HORIZONS:
            subset = [row for row in rows if row["analysis_mode"] == mode and row["horizon_level"] == horizon]
            if not subset and mode == "C_pure_latent" and horizon == "short":
                continue
            med, iqr = median_iqr([float(row.get("coordination_fraction") or 0.0) for row in subset])
            out.append({"mode": mode, "horizon": horizon, "runs": len(subset), "median": med, "iqr": iqr})
    return out


def peak_vram_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for mode in ["A_single", "B_textmas", "C_text_reset", "C_pure_latent"]:
        for horizon in HORIZONS:
            subset = [row for row in rows if row["analysis_mode"] == mode and row["horizon_level"] == horizon]
            if not subset and mode == "C_pure_latent" and horizon == "short":
                continue
            med, iqr = median_iqr([float(row.get("peak_vram_mb") or 0.0) for row in subset])
            out.append({"mode": mode, "horizon": horizon, "runs": len(subset), "median": med, "iqr": iqr})
    return out


def repair_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        ("pass_rate", lambda subset: sum(_coerce_bool(row["passed"]) for row in subset) / len(subset) if subset else 0.0),
        ("attempts_median", lambda subset: median_iqr([float(row.get("attempts") or 0) for row in subset])[0]),
        ("model_latency_ms_median", lambda subset: median_iqr([float(row.get("model_latency_ms") or 0) for row in subset])[0]),
        ("generated_tokens_median", lambda subset: median_iqr([float(row.get("generated_tokens") or 0) for row in subset])[0]),
        ("forward_passes_median", lambda subset: median_iqr([float(row.get("forward_passes") or 0) for row in subset])[0]),
    ]
    out = []
    for horizon in ("medium", "long"):
        text_reset = [row for row in rows if row["analysis_mode"] == "C_text_reset" and row["horizon_level"] == horizon]
        pure = [row for row in rows if row["analysis_mode"] == "C_pure_latent" and row["horizon_level"] == horizon]
        for metric, fn in metrics:
            text_value = fn(text_reset)
            pure_value = fn(pure)
            out.append({"horizon": horizon, "metric": metric, "text_reset": text_value, "pure_latent": pure_value, "delta": pure_value - text_value})
    return out


def research_readout(pass_rows_: list[dict[str, Any]], efficiency_rows_: list[dict[str, Any]]) -> str:
    lookup = {(row["mode"], row["horizon"]): row for row in pass_rows_}
    short_a = lookup.get(("A_single", "short"), {"pass_rate": 0.0})
    short_b = lookup.get(("B_textmas", "short"), {"pass_rate": 0.0})
    short_c = lookup.get(("C_text_reset", "short"), {"pass_rate": 0.0})
    long_eff = next((row for row in efficiency_rows_ if row["horizon"] == "long"), efficiency_rows_[-1] if efficiency_rows_ else {})
    relations = a_vs_multi_rows(pass_rows_)
    medium_long = [row for row in relations if row["horizon"] in {"medium", "long"} and row["mode"] in {"B_textmas", "C_text_reset"}]
    relation_text = "; ".join(f"{row['mode']} {row['relation']} A at {row['horizon']} ({row['signed_delta']:+.3f})" for row in medium_long) or "medium/long accuracy is pending"
    return (
        f"The short horizon remains a ceiling check (A={short_a['pass_rate']:.2f}, B={short_b['pass_rate']:.2f}, C={short_c['pass_rate']:.2f}), "
        "so the main accuracy result comes from medium and long horizons. "
        f"On efficiency, the long-horizon text-vs-latent decoded coordination gap is {float(long_eff.get('coord_gap', 0.0)):.0f} tokens, "
        f"with a model-latency gap of {float(long_eff.get('model_gap_ms', 0.0)):.1f} ms and a forward-pass gap of {float(long_eff.get('forward_gap', 0.0)):.0f}. "
        f"Accuracy framing: {relation_text}. This states directly whether decomposition helps or trails the single-agent baseline rather than claiming latent wins from efficiency alone."
    )


def write_plots(
    pass_rows_: list[dict[str, Any]],
    efficiency_rows_: list[dict[str, Any]],
    vram_rows_: list[dict[str, Any]],
    ablation_rows_: list[dict[str, Any]],
) -> dict[str, str]:
    plots = {
        "pass_rate_vs_horizon": PLOT_DIR / "pass_rate_vs_horizon.svg",
        "bc_coordination_gap": PLOT_DIR / "bc_coordination_gap.svg",
        "bc_model_latency_gap": PLOT_DIR / "bc_model_latency_gap.svg",
        "bc_forward_gap": PLOT_DIR / "bc_forward_gap.svg",
        "peak_vram_vs_horizon": PLOT_DIR / "peak_vram_vs_horizon.svg",
        "repair_ablation": PLOT_DIR / "repair_ablation.svg",
    }
    pass_lookup = {(row["mode"], row["horizon"]): row["pass_rate"] for row in pass_rows_}
    write_svg_line(
        plots["pass_rate_vs_horizon"],
        {mode: {h: pass_lookup.get((mode, h)) for h in HORIZONS} for mode in ["A_single", "B_textmas", "C_text_reset", "C_pure_latent"]},
        "Pass Rate Vs Horizon",
        "Pass rate",
    )
    eff_lookup = {row["horizon"]: row for row in efficiency_rows_}
    write_svg_line(plots["bc_coordination_gap"], {"B-C coord gap": {h: eff_lookup.get(h, {}).get("coord_gap") for h in HORIZONS}}, "B-C Decoded Coordination Gap", "Tokens")
    write_svg_line(plots["bc_model_latency_gap"], {"B-C model gap": {h: eff_lookup.get(h, {}).get("model_gap_ms") for h in HORIZONS}}, "B-C Model-Only Latency Gap", "ms")
    write_svg_line(plots["bc_forward_gap"], {"B-C forward gap": {h: eff_lookup.get(h, {}).get("forward_gap") for h in HORIZONS}}, "B-C Forward-Pass Gap", "passes")
    vram_lookup = {(row["mode"], row["horizon"]): row["median"] for row in vram_rows_}
    write_svg_line(
        plots["peak_vram_vs_horizon"],
        {mode: {h: vram_lookup.get((mode, h)) for h in HORIZONS} for mode in ["A_single", "B_textmas", "C_text_reset", "C_pure_latent"]},
        "Peak VRAM Vs Horizon",
        "MB",
    )
    ablation_lookup = {(row["metric"], row["horizon"]): row for row in ablation_rows_}
    write_svg_line(
        plots["repair_ablation"],
        {
            "C_text_reset pass": {h: ablation_lookup.get(("pass_rate", h), {}).get("text_reset") for h in HORIZONS},
            "C_pure_latent pass": {h: ablation_lookup.get(("pass_rate", h), {}).get("pure_latent") for h in HORIZONS},
        },
        "C Repair Ablation Pass Rate",
        "Pass rate",
    )
    return {key: str(path) for key, path in plots.items()}


def write_svg_line(path: Path, series: dict[str, dict[str, float | None]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 420
    ml, mr, mt, mb = 72, 34, 45, 65
    values = [float(v) for points in series.values() for v in points.values() if v is not None and not math.isnan(float(v))]
    y_min = min(0.0, min(values) if values else 0.0)
    y_max = max(values) if values else 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

    def x_at(index: int) -> float:
        return ml + index * (width - ml - mr) / max(1, len(HORIZONS) - 1)

    def y_at(value: float) -> float:
        return mt + (y_max - value) * (height - mt - mb) / (y_max - y_min)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='24' text-anchor='middle' font-family='Arial' font-size='18' font-weight='700'>{xml(title)}</text>",
        f"<text x='18' y='{height/2}' transform='rotate(-90 18 {height/2})' text-anchor='middle' font-family='Arial' font-size='12'>{xml(ylabel)}</text>",
        f"<line x1='{ml}' y1='{height-mb}' x2='{width-mr}' y2='{height-mb}' stroke='#111827'/>",
        f"<line x1='{ml}' y1='{mt}' x2='{ml}' y2='{height-mb}' stroke='#111827'/>",
    ]
    for index, horizon in enumerate(HORIZONS):
        parts.append(f"<text x='{x_at(index)}' y='{height-mb+24}' text-anchor='middle' font-family='Arial' font-size='12'>{horizon}</text>")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_min + (y_max - y_min) * frac
        y = y_at(value)
        parts.append(f"<line x1='{ml}' y1='{y}' x2='{width-mr}' y2='{y}' stroke='#e5e7eb'/>")
        parts.append(f"<text x='{ml-8}' y='{y+4}' text-anchor='end' font-family='Arial' font-size='11'>{value:.2f}</text>")
    for sidx, (name, points) in enumerate(series.items()):
        color = colors[sidx % len(colors)]
        ordered = [(idx, points.get(horizon)) for idx, horizon in enumerate(HORIZONS)]
        poly: list[str] = []
        for idx, raw in ordered:
            if raw is None:
                if len(poly) > 1:
                    parts.append(f"<polyline points='{' '.join(poly)}' fill='none' stroke='{color}' stroke-width='3'/>")
                poly = []
                continue
            value = float(raw)
            coord = f"{x_at(idx)},{y_at(value)}"
            poly.append(coord)
            parts.append(f"<circle cx='{x_at(idx)}' cy='{y_at(value)}' r='4' fill='{color}'/>")
        if len(poly) > 1:
            parts.append(f"<polyline points='{' '.join(poly)}' fill='none' stroke='{color}' stroke-width='3'/>")
        legend_y = mt + 20 + sidx * 22
        parts.append(f"<rect x='{width-245}' y='{legend_y-10}' width='12' height='12' fill='{color}'/>")
        parts.append(f"<text x='{width-227}' y='{legend_y}' font-family='Arial' font-size='12'>{xml(name)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def median_iqr(values: list[float]) -> tuple[float, float]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0, 0.0
    if len(clean) == 1:
        return clean[0], 0.0
    return percentile(clean, 0.50), percentile(clean, 0.75) - percentile(clean, 0.25)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    return values[lower] * (upper - pos) + values[upper] * (pos - lower)


def _combined_row(record: RunRecord, *, analysis_mode: str, experiment_part: str) -> dict[str, Any]:
    row = record.to_flat_dict()
    row["analysis_mode"] = analysis_mode
    row["experiment_part"] = experiment_part
    row["repair_strategy_scope"] = "pure_latent" if analysis_mode == "C_pure_latent" else row.get("latent_repair_strategy", "")
    return row


def write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_phase3_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        partial = path.with_name(path.name.replace("_metrics.csv", "_metrics.partial.csv"))
        if partial.exists():
            path = partial
        else:
            return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [run_phase3._run_record_from_flat_row(row) for row in rows]


def prefix_path(prefix: str) -> Path:
    metrics = project_path("exports", f"{prefix}_metrics.csv")
    partial = project_path("exports", f"{prefix}_metrics.partial.csv")
    return partial if partial.exists() else metrics


def expected_primary_rows(repeat: int) -> int:
    return 3 * 3 * 3 * int(repeat)


def expected_ablation_rows(repeat: int) -> int:
    return 1 * 3 * 2 * int(repeat)


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
        target = runtime_path("tmp", "tier2_phase3_resume.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source, target)
        return target
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Resume zip not found: {source}")
    return path


def zip_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    patterns = [
        "exports/tier2_phase3*",
        "reports/tier2_phase3*",
        "README.md",
    ]
    for pattern in patterns:
        candidates.extend(project_path().glob(pattern))
    for manifest_name in (f"{PRIMARY_PREFIX}_run_manifest.json", f"{ABLATION_PREFIX}_run_manifest.json"):
        manifest = project_path("exports", manifest_name)
        if manifest.exists():
            batch_id = json.loads(manifest.read_text(encoding="utf-8")).get("batch_id")
            if batch_id:
                candidates.extend(runtime_path("runs").glob(f"{batch_id}*"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for candidate in sorted(set(candidates)):
            if not candidate.exists():
                continue
            if candidate.is_file():
                zf.write(candidate, arcname(candidate))
            else:
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


def default_checkpoint_zip(*, complete: bool) -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/tier2_phase3_full_results.zip" if complete else "/kaggle/working/tier2_phase3_full_checkpoint.zip")
    if Path("/content").exists():
        return Path("/content/tier2_phase3_full_results.zip" if complete else "/content/tier2_phase3_full_checkpoint.zip")
    return project_path("dist", "tier2_phase3_full_results.zip" if complete else "tier2_phase3_full_checkpoint.zip")


def run_command(command: list[str]) -> None:
    print("RUN_CMD", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def xml(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
