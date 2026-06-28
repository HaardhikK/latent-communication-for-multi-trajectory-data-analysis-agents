from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.agents import AgentSettings, LATENT_REPAIR_STRATEGIES, run_latentmas_agent, run_single_agent, run_textmas  # noqa: E402
from latent_agent.metrics import RunRecord, write_csv, write_json  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402
from latent_agent.tasks import selected_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 A/B/C comparison.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: all toy tasks.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--debug-decode-latent", action="store_true")
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--latent-observation-steps", type=int, default=1)
    parser.add_argument("--latent-repair-strategy", choices=LATENT_REPAIR_STRATEGIES, default="latent")
    parser.add_argument("--max-new-tokens-code", type=int, default=320)
    parser.add_argument("--max-new-tokens-plan", type=int, default=100)
    parser.add_argument("--max-new-tokens-critic", type=int, default=140)
    parser.add_argument("--execution-timeout-seconds", type=int, default=30)
    parser.add_argument("--output-prefix", default="phase2")
    parser.add_argument("--baseline-metrics", default="", help="Optional CSV used for Phase 2.1 diagnosis comparison.")
    return parser.parse_args()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def _median_iqr(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    median = float(statistics.median(values))
    iqr = float(_percentile(values, 0.75) - _percentile(values, 0.25))
    return median, iqr


def summarize_phase2(records: list[RunRecord]) -> str:
    lines = [
        "# Phase 2 Summary",
        "",
        f"- Runs summarized: {len(records)}",
        "",
        "| Mode | Runs | Pass rate | Median wall s | IQR wall s | Median model ms | IQR model ms | Median exec s | Median coord tokens | Median forward passes | Median peak VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_mode: dict[str, list[RunRecord]] = defaultdict(list)
    by_task_mode: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        by_mode[record.mode].append(record)
        by_task_mode[(record.task_id, record.mode)].append(record)

    mode_stats: dict[str, dict[str, float]] = {}
    for mode in sorted(by_mode):
        subset = by_mode[mode]
        wall_med, wall_iqr = _median_iqr([r.wall_latency_s for r in subset])
        model_med, model_iqr = _median_iqr([r.model_latency_ms for r in subset])
        exec_med, _ = _median_iqr([r.code_exec_latency_s for r in subset])
        coord_med, _ = _median_iqr([float(r.coordination_tokens) for r in subset])
        fwd_med, _ = _median_iqr([float(r.forward_passes) for r in subset])
        vram_med, _ = _median_iqr([r.peak_vram_mb for r in subset])
        pass_rate = sum(r.passed for r in subset) / len(subset)
        mode_stats[mode] = {
            "wall": wall_med,
            "model": model_med,
            "coord": coord_med,
            "pass_rate": pass_rate,
            "forward": fwd_med,
            "vram": vram_med,
        }
        lines.append(
            f"| {mode} | {len(subset)} | {pass_rate:.3f} | {wall_med:.3f} | {wall_iqr:.3f} | "
            f"{model_med:.1f} | {model_iqr:.1f} | {exec_med:.3f} | {coord_med:.0f} | {fwd_med:.0f} | {vram_med:.1f} |"
        )

    lines.extend(["", "## Task Breakdown", "", "| Task | Mode | Runs | Pass rate | Median wall s | Median model ms | Median coordination tokens |", "|---|---|---:|---:|---:|---:|---:|"])
    for (task_id, mode), subset in sorted(by_task_mode.items()):
        wall_med, _ = _median_iqr([r.wall_latency_s for r in subset])
        model_med, _ = _median_iqr([r.model_latency_ms for r in subset])
        coord_med, _ = _median_iqr([float(r.coordination_tokens) for r in subset])
        pass_rate = sum(r.passed for r in subset) / len(subset)
        lines.append(f"| {task_id} | {mode} | {len(subset)} | {pass_rate:.3f} | {wall_med:.3f} | {model_med:.1f} | {coord_med:.0f} |")

    lines.extend(["", "## B vs C Deltas", ""])
    b = mode_stats.get("B_textmas")
    c = mode_stats.get("C_latentmas")
    if b and c:
        c_task_rates = {
            task_id: sum(record.passed for record in subset) / len(subset)
            for (task_id, mode), subset in by_task_mode.items()
            if mode == "C_latentmas"
        }
        decoded_token_gate = c["coord"] < b["coord"]
        pass_rate_gate = c["pass_rate"] >= 0.8
        no_zero_task_gate = all(rate > 0.0 for rate in c_task_rates.values())
        latency_separated_gate = all(record.model_latency_ms >= 0.0 and record.code_exec_latency_s >= 0.0 for record in records)
        speed_gate = c["model"] * 2 <= b["model"]
        lines.append(f"- Decoded coordination token delta, C - B: `{c['coord'] - b['coord']:.0f}` median tokens.")
        lines.append(f"- Model-only latency delta, C - B: `{c['model'] - b['model']:.1f}` median ms.")
        lines.append(f"- Wall-clock delta, C - B: `{c['wall'] - b['wall']:.3f}` median seconds.")
        lines.append(f"- Forward-pass delta, C - B: `{c['forward'] - b['forward']:.0f}` median passes.")
        lines.append(f"- Peak VRAM delta, C - B: `{c['vram'] - b['vram']:.1f}` median MB.")
        lines.append(f"- Pass-rate delta, C - B: `{c['pass_rate'] - b['pass_rate']:.3f}`.")
        lines.append(f"- Decoded coordination token gate: `{'PASS' if decoded_token_gate else 'FAIL'}`.")
        lines.append(f"- C overall pass-rate gate >= 0.800: `{'PASS' if pass_rate_gate else 'FAIL'}` (`{c['pass_rate']:.3f}`).")
        zero_tasks = [task_id for task_id, rate in sorted(c_task_rates.items()) if rate == 0.0]
        if zero_tasks:
            lines.append(f"- C no-task-zero gate: `FAIL` (`{', '.join(zero_tasks)}` had 0/5 passes).")
        else:
            lines.append("- C no-task-zero gate: `PASS`.")
        lines.append(f"- Latency separation gate: `{'PASS' if latency_separated_gate else 'FAIL'}`.")
        lines.append(f"- C model-latency at least 2x faster than B gate: `{'PASS' if speed_gate else 'FAIL'}`.")
        if decoded_token_gate and pass_rate_gate and no_zero_task_gate and latency_separated_gate and speed_gate:
            lines.append("- Gate result: **PASS**.")
        else:
            lines.append("- Gate result: **FAIL**.")
    else:
        lines.append("- B/C delta unavailable because both modes were not present.")

    lines.append("")
    return "\n".join(lines)


def load_flat_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not path.exists():
        return records
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                RunRecord(
                    run_id=row["run_id"],
                    task_id=row["task_id"],
                    mode=row["mode"],
                    model_id=row["model_id"],
                    repeat=int(row["repeat"]),
                    passed=_parse_bool(row["passed"]),
                    score=float(row["score"]),
                    message=row["message"],
                    wall_latency_s=float(row["wall_latency_s"]),
                    model_latency_ms=float(row["model_latency_ms"]),
                    code_exec_latency_s=float(row["code_exec_latency_s"]),
                    forward_passes=int(float(row["forward_passes"])),
                    generated_tokens=int(float(row["generated_tokens"])),
                    model_input_tokens=int(float(row["model_input_tokens"])),
                    coordination_tokens=int(float(row["coordination_tokens"])),
                    tool_io_tokens=int(float(row["tool_io_tokens"])),
                    fixed_prompt_tokens=int(float(row["fixed_prompt_tokens"])),
                    coordination_fraction=float(row["coordination_fraction"]),
                    peak_vram_mb=float(row["peak_vram_mb"]),
                    attempts=int(float(row["attempts"])),
                    run_dir=row["run_dir"],
                    latent_steps=int(float(row.get("latent_steps") or 0)),
                    latent_repair_strategy=row.get("latent_repair_strategy", ""),
                )
            )
    return records


def summarize_phase21_diagnosis(records: list[RunRecord], baseline_records: list[RunRecord], repair_strategy: str) -> str:
    lines = [
        "# Phase 2.1 Diagnosis",
        "",
        f"- Current latent repair strategy: `{repair_strategy}`.",
        f"- Current runs summarized: `{len(records)}`.",
    ]
    if baseline_records:
        lines.append(f"- Baseline runs summarized: `{len(baseline_records)}`.")
    else:
        lines.append("- Baseline runs summarized: `0`.")

    lines.extend(
        [
            "",
            "## Clean Summary Comparison",
            "",
            "| Source | Mode | Runs | Passes | Avg attempts | Avg model ms | Avg forward passes | Avg generated tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, subset in (("baseline", baseline_records), ("current", records)):
        for mode in ("A_single", "B_textmas", "C_latentmas"):
            task_records = [r for r in subset if r.mode == mode and r.task_id == "clean_summary"]
            if task_records:
                lines.append(_diagnosis_stats_row(source, mode, task_records))

    lines.extend(
        [
            "",
            "## Mode C Clean-Summary Code Patterns",
            "",
            "| Source | Runs | Attempt 1 contains `pd.to_numeric_dtype` | Repair contains `pd.to_numeric_dtype` | Repair contains `pd.to_numeric(` |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source, subset in (("baseline", baseline_records), ("current", records)):
        pattern = _clean_summary_c_pattern_counts(subset)
        if pattern["runs"]:
            lines.append(
                f"| {source} | {pattern['runs']} | {pattern['attempt1_bad_api']} | "
                f"{pattern['repair_bad_api']} | {pattern['repair_good_api']} |"
            )

    lines.extend(
        [
            "",
            "## Latency Confound Check",
            "",
            "| Source | Mode | Runs | Passes | Avg attempts | Avg model ms | Avg exec s | Avg forward passes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, subset in (("baseline", baseline_records), ("current", records)):
        for mode in ("A_single", "B_textmas", "C_latentmas"):
            mode_records = [r for r in subset if r.mode == mode]
            if mode_records:
                lines.append(_latency_confound_row(source, mode, mode_records))

    current_c = [r for r in records if r.mode == "C_latentmas"]
    current_b = [r for r in records if r.mode == "B_textmas"]
    if current_c and current_b:
        c_attempts = _avg([r.attempts for r in current_c])
        b_attempts = _avg([r.attempts for r in current_b])
        c_forward = _avg([r.forward_passes for r in current_c])
        b_forward = _avg([r.forward_passes for r in current_b])
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- C average attempts: `{c_attempts:.2f}` vs B `{b_attempts:.2f}`.",
                f"- C average forward passes: `{c_forward:.1f}` vs B `{b_forward:.1f}`.",
                "- If C remains faster with similar attempts, the speed win is from less decoded/model work, not from giving up earlier.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _diagnosis_stats_row(source: str, mode: str, records: list[RunRecord]) -> str:
    return (
        f"| {source} | {mode} | {len(records)} | {sum(r.passed for r in records)} | "
        f"{_avg([r.attempts for r in records]):.2f} | {_avg([r.model_latency_ms for r in records]):.1f} | "
        f"{_avg([r.forward_passes for r in records]):.1f} | {_avg([r.generated_tokens for r in records]):.1f} |"
    )


def _latency_confound_row(source: str, mode: str, records: list[RunRecord]) -> str:
    return (
        f"| {source} | {mode} | {len(records)} | {sum(r.passed for r in records)} | "
        f"{_avg([r.attempts for r in records]):.2f} | {_avg([r.model_latency_ms for r in records]):.1f} | "
        f"{_avg([r.code_exec_latency_s for r in records]):.3f} | {_avg([r.forward_passes for r in records]):.1f} |"
    )


def _clean_summary_c_pattern_counts(records: list[RunRecord]) -> dict[str, int]:
    counts = {"runs": 0, "attempt1_bad_api": 0, "repair_bad_api": 0, "repair_good_api": 0}
    for record in records:
        if record.mode != "C_latentmas" or record.task_id != "clean_summary":
            continue
        counts["runs"] += 1
        run_dir = Path(record.run_dir)
        attempt1 = _read_text_if_exists(run_dir / "attempt_1.py")
        repair = _read_text_if_exists(run_dir / "attempt_2.py")
        if "pd.to_numeric_dtype" in attempt1:
            counts["attempt1_bad_api"] += 1
        if "pd.to_numeric_dtype" in repair:
            counts["repair_bad_api"] += 1
        if "pd.to_numeric(" in repair:
            counts["repair_good_api"] += 1
    return counts


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def _avg(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    configure_runtime(create=True)

    task_ids = [part.strip() for part in args.tasks.split(",") if part.strip()] or None
    tasks = selected_tasks(task_ids)
    modes = ["A", "B", "C"] if args.mode == "all" else [args.mode]
    settings = AgentSettings(
        max_new_tokens_code=args.max_new_tokens_code,
        max_new_tokens_plan=args.max_new_tokens_plan,
        max_new_tokens_critic=args.max_new_tokens_critic,
        execution_timeout_seconds=args.execution_timeout_seconds,
        allow_repair=not args.no_repair,
        latent_steps=args.latent_steps,
        latent_observation_steps=args.latent_observation_steps,
        latent_repair_strategy=args.latent_repair_strategy,
    )

    backend = ModelBackend(args.model)
    batch_id = dt.datetime.now(dt.timezone.utc).strftime("phase2_%Y%m%dT%H%M%SZ")
    run_root = runtime_path("runs", batch_id)
    run_root.mkdir(parents=True, exist_ok=True)

    schedule = []
    for repeat in range(1, args.repeat + 1):
        for task in tasks:
            for mode in modes:
                schedule.append((repeat, task, mode))
    random.shuffle(schedule)

    records: list[RunRecord] = []
    for repeat, task, mode in schedule:
        print(f"RUN repeat={repeat} task={task.task_id} mode={mode}", flush=True)
        if mode == "A":
            record = run_single_agent(backend, task, run_root, repeat=repeat, settings=settings)
        elif mode == "B":
            record = run_textmas(backend, task, run_root, repeat=repeat, settings=settings)
        else:
            record = run_latentmas_agent(
                backend,
                task,
                run_root,
                repeat=repeat,
                settings=settings,
                debug_decode_latent=args.debug_decode_latent,
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "task_id": record.task_id,
                    "mode": record.mode,
                    "passed": bool(record.passed),
                    "score": float(record.score),
                    "coordination_tokens": int(record.coordination_tokens),
                    "run_dir": record.run_dir,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    prefix = args.output_prefix
    metrics_name = f"{prefix}_metrics.csv"
    summary_name = f"{prefix}_summary.md"
    manifest_name = f"{prefix}_run_manifest.json"
    diagnosis_name = f"{prefix}_diagnosis.md"

    write_csv(project_path("exports", metrics_name), records)
    write_csv(runtime_path("exports", f"{batch_id}_metrics.csv"), records)
    summary = summarize_phase2(records)
    report_path = project_path("reports", summary_name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")
    baseline_records = load_flat_records(Path(args.baseline_metrics)) if args.baseline_metrics else []
    diagnosis = summarize_phase21_diagnosis(records, baseline_records, args.latent_repair_strategy)
    project_path("reports", diagnosis_name).write_text(diagnosis, encoding="utf-8")
    write_json(
        project_path("exports", manifest_name),
        {
            "batch_id": batch_id,
            "run_root": str(run_root),
            "model": args.model,
            "modes": modes,
            "repeat": args.repeat,
            "latent_steps": args.latent_steps,
            "latent_observation_steps": args.latent_observation_steps,
            "latent_repair_strategy": args.latent_repair_strategy,
            "output_prefix": prefix,
            "baseline_metrics": args.baseline_metrics,
        },
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
