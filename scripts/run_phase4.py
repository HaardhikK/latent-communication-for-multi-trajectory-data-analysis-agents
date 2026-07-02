from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_phase3  # noqa: E402
from latent_agent.agents import (  # noqa: E402
    _build_record,
    _execute_and_score,
    _repair_context,
    _repair_latent_failure,
)
from latent_agent.code_quality import assess_code_quality  # noqa: E402
from latent_agent.executor import ExecutionResult  # noqa: E402
from latent_agent.horizon_tasks import selected_horizon_tasks  # noqa: E402
from latent_agent.latent_backend import LatentBackend, past_length  # noqa: E402
from latent_agent.metrics import ModelCallRecord, RunRecord, write_csv, write_json  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402
from latent_agent.tasks import ScoreResult, ToyTask  # noqa: E402
from latent_agent.token_split import FIXED_PROMPT, TOOL_IO, PromptPart, TokenLedger, extract_python_code, render_prompt  # noqa: E402


C_VARIANTS = ("C1_current", "C2_dedup", "C3_no_latent")
BASELINE_MODE_NAMES = {"A": "A_single", "B": "B_textmas"}


@dataclass(frozen=True)
class StageAppendSpec:
    call_name: str
    parts: list[PromptPart]
    latent_steps: int
    raw_continuation: bool
    stage_index: int = 0

    @property
    def rendered_text(self) -> str:
        return render_prompt(self.parts)


@dataclass(frozen=True)
class ScheduleItem:
    repeat: int
    task: ToyTask
    mode: str
    c_variant: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4A-lite C-variant forensic pilot.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--variants", default="C1_current,C2_dedup,C3_no_latent")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-modes", default="A,B")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--families", default="orders_kpi,sensor_quality,campaign_roi")
    parser.add_argument("--horizons", default="long")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--base-generation-seed", type=int, default=17)
    parser.add_argument("--schedule-seed", type=int, default=1701)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--latent-observation-steps", type=int, default=1)
    parser.add_argument("--fallback-latent-steps", type=int, default=2)
    parser.add_argument("--latent-repair-strategy", choices=["text_reset", "latent"], default="text_reset")
    parser.add_argument("--max-new-tokens-plan", type=int, default=80)
    parser.add_argument("--max-new-tokens-critic", type=int, default=140)
    parser.add_argument("--execution-timeout-seconds", type=int, default=30)
    parser.add_argument("--output-prefix", default="phase4_lite")
    parser.add_argument("--resume-from-partial", default="")
    parser.add_argument("--max-new-rows", type=int, default=0)
    parser.add_argument("--skip-reference-fit", action="store_true")
    parser.add_argument("--debug-decode-latent", action="store_true")
    return parser.parse_args()


def generation_seed_for_repeat(repeat: int, base_seed: int = 17) -> int:
    return run_phase3.generation_seed_for_repeat(repeat, base_seed)


def parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def validate_variants(variants: list[str]) -> list[str]:
    unknown = [variant for variant in variants if variant not in C_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown C variants: {unknown}; expected {C_VARIANTS}")
    return variants


def make_settings(args: argparse.Namespace, task: ToyTask, budget: dict[str, Any], *, repeat: int):
    return run_phase3.make_settings(
        args,
        task,
        budget,
        repeat=repeat,
        effective_latent_steps=args.latent_steps,
        oom_fallback_used=False,
    )


def build_stage_append_plan(task: ToyTask, variant: str, latent_steps: int) -> list[StageAppendSpec]:
    if variant not in C_VARIANTS:
        raise ValueError(f"Unknown C variant: {variant}")
    specs: list[StageAppendSpec] = []
    if variant in {"C2_dedup", "C3_no_latent"}:
        specs.append(
            StageAppendSpec(
                call_name="latent_context_once",
                parts=[
                    PromptPart(
                        "planner_system",
                        "You are the latent planner in a multi-stage planner -> coder -> critic pipeline. "
                        "Keep the task specification in hidden working memory. Do not emit text.",
                        FIXED_PROMPT,
                    ),
                    PromptPart("task", task.prompt, FIXED_PROMPT),
                ],
                latent_steps=0,
                raw_continuation=False,
            )
        )

    for index, stage in enumerate(task.stage_specs, start=1):
        if variant == "C1_current":
            specs.append(
                StageAppendSpec(
                    call_name=f"latent_planner_stage_{index}",
                    parts=[
                        PromptPart(
                            "planner_system",
                            "You are the latent planner in a multi-stage planner -> coder -> critic pipeline. "
                            "Update hidden working memory with an advisory checklist for only the current stage. "
                            "Do not invent columns, file names, formulas, or output keys; exact requirements come from the decoded task spec. Do not emit text.",
                            FIXED_PROMPT,
                        ),
                        PromptPart("task", task.prompt, FIXED_PROMPT),
                        PromptPart("current_stage", f"Stage {index}/{task.horizon_stages}: {stage}", FIXED_PROMPT),
                    ],
                    latent_steps=latent_steps,
                    raw_continuation=False,
                    stage_index=index,
                )
            )
        else:
            specs.append(
                StageAppendSpec(
                    call_name=f"latent_stage_line_{index}",
                    parts=[PromptPart("current_stage", f"Stage {index}/{task.horizon_stages}: {stage}", FIXED_PROMPT)],
                    latent_steps=0 if variant == "C3_no_latent" else latent_steps,
                    raw_continuation=True,
                    stage_index=index,
                )
            )
    return specs


def run_phase4_latent_variant(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings,
    c_variant: str,
    debug_decode_latent: bool = False,
) -> RunRecord:
    run_dir = run_phase3._prepare_run_dir(run_root, f"C_latentmas_{c_variant}", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results: list[ExecutionResult] = []
    latent = LatentBackend(backend)
    past = None
    stage_append_audit: list[dict[str, Any]] = []

    start = time.perf_counter()
    task.setup(run_dir)

    for spec in build_stage_append_plan(task, c_variant, settings.latent_steps):
        past = _append_latent_with_audit(
            latent,
            backend,
            ledger,
            calls,
            spec,
            past_key_values=past,
            audit=stage_append_audit,
        )

    cache_len_at_decode = past_length(past)
    coder_parts = run_phase3._phase3_coder_parts(task, mode_label=f"phase4 {c_variant} latent planner -> coder -> critic")
    run_phase3._write_prompt_audit(run_dir, "latent_coder_prompt.txt", coder_parts)
    code_text, past = run_phase3._latent_decode(
        latent,
        backend,
        ledger,
        calls,
        "latent_coder_decode",
        coder_parts,
        output_category=TOOL_IO,
        max_new_tokens=settings.max_new_tokens_code,
        past_key_values=past,
        settings=settings,
    )
    code_text = extract_python_code(code_text)
    first_code = code_text
    exec_result, score = _execute_and_score(code_text, task, run_dir, 1, settings.execution_timeout_seconds)
    first_score = score
    exec_results.append(exec_result)
    attempts = 1

    past = run_phase3._latent_append(
        latent,
        backend,
        ledger,
        calls,
        "latent_tool_observation",
        [
            PromptPart(
                "tool_observation_system",
                "Ingest this decoded code execution result into latent working memory for the critic. Do not emit text.",
                FIXED_PROMPT,
            ),
            PromptPart("code_and_execution", _repair_context(code_text, exec_result, score), TOOL_IO),
        ],
        latent_steps=settings.latent_observation_steps,
        past_key_values=past,
    )

    past = run_phase3._latent_append(
        latent,
        backend,
        ledger,
        calls,
        "latent_critic",
        [
            PromptPart(
                "critic_system",
                "You are the latent critic. Update hidden memory with whether all stages and outputs passed. Do not emit text.",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        latent_steps=settings.latent_steps,
        past_key_values=past,
    )

    if settings.allow_repair and not score.passed:
        if settings.latent_repair_strategy == "text_reset":
            repair_code = run_phase3._generate_phase3_text_reset_repair(
                backend,
                ledger,
                calls,
                task,
                code_text,
                exec_result,
                score,
                run_dir,
                settings,
                call_name="latent_text_reset_repair",
            )
        else:
            repair_code, past = _repair_latent_failure(
                settings.latent_repair_strategy,
                latent,
                backend,
                ledger,
                calls,
                task,
                code_text,
                exec_result,
                score,
                past,
                settings,
            )
        repair_code = extract_python_code(repair_code)
        exec_result, score = _execute_and_score(repair_code, task, run_dir, 2, settings.execution_timeout_seconds)
        exec_results.append(exec_result)
        attempts = 2

    if debug_decode_latent:
        debug = latent.decode_from_past(
            "Debug-decode the latent state in one short diagnostic sentence.",
            max_new_tokens=settings.debug_decode_tokens,
            past_key_values=past,
        )
        (run_dir / "latent_debug_decode.txt").write_text(debug.text, encoding="utf-8")
        write_json(
            run_dir / "latent_debug_decode_metrics.json",
            {
                "excluded_from_primary_metrics": True,
                "text_preview": debug.text[:500],
                "metrics": debug.metrics.__dict__,
            },
        )

    record = _build_record(
        run_id=run_dir.name,
        task=task,
        mode="C_latentmas",
        model_id=backend.model_id,
        repeat=repeat,
        score=score,
        wall_latency_s=time.perf_counter() - start,
        exec_results=exec_results,
        calls=calls,
        ledger=ledger,
        attempts=attempts,
        run_dir=run_dir,
        latent_repair_strategy=settings.latent_repair_strategy,
        settings=settings,
    )
    return enrich_forensics(
        record,
        c_variant=c_variant,
        first_code=first_code,
        first_attempt_passed=bool(first_score.passed),
        cache_len_at_decode=cache_len_at_decode,
        stage_append_audit=stage_append_audit,
        anchor_texts=[],
    )


def _append_latent_with_audit(
    latent: LatentBackend,
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    spec: StageAppendSpec,
    *,
    past_key_values,
    audit: list[dict[str, Any]],
):
    cache_len_before = past_length(past_key_values)
    ledger.add_prompt_parts(backend.tokenizer, spec.call_name, spec.parts)
    result = latent.append_latent(
        spec.rendered_text,
        latent_steps=spec.latent_steps,
        past_key_values=past_key_values,
        raw_continuation=spec.raw_continuation,
    )
    calls.append(ModelCallRecord(call_name=spec.call_name, **result.metrics.__dict__))
    audit.append(
        {
            "call_name": spec.call_name,
            "stage_index": spec.stage_index,
            "latent_steps": spec.latent_steps,
            "raw_continuation": spec.raw_continuation,
            "cache_len_before": cache_len_before,
            "cache_len_after": past_length(result.past_key_values),
            "input_tokens": result.metrics.input_tokens,
            "text": spec.rendered_text,
        }
    )
    return result.past_key_values


def enrich_forensics(
    record: RunRecord,
    *,
    c_variant: str = "",
    first_code: str | None = None,
    first_attempt_passed: bool | None = None,
    cache_len_at_decode: int = 0,
    stage_append_audit: list[dict[str, Any]] | None = None,
    anchor_texts: list[str] | None = None,
) -> RunRecord:
    run_dir = Path(record.run_dir)
    if first_code is None:
        first_path = run_dir / "attempt_1.py"
        first_code = first_path.read_text(encoding="utf-8") if first_path.exists() else ""
    if first_attempt_passed is None:
        first_attempt_passed = _first_attempt_passed(run_dir)
    quality = assess_code_quality(first_code)
    stage_append_audit = stage_append_audit or []
    anchor_texts = anchor_texts or []
    stage_path = run_dir / "stage_append_audit.json"
    anchor_path = run_dir / "anchor_texts.json"
    quality_path = run_dir / "first_attempt_code_quality.json"
    write_json(stage_path, stage_append_audit)
    write_json(anchor_path, anchor_texts)
    write_json(quality_path, quality.to_dict())

    record.c_variant = c_variant
    record.first_attempt_passed = bool(first_attempt_passed)
    record.first_attempt_ast_ok = bool(quality.ast_ok)
    record.first_attempt_empty = bool(quality.empty)
    record.first_attempt_repetition_ratio = float(quality.repetition_ratio)
    record.cache_len_at_decode = int(cache_len_at_decode)
    record.stage_append_audit_path = str(stage_path)
    record.anchor_texts_path = str(anchor_path)
    record.details.setdefault("phase4", {})
    record.details["phase4"].update(
        {
            "c_variant": c_variant,
            "first_attempt_code_quality": quality.to_dict(),
            "cache_len_at_decode": int(cache_len_at_decode),
            "stage_append_audit": stage_append_audit,
            "anchor_texts": anchor_texts,
        }
    )
    write_json(run_dir / "run_record.json", record.to_dict())
    return record


def _first_attempt_passed(run_dir: Path) -> bool:
    path = run_dir / "attempt_1_execution.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("score", {}).get("passed", False))


def build_schedule(
    tasks: list[ToyTask],
    variants: list[str],
    repeat: int,
    schedule_seed: int,
    *,
    include_baselines: bool,
    baseline_modes: list[str],
) -> list[ScheduleItem]:
    schedule: list[ScheduleItem] = []
    if include_baselines:
        for rep in range(1, repeat + 1):
            for task in tasks:
                for mode in baseline_modes:
                    schedule.append(ScheduleItem(rep, task, mode, ""))
    for rep in range(1, repeat + 1):
        for task in tasks:
            for variant in variants:
                schedule.append(ScheduleItem(rep, task, "C", variant))
    random.Random(schedule_seed).shuffle(schedule)
    return schedule


def record_key(record: RunRecord) -> tuple[str, str, int, str]:
    return (record.mode, record.task_id, int(record.repeat), record.c_variant or "")


def schedule_key(item: ScheduleItem) -> tuple[str, str, int, str]:
    mode = "C_latentmas" if item.mode == "C" else BASELINE_MODE_NAMES[item.mode]
    return (mode, item.task.task_id, int(item.repeat), item.c_variant or "")


def run_schedule(
    args: argparse.Namespace,
    backend: ModelBackend,
    tasks: list[ToyTask],
    variants: list[str],
    budget_table: dict[str, dict[str, Any]],
    batch_id: str,
    *,
    initial_records: list[RunRecord] | None = None,
) -> list[RunRecord]:
    run_root = runtime_path("runs", batch_id)
    run_root.mkdir(parents=True, exist_ok=True)
    records = list(initial_records or [])
    completed = {record_key(record) for record in records}
    partial_paths = [
        project_path("exports", f"{args.output_prefix}_metrics.partial.csv"),
        runtime_path("exports", f"{batch_id}_metrics.partial.csv"),
    ]
    baseline_modes = [mode for mode in parse_csv_arg(args.baseline_modes) if mode in BASELINE_MODE_NAMES]
    new_rows = 0
    for item in build_schedule(
        tasks,
        variants,
        args.repeat,
        args.schedule_seed,
        include_baselines=args.include_baselines,
        baseline_modes=baseline_modes,
    ):
        key = schedule_key(item)
        if key in completed:
            print(f"SKIP_RESUMED repeat={item.repeat} task={item.task.task_id} mode={key[0]} c_variant={item.c_variant}", flush=True)
            continue
        settings = make_settings(args, item.task, budget_table[item.task.task_id], repeat=item.repeat)
        print(
            f"RUN_PHASE4 repeat={item.repeat} seed={settings.generation_seed} task={item.task.task_id} "
            f"mode={item.mode} c_variant={item.c_variant or '-'}",
            flush=True,
        )
        if item.mode == "C":
            record = run_phase4_latent_variant(
                backend,
                item.task,
                run_root,
                repeat=item.repeat,
                settings=settings,
                c_variant=item.c_variant,
                debug_decode_latent=args.debug_decode_latent,
            )
        else:
            record = run_phase3.run_record(
                backend,
                item.task,
                item.mode,
                run_root,
                item.repeat,
                settings,
                debug_decode_latent=args.debug_decode_latent,
            )
            record = enrich_forensics(record)
        records.append(record)
        completed.add(key)
        new_rows += 1
        for path in partial_paths:
            write_csv(path, records)
        print(
            json.dumps(
                {
                    "task_id": record.task_id,
                    "mode": record.mode,
                    "c_variant": record.c_variant,
                    "repeat": record.repeat,
                    "passed": record.passed,
                    "first_attempt_passed": record.first_attempt_passed,
                    "cache_len_at_decode": record.cache_len_at_decode,
                    "ast_ok": record.first_attempt_ast_ok,
                    "empty": record.first_attempt_empty,
                    "repetition_ratio": round(record.first_attempt_repetition_ratio, 3),
                    "run_dir": record.run_dir,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.max_new_rows > 0 and new_rows >= args.max_new_rows:
            print(f"PHASE4_CHUNK_LIMIT new_rows={new_rows} max_new_rows={args.max_new_rows}", flush=True)
            break
    return records


def load_resume_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Resume CSV does not exist: {path}")
    records_by_key: dict[tuple[str, str, int, str], RunRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_dir = Path(row.get("run_dir", ""))
            record_path = run_dir / "run_record.json"
            record = _run_record_from_json(record_path) if record_path.exists() else run_phase3._run_record_from_flat_row(row)
            records_by_key[record_key(record)] = record
    return list(records_by_key.values())


def _run_record_from_json(path: Path) -> RunRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_calls"] = [ModelCallRecord(**call) for call in data.get("model_calls", [])]
    allowed = {field.name for field in fields(RunRecord)}
    return RunRecord(**{key: value for key, value in data.items() if key in allowed})


def summarize_phase4(records: list[RunRecord]) -> str:
    rows = summary_rows(records)
    lines = [
        "# Phase 4A-Lite Source-Only Forensic Pilot",
        "",
        "This pilot tests whether long-horizon latent collapse is partly caused by duplicated/polluted KV-cache construction.",
        "",
        f"- Rows summarized: `{len(records)}`",
        f"- Pilot case: **{classify_pilot_case(rows)}**",
        "",
        "| Mode | C variant | Task family | Horizon | Runs | Pass rate | First-attempt pass | AST OK | Empty code | Median repetition | Median cache len |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['c_variant'] or '-'} | {row['task_family']} | {row['horizon_level']} | "
            f"{row['runs']} | {row['pass_rate']:.3f} | {row['first_attempt_pass_rate']:.3f} | "
            f"{row['ast_ok_rate']:.3f} | {row['empty_rate']:.3f} | {row['median_repetition_ratio']:.3f} | "
            f"{row['median_cache_len_at_decode']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Case guide: A means C2_dedup clearly beats C1_current; B means C2 and C3 are similar; "
            "C means C2 remains poor but usually emits valid Python; D means outputs are often degenerate.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.mode, record.c_variant, record.task_family, record.horizon_level), []).append(record)
    rows: list[dict[str, Any]] = []
    for (mode, variant, family, horizon), subset in sorted(groups.items()):
        rows.append(
            {
                "mode": mode,
                "c_variant": variant,
                "task_family": family,
                "horizon_level": horizon,
                "runs": len(subset),
                "pass_rate": _rate(record.passed for record in subset),
                "first_attempt_pass_rate": _rate(record.first_attempt_passed for record in subset),
                "ast_ok_rate": _rate(record.first_attempt_ast_ok for record in subset),
                "empty_rate": _rate(record.first_attempt_empty for record in subset),
                "median_repetition_ratio": _median([record.first_attempt_repetition_ratio for record in subset]),
                "median_cache_len_at_decode": _median([float(record.cache_len_at_decode) for record in subset]),
            }
        )
    return rows


def classify_pilot_case(rows: list[dict[str, Any]]) -> str:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["mode"] == "C_latentmas":
            by_variant.setdefault(row["c_variant"], []).append(row)

    def avg(variant: str, field: str) -> float:
        subset = by_variant.get(variant, [])
        return sum(float(row[field]) for row in subset) / len(subset) if subset else 0.0

    c1_pass = avg("C1_current", "pass_rate")
    c2_pass = avg("C2_dedup", "pass_rate")
    c3_pass = avg("C3_no_latent", "pass_rate")
    c2_ast = avg("C2_dedup", "ast_ok_rate")
    c_all_empty = sum(avg(variant, "empty_rate") for variant in C_VARIANTS) / len(C_VARIANTS)
    if c2_pass >= c1_pass + 0.25:
        return "Case A: C2_dedup directionally beats C1_current; duplicate prompt/cache pollution is implicated."
    if abs(c2_pass - c3_pass) <= 0.17:
        return "Case B: C2_dedup and C3_no_latent are similar; latent vectors may be inert or weak."
    if c2_pass < 0.5 and c2_ast >= 0.67:
        return "Case C: C2_dedup remains poor but code is usually valid Python; inspect semantic failures."
    if c_all_empty >= 0.5 or avg("C2_dedup", "ast_ok_rate") < 0.5:
        return "Case D: outputs look degenerate; investigate decoding, quantization, or capacity."
    return "No single case fired cleanly; expand only after inspecting run artifacts."


def _rate(values) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _median(values: list[float]) -> float:
    clean = [float(value) for value in values]
    return float(statistics.median(clean)) if clean else 0.0


def main() -> int:
    args = parse_args()
    configure_runtime(create=True)
    variants = validate_variants(parse_csv_arg(args.variants))
    task_ids = parse_csv_arg(args.tasks) or None
    families = parse_csv_arg(args.families) or None
    horizons = parse_csv_arg(args.horizons) or None
    tasks = selected_horizon_tasks(task_ids=task_ids, families=families, horizons=horizons)

    backend = ModelBackend(args.model, quantization=args.quantization)
    batch_id = dt.datetime.now(dt.timezone.utc).strftime("phase4_%Y%m%dT%H%M%SZ")
    budget_table = run_phase3.build_budget_table(tasks, backend.tokenizer)
    reference_rows = [] if args.skip_reference_fit else run_phase3.validate_reference_scripts(tasks, batch_id)
    if reference_rows:
        print("PHASE4_REFERENCE_FIT_OK", json.dumps(reference_rows, sort_keys=True), flush=True)

    initial_records = load_resume_records(Path(args.resume_from_partial)) if args.resume_from_partial else []
    if initial_records:
        print(f"PHASE4_RESUME_LOADED rows={len(initial_records)} path={args.resume_from_partial}", flush=True)
    records = run_schedule(args, backend, tasks, variants, budget_table, batch_id, initial_records=initial_records)

    prefix = args.output_prefix
    write_csv(project_path("exports", f"{prefix}_metrics.csv"), records)
    write_csv(runtime_path("exports", f"{batch_id}_metrics.csv"), records)
    write_json(project_path("exports", f"{prefix}_budget_fit.json"), budget_table)
    expected_rows = len(tasks) * len(variants) * args.repeat
    if args.include_baselines:
        baseline_modes = [mode for mode in parse_csv_arg(args.baseline_modes) if mode in BASELINE_MODE_NAMES]
        expected_rows += len(tasks) * len(baseline_modes) * args.repeat
    manifest = {
        "batch_id": batch_id,
        "model": args.model,
        "quantization": args.quantization,
        "variants": variants,
        "include_baselines": args.include_baselines,
        "families": [task.task_family for task in tasks],
        "horizons": sorted({task.horizon_level for task in tasks}),
        "repeat": args.repeat,
        "base_generation_seed": args.base_generation_seed,
        "schedule_seed": args.schedule_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "latent_steps": args.latent_steps,
        "latent_repair_strategy": args.latent_repair_strategy,
        "rows": len(records),
        "expected_rows": expected_rows,
        "complete": len(records) >= expected_rows,
        "max_new_rows": args.max_new_rows,
        "reference_fit": reference_rows,
    }
    write_json(project_path("exports", f"{prefix}_manifest.json"), manifest)
    summary = summarize_phase4(records)
    summary_path = project_path("reports", f"{prefix}_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
