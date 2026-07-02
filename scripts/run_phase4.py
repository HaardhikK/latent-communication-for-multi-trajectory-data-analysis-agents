from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
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
from latent_agent.token_split import COORDINATION, FIXED_PROMPT, TOOL_IO, PromptPart, TokenLedger, extract_python_code, render_prompt  # noqa: E402


C_VARIANTS = ("C1_phase3_exact", "C1_current", "C2_dedup", "C3_no_latent", "C5_anchor")
BASELINE_MODE_NAMES = {"A": "A_single", "B": "B_textmas"}
DEFAULT_EXPERIMENT_PART = "session2"
PRIOR_C1_LONG_PASSES = 3
PRIOR_C1_LONG_RUNS = 15
CURRENT_EXPERIMENT_PART = DEFAULT_EXPERIMENT_PART


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
    experiment_part: str = DEFAULT_EXPERIMENT_PART


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4A-lite C-variant forensic pilot.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--variants", default="C1_phase3_exact,C2_dedup,C3_no_latent,C5_anchor")
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
    parser.add_argument("--experiment-part", default=DEFAULT_EXPERIMENT_PART)
    parser.add_argument("--resume-from-partial", default="")
    parser.add_argument("--strict-reuse", action=argparse.BooleanOptionalAction, default=True)
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
    if variant == "C1_phase3_exact":
        return []
    specs: list[StageAppendSpec] = []
    if variant in {"C2_dedup", "C3_no_latent", "C5_anchor"}:
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


def run_phase4_phase3_exact(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings,
    debug_decode_latent: bool = False,
) -> RunRecord:
    diagnostics = run_phase3.Phase3LatentDiagnostics()
    record = run_phase3.run_phase3_latentmas(
        backend,
        task,
        run_root,
        repeat=repeat,
        settings=settings,
        debug_decode_latent=debug_decode_latent,
        diagnostics=diagnostics,
    )
    return enrich_forensics(
        record,
        c_variant="C1_phase3_exact",
        first_code=diagnostics.first_code,
        first_attempt_passed=diagnostics.first_attempt_passed,
        cache_len_at_decode=diagnostics.cache_len_at_decode,
        stage_append_audit=diagnostics.latent_append_audit,
        anchor_texts=[],
    )


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
    if c_variant == "C1_phase3_exact":
        return run_phase4_phase3_exact(
            backend,
            task,
            run_root,
            repeat=repeat,
            settings=settings,
            debug_decode_latent=debug_decode_latent,
        )

    run_dir = run_phase3._prepare_run_dir(run_root, f"C_latentmas_{c_variant}", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results: list[ExecutionResult] = []
    latent = LatentBackend(backend)
    past = None
    stage_append_audit: list[dict[str, Any]] = []
    anchor_texts: list[str] = []

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
        if c_variant == "C5_anchor" and spec.stage_index:
            anchor_text, past = _decode_and_append_anchor(
                latent,
                backend,
                ledger,
                calls,
                past,
                stage_index=spec.stage_index,
                task=task,
                audit=stage_append_audit,
            )
            anchor_texts.append(anchor_text)

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
        anchor_texts=anchor_texts,
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


def _decode_and_append_anchor(
    latent: LatentBackend,
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    past_key_values,
    *,
    stage_index: int,
    task: ToyTask,
    audit: list[dict[str, Any]],
) -> tuple[str, object]:
    prompt_parts = [
        PromptPart(
            "anchor_system",
            "Decode one terse stage anchor for the coder. Use one line. No code, imports, assignments, dataframe snippets, or invented schema.",
            FIXED_PROMPT,
        ),
        PromptPart("current_stage", f"Stage {stage_index}/{task.horizon_stages}: {task.stage_specs[stage_index - 1]}", FIXED_PROMPT),
    ]
    call_name = f"anchor_decode_stage_{stage_index}"
    ledger.add_prompt_parts(backend.tokenizer, call_name, prompt_parts)
    result = latent.decode_from_past(
        render_prompt(prompt_parts),
        max_new_tokens=24,
        past_key_values=past_key_values,
        temperature=0.0,
        top_p=1.0,
        generation_seed=None,
    )
    anchor = _sanitize_anchor(result.text)
    ledger.add_generated(backend.tokenizer, call_name, anchor, COORDINATION)
    calls.append(ModelCallRecord(call_name=call_name, **result.metrics.__dict__))

    cache_len_before = past_length(result.past_key_values)
    append = latent.append_latent(
        anchor,
        latent_steps=0,
        past_key_values=result.past_key_values,
        raw_continuation=True,
    )
    calls.append(ModelCallRecord(call_name=f"anchor_append_stage_{stage_index}", **append.metrics.__dict__))
    audit.append(
        {
            "call_name": f"anchor_append_stage_{stage_index}",
            "stage_index": stage_index,
            "latent_steps": 0,
            "raw_continuation": True,
            "cache_len_before": cache_len_before,
            "cache_len_after": past_length(append.past_key_values),
            "input_tokens": append.metrics.input_tokens,
            "text": anchor,
            "anchor_decode_raw": result.text,
        }
    )
    return anchor, append.past_key_values


def _sanitize_anchor(text: str) -> str:
    cleaned = run_phase3.sanitize_planner_report(text)
    line = " ".join(cleaned.splitlines()).strip()
    line = re.sub(r"\s+", " ", line)
    return line[:240] or "Follow the authoritative task specification for this stage."


def source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def generation_path_hash() -> str:
    digest = hashlib.sha256()
    for path in [
        PROJECT_ROOT / "scripts" / "run_phase3.py",
        PROJECT_ROOT / "scripts" / "run_phase4.py",
        PROJECT_ROOT / "src" / "latent_agent" / "agents.py",
        PROJECT_ROOT / "src" / "latent_agent" / "latent_backend.py",
        PROJECT_ROOT / "src" / "latent_agent" / "models.py",
        PROJECT_ROOT / "src" / "latent_agent" / "token_split.py",
    ]:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def analyze_failure(run_dir: Path, record: RunRecord, *, quality_empty: bool, quality_ast_ok: bool) -> dict[str, Any]:
    attempt1 = _load_attempt_execution(run_dir, 1)
    attempt2 = _load_attempt_execution(run_dir, 2)
    exception1 = _exception_signature(attempt1)
    exception2 = _exception_signature(attempt2)
    code1 = _read_text(run_dir / "attempt_1.py")
    code2 = _read_text(run_dir / "attempt_2.py")
    similarity = difflib.SequenceMatcher(None, code1, code2).ratio() if code1 and code2 else 0.0

    if record.passed:
        failure_type = "passed"
    elif quality_empty:
        failure_type = "empty_or_degenerate_code"
    elif not quality_ast_ok:
        failure_type = "invalid_python"
    elif _attempt_runtime_failed(attempt1):
        failure_type = "runtime_bug"
    elif attempt1 and not (attempt1.get("score", {}) or {}).get("passed", False):
        failure_type = "semantic_scorer_slip"
    else:
        failure_type = "other"

    return {
        "failure_type": failure_type,
        "repair_similarity": similarity,
        "repeated_exception": bool(exception1 and exception1 == exception2),
        "attempt_1_exception": exception1,
        "attempt_2_exception": exception2,
        "attempt_1_returncode": (attempt1.get("execution", {}) or {}).get("returncode") if attempt1 else None,
        "attempt_2_returncode": (attempt2.get("execution", {}) or {}).get("returncode") if attempt2 else None,
    }


def _load_attempt_execution(run_dir: Path, attempt: int) -> dict[str, Any]:
    path = run_dir / f"attempt_{attempt}_execution.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _attempt_runtime_failed(attempt: dict[str, Any]) -> bool:
    execution = attempt.get("execution", {}) if attempt else {}
    return bool(execution.get("timed_out") or int(execution.get("returncode") or 0) != 0 or execution.get("stderr"))


def _exception_signature(attempt: dict[str, Any]) -> str:
    stderr = ((attempt.get("execution", {}) if attempt else {}).get("stderr") or "").strip()
    if not stderr:
        return ""
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:300]
    return ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


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
    record.experiment_part = getattr(record, "experiment_part", "") or CURRENT_EXPERIMENT_PART
    record.source_commit = getattr(record, "source_commit", "") or source_commit()
    record.script_hash = getattr(record, "script_hash", "") or generation_path_hash()
    failure = analyze_failure(run_dir, record, quality_empty=quality.empty, quality_ast_ok=quality.ast_ok)
    record.failure_type = failure["failure_type"]
    record.repair_similarity = float(failure["repair_similarity"])
    record.repeated_exception = bool(failure["repeated_exception"])
    record.details.setdefault("phase4", {})
    record.details["phase4"].update(
        {
            "c_variant": c_variant,
            "experiment_part": record.experiment_part,
            "source_commit": record.source_commit,
            "script_hash": record.script_hash,
            "first_attempt_code_quality": quality.to_dict(),
            "cache_len_at_decode": int(cache_len_at_decode),
            "stage_append_audit": stage_append_audit,
            "anchor_texts": anchor_texts,
            "failure_analysis": failure,
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
    experiment_part: str = DEFAULT_EXPERIMENT_PART,
) -> list[ScheduleItem]:
    baseline_schedule: list[ScheduleItem] = []
    c_schedule: list[ScheduleItem] = []
    if include_baselines:
        for rep in range(1, repeat + 1):
            for task in tasks:
                for mode in baseline_modes:
                    baseline_schedule.append(ScheduleItem(rep, task, mode, "", experiment_part))
    for rep in range(1, repeat + 1):
        for task in tasks:
            for variant in variants:
                c_schedule.append(ScheduleItem(rep, task, "C", variant, experiment_part))
    rng = random.Random(schedule_seed)
    rng.shuffle(baseline_schedule)
    rng.shuffle(c_schedule)
    return baseline_schedule + c_schedule


def record_key(record: RunRecord) -> tuple[str, str, int, str, str]:
    return (record.mode, record.task_id, int(record.repeat), record.c_variant or "", record.experiment_part or "")


def schedule_key(item: ScheduleItem) -> tuple[str, str, int, str, str]:
    mode = "C_latentmas" if item.mode == "C" else BASELINE_MODE_NAMES[item.mode]
    return (mode, item.task.task_id, int(item.repeat), item.c_variant or "", item.experiment_part or DEFAULT_EXPERIMENT_PART)


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
        experiment_part=args.experiment_part,
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
                    "experiment_part": record.experiment_part,
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


def filter_resume_records(
    records: list[RunRecord],
    *,
    tasks: list[ToyTask],
    variants: list[str],
    args: argparse.Namespace,
) -> list[RunRecord]:
    if not args.strict_reuse:
        return records
    baseline_modes = [mode for mode in parse_csv_arg(args.baseline_modes) if mode in BASELINE_MODE_NAMES]
    allowed = {
        schedule_key(item)
        for item in build_schedule(
            tasks,
            variants,
            args.repeat,
            args.schedule_seed,
            include_baselines=args.include_baselines,
            baseline_modes=baseline_modes,
            experiment_part=args.experiment_part,
        )
    }
    kept = [record for record in records if record_key(record) in allowed]
    dropped = len(records) - len(kept)
    if dropped:
        print(f"PHASE4_RESUME_FILTER dropped={dropped} kept={len(kept)} strict_reuse=True", flush=True)
    return kept


def _run_record_from_json(path: Path) -> RunRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_calls"] = [ModelCallRecord(**call) for call in data.get("model_calls", [])]
    allowed = {field.name for field in fields(RunRecord)}
    return RunRecord(**{key: value for key, value in data.items() if key in allowed})


def summarize_phase4(records: list[RunRecord]) -> str:
    rows = summary_rows(records)
    by_variant = _aggregate_by_variant(records)
    decisions = _decision_lines(by_variant)
    failure_rows = _failure_rows(records)
    lines = [
        "# Phase 4A Session 2 Reproduction And Attribution",
        "",
        "This run treats final pass and first-attempt pass as co-primary metrics for long-horizon reproduction and attribution.",
        "",
        f"- Rows summarized: `{len(records)}`",
        f"- Source commit: `{source_commit()}`",
        f"- Generation-path hash: `{generation_path_hash()[:16]}`",
        "",
        "## Decisions",
        "",
        *[f"- {line}" for line in decisions],
        "",
        "## By Family",
        "",
        "| Mode | C variant | Task family | Horizon | Runs | Final pass | First-attempt pass | AST OK | Empty code | Median repetition | Median cache len |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['c_variant'] or '-'} | {row['task_family']} | {row['horizon_level']} | "
            f"{row['runs']} | {row['pass_rate']:.3f} | {row['first_attempt_pass_rate']:.3f} | "
            f"{row['ast_ok_rate']:.3f} | {row['empty_rate']:.3f} | {row['median_repetition_ratio']:.3f} | "
            f"{row['median_cache_len_at_decode']:.0f} |"
        )
    lines.extend(["", "## By Variant", ""])
    lines.append("| Mode | C variant | Runs | Final pass | Final Wilson CI | First-attempt pass | First Wilson CI | Median cache len |")
    lines.append("|---|---|---:|---:|---|---:|---|---:|")
    for key, row in sorted(by_variant.items()):
        lines.append(
            f"| {row['mode']} | {row['c_variant'] or '-'} | {row['runs']} | {row['pass_rate']:.3f} | "
            f"[{row['pass_ci'][0]:.3f}, {row['pass_ci'][1]:.3f}] | {row['first_attempt_pass_rate']:.3f} | "
            f"[{row['first_ci'][0]:.3f}, {row['first_ci'][1]:.3f}] | {row['median_cache_len_at_decode']:.0f} |"
        )
    lines.extend(["", "## Failure Classes", ""])
    lines.append("| Mode | C variant | Failure type | Rows |")
    lines.append("|---|---|---|---:|")
    for row in failure_rows:
        lines.append(f"| {row['mode']} | {row['c_variant'] or '-'} | {row['failure_type']} | {row['rows']} |")
    lines.extend(["", ""])
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
                "pass_count": sum(record.passed for record in subset),
                "pass_rate": _rate(record.passed for record in subset),
                "first_attempt_pass_count": sum(record.first_attempt_passed for record in subset),
                "first_attempt_pass_rate": _rate(record.first_attempt_passed for record in subset),
                "ast_ok_rate": _rate(record.first_attempt_ast_ok for record in subset),
                "empty_rate": _rate(record.first_attempt_empty for record in subset),
                "median_repetition_ratio": _median([record.first_attempt_repetition_ratio for record in subset]),
                "median_cache_len_at_decode": _median([float(record.cache_len_at_decode) for record in subset]),
            }
        )
    return rows


def _aggregate_by_variant(records: list[RunRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.mode, record.c_variant), []).append(record)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for key, subset in groups.items():
        pass_count = sum(record.passed for record in subset)
        first_count = sum(record.first_attempt_passed for record in subset)
        rows[key] = {
            "mode": key[0],
            "c_variant": key[1],
            "runs": len(subset),
            "pass_count": pass_count,
            "pass_rate": pass_count / len(subset),
            "pass_ci": wilson_ci(pass_count, len(subset)),
            "first_attempt_pass_count": first_count,
            "first_attempt_pass_rate": first_count / len(subset),
            "first_ci": wilson_ci(first_count, len(subset)),
            "median_cache_len_at_decode": _median([float(record.cache_len_at_decode) for record in subset]),
        }
    return rows


def _decision_lines(by_variant: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    a = by_variant.get(("A_single", ""))
    b = by_variant.get(("B_textmas", ""))
    if a:
        lines.append(f"Gate 0 A_long: {a['pass_count']}/{a['runs']} final pass; flag threshold is <0.87.")
    if b:
        lines.append(f"Gate 0 B_long: {b['pass_count']}/{b['runs']} final pass; expected acceptable band is [0.55, 0.95].")
    c1 = by_variant.get(("C_latentmas", "C1_phase3_exact"))
    c2 = by_variant.get(("C_latentmas", "C2_dedup"))
    c3 = by_variant.get(("C_latentmas", "C3_no_latent"))
    c5 = by_variant.get(("C_latentmas", "C5_anchor"))
    if c1:
        for field, label in (("pass_rate", "final"), ("first_attempt_pass_rate", "first-attempt")):
            rate = c1[field]
            if rate <= 0.35:
                decision = "collapse reproduces"
            elif rate >= 0.50:
                decision = "prior collapse looks fragile/noisy"
            else:
                decision = "inconclusive; add C1 repeats 6-8 before branching"
            if label == "final":
                fisher = fisher_exact_two_sided(int(c1["pass_count"]), int(c1["runs"]), PRIOR_C1_LONG_PASSES, PRIOR_C1_LONG_RUNS)
                lines.append(f"C1_phase3_exact {label}: {rate:.3f}; {decision}; Fisher vs prior 3/15 p={fisher:.3f}.")
            else:
                lines.append(f"C1_phase3_exact {label}: {rate:.3f}; {decision}; no prior first-attempt baseline for Fisher test.")
    if c1 and c2:
        lines.append(_pair_delta_line("Cache pollution", "C2-C1", c2, c1, "pass"))
        lines.append(_pair_delta_line("Cache pollution", "C2-C1", c2, c1, "first"))
    if c2 and c3:
        delta = c2["pass_rate"] - c3["pass_rate"]
        p_value = _fisher_between(c2, c3, "pass")
        if abs(delta) <= 0.13:
            lines.append(f"Latent-step readout C2-C3 final={delta:.3f}; Fisher p={p_value:.3f}: no detectable latent-step contribution at this sample size.")
        else:
            lines.append(f"Latent-step readout C2-C3 final={delta:.3f}; Fisher p={p_value:.3f}: latent steps differ directionally in this run.")
        lines.append(_pair_delta_line("Latent-step readout", "C2-C3", c2, c3, "first"))
    if c5 and c2:
        lines.append(_pair_delta_line("Anchor effect primary", "C5-C2", c5, c2, "pass"))
        lines.append(_pair_delta_line("Anchor effect primary", "C5-C2", c5, c2, "first"))
    if c5 and c3:
        lines.append(_pair_delta_line("Anchor secondary", "C5-C3", c5, c3, "pass") + "; interpret only if C2 and C3 are similar.")
    if b:
        for label, row in (
            ("C1-vs-B", c1),
            ("C2-vs-B", c2),
            ("C3-vs-B", c3),
            ("C5-vs-B", c5),
        ):
            if row:
                lines.append(_pair_delta_line("Text-baseline comparison", label, row, b, "pass"))
    return lines or ["No decision rows available yet."]


def _pair_delta_line(prefix: str, label: str, left: dict[str, Any], right: dict[str, Any], metric: str) -> str:
    if metric == "pass":
        left_rate = float(left["pass_rate"])
        right_rate = float(right["pass_rate"])
        metric_label = "final"
    elif metric == "first":
        left_rate = float(left["first_attempt_pass_rate"])
        right_rate = float(right["first_attempt_pass_rate"])
        metric_label = "first-attempt"
    else:
        raise ValueError(f"unknown metric: {metric}")
    return f"{prefix} delta {label} {metric_label}={left_rate - right_rate:.3f}; Fisher p={_fisher_between(left, right, metric):.3f}"


def _fisher_between(left: dict[str, Any], right: dict[str, Any], metric: str) -> float:
    if metric == "pass":
        left_count = int(left["pass_count"])
        right_count = int(right["pass_count"])
    elif metric == "first":
        left_count = int(left["first_attempt_pass_count"])
        right_count = int(right["first_attempt_pass_count"])
    else:
        raise ValueError(f"unknown metric: {metric}")
    return fisher_exact_two_sided(left_count, int(left["runs"]), right_count, int(right["runs"]))


def _failure_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], int] = {}
    for record in records:
        groups[(record.mode, record.c_variant, record.failure_type or "unknown")] = groups.get((record.mode, record.c_variant, record.failure_type or "unknown"), 0) + 1
    return [
        {"mode": mode, "c_variant": variant, "failure_type": failure, "rows": rows}
        for (mode, variant, failure), rows in sorted(groups.items())
    ]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_two_sided(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    total_success = a_success + b_success
    total_n = a_n + b_n
    observed = _hypergeom_prob(a_success, a_n, total_success, total_n)
    lo = max(0, total_success - b_n)
    hi = min(a_n, total_success)
    p = 0.0
    for x in range(lo, hi + 1):
        prob = _hypergeom_prob(x, a_n, total_success, total_n)
        if prob <= observed + 1e-12:
            p += prob
    return min(1.0, p)


def _hypergeom_prob(x: int, draws: int, successes: int, total: int) -> float:
    return math.comb(successes, x) * math.comb(total - successes, draws - x) / math.comb(total, draws)


def _rate(values) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _median(values: list[float]) -> float:
    clean = [float(value) for value in values]
    return float(statistics.median(clean)) if clean else 0.0


def main() -> int:
    global CURRENT_EXPERIMENT_PART
    args = parse_args()
    CURRENT_EXPERIMENT_PART = args.experiment_part
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
    initial_records = filter_resume_records(initial_records, tasks=tasks, variants=variants, args=args) if initial_records else []
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
        "experiment_part": args.experiment_part,
        "source_commit": source_commit(),
        "script_hash": generation_path_hash(),
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
