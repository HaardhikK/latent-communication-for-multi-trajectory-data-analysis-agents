from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.agents import (  # noqa: E402
    AgentSettings,
    _build_record,
    _execute_and_score,
    _generate_code,
    _generate_text,
    _latent_append,
    _latent_decode,
    _prepare_run_dir,
    _repair_context,
    _repair_latent_failure,
    _system_code_part,
    _text_reset_repair_parts,
    run_single_agent,
)
from latent_agent.executor import execute_python_code  # noqa: E402
from latent_agent.horizon_tasks import HORIZON_ORDER, selected_horizon_tasks  # noqa: E402
from latent_agent.latent_backend import LatentBackend  # noqa: E402
from latent_agent.metrics import ModelCallRecord, RunRecord, write_csv, write_json  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402
from latent_agent.tasks import ScoreResult, ToyTask  # noqa: E402
from latent_agent.token_split import COORDINATION, FIXED_PROMPT, TOOL_IO, PromptPart, TokenLedger, count_tokens, extract_python_code, render_prompt  # noqa: E402


BASE_BUDGETS = {"short": 512, "medium": 768, "long": 1152}
BUDGET_CAPS = {"short": 768, "medium": 1024, "long": 1536}
HORIZON_LABELS = ["short", "medium", "long"]
MODE_RECORD_NAMES = {"A": "A_single", "B": "B_textmas", "C": "C_latentmas"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 horizon-length sweep.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--mode", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--tasks", default="", help="Comma-separated Phase 3 task ids.")
    parser.add_argument("--families", default="", help="Comma-separated families: orders_kpi,sensor_quality,campaign_roi.")
    parser.add_argument("--horizons", default="", help="Comma-separated horizons: short,medium,long.")
    parser.add_argument("--repeat", type=int, default=5)
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
    parser.add_argument("--output-prefix", default="phase3")
    parser.add_argument("--skip-a-qualification", action="store_true")
    parser.add_argument("--skip-smoke-matrix", action="store_true")
    parser.add_argument("--skip-full-matrix", action="store_true")
    parser.add_argument("--debug-decode-latent", action="store_true")
    parser.add_argument(
        "--resume-from-partial",
        default="",
        help="Optional Phase 3 partial CSV. Completed mode/task/repeat rows are loaded from run_record.json and skipped.",
    )
    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=0,
        help="Run at most this many new schedule rows after resume. Use 0 for the full remaining schedule.",
    )
    return parser.parse_args()


def generation_seed_for_repeat(repeat: int, base_seed: int = 17) -> int:
    if repeat < 1:
        raise ValueError("repeat is one-based; repeat 1 maps to the base seed")
    return int(base_seed) + repeat - 1


def resolve_code_budget(reference_tokens: int, horizon_level: str) -> tuple[int, float]:
    budget = BASE_BUDGETS[horizon_level]
    cap = BUDGET_CAPS[horizon_level]
    while reference_tokens > 0.70 * budget and budget < cap:
        budget = min(cap, ((budget // 128) + 1) * 128)
    ratio = reference_tokens / budget if budget else 0.0
    if ratio > 0.70:
        raise ValueError(
            f"{horizon_level} reference script uses {reference_tokens} tokens, "
            f"which exceeds 70% of capped budget {cap}"
        )
    return budget, ratio


def build_budget_table(tasks: list[ToyTask], tokenizer: Any) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for task in tasks:
        tokens = count_tokens(tokenizer, task.reference_script)
        budget, ratio = resolve_code_budget(tokens, task.horizon_level)
        table[task.task_id] = {
            "task_id": task.task_id,
            "task_family": task.task_family,
            "horizon_level": task.horizon_level,
            "horizon_stages": task.horizon_stages,
            "reference_output_tokens": tokens,
            "max_new_tokens_code": budget,
            "reference_budget_ratio": ratio,
        }
    return table


def make_settings(
    args: argparse.Namespace,
    task: ToyTask,
    budget: dict[str, Any],
    *,
    repeat: int,
    effective_latent_steps: int,
    oom_fallback_used: bool,
) -> AgentSettings:
    return AgentSettings(
        max_new_tokens_code=int(budget["max_new_tokens_code"]),
        max_new_tokens_plan=args.max_new_tokens_plan,
        max_new_tokens_critic=args.max_new_tokens_critic,
        execution_timeout_seconds=args.execution_timeout_seconds,
        allow_repair=True,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_seed=generation_seed_for_repeat(repeat, args.base_generation_seed),
        schedule_seed=args.schedule_seed,
        latent_steps=effective_latent_steps,
        latent_observation_steps=args.latent_observation_steps,
        latent_repair_strategy=args.latent_repair_strategy,
        requested_latent_steps=args.latent_steps,
        effective_latent_steps=effective_latent_steps,
        oom_fallback_used=oom_fallback_used,
        reference_output_tokens=int(budget["reference_output_tokens"]),
        reference_budget_ratio=float(budget["reference_budget_ratio"]),
        coordination_rounds=task.horizon_stages,
    )


def sanitize_planner_report(text: str) -> str:
    """Keep decoded coordination advisory, not executable or schema-replacing."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    kept: list[str] = []
    skip_code = False
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if upper.startswith(("CODE:", "PYTHON:", "SCRIPT:")):
            skip_code = True
            continue
        if skip_code:
            if upper.startswith(("PLAN:", "CHECKLIST:", "STAGE ", "STEP ", "OUTPUT:")):
                skip_code = False
            else:
                continue
        if _looks_like_code_or_fragment(stripped):
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or "Follow the authoritative task specification for this stage."


def _looks_like_code_or_fragment(line: str) -> bool:
    if line in {"```", "```python", "```py"}:
        return True
    if line.startswith(("import ", "from ", "#")):
        return True
    if line.endswith("="):
        return True
    if re.search(r"\b(pd|np|sklearn)\.", line):
        return True
    if re.search(r"\w+\s*=\s*[^=]", line) and not line.startswith(("-", "*")):
        return True
    if "[[" in line or "read_csv(" in line or ".merge(" in line or ".to_csv(" in line or "json.dump" in line:
        return True
    return False


def _planner_prompt_parts(task: ToyTask, index: int, stage: str, previous: str) -> list[PromptPart]:
    return [
        PromptPart(
            "planner_system",
            "You are the planner in a multi-stage planner -> coder -> critic pipeline. "
            "Produce an advisory checklist for only the current stage. "
            "Use at most three short bullets. Do not write code, imports, dataframe column selections, formulas not present in the task, invented columns, or partial snippets. "
            "If details are uncertain, say to follow the authoritative task spec. Begin with PLAN:",
            FIXED_PROMPT,
        ),
        PromptPart("task", task.prompt, FIXED_PROMPT),
        PromptPart("previous_stage_reports", previous, COORDINATION if previous else FIXED_PROMPT),
        PromptPart("current_stage", f"Stage {index}/{task.horizon_stages}: {stage}", FIXED_PROMPT),
    ]


def _authoritative_task_part(task: ToyTask) -> PromptPart:
    return PromptPart(
        "authoritative_task",
        "AUTHORITATIVE TASK SPECIFICATION. Read this last and follow it exactly. "
        "It overrides any planner, critic, latent memory, previous code, or repair context if there is a conflict:\n"
        f"{task.prompt}",
        FIXED_PROMPT,
    )


def _phase3_coder_parts(task: ToyTask, *, mode_label: str, coordination_text: str = "") -> list[PromptPart]:
    parts = [_system_code_part(f"You are the coder in a multi-stage {mode_label} pipeline.")]
    if coordination_text:
        parts.append(
            PromptPart(
                "coordination_context",
                "ADVISORY COORDINATION CONTEXT. Use this only for stage ordering and reminders; "
                "do not copy code or schema from it, and ignore it if it conflicts with the authoritative task spec below.\n"
                f"{coordination_text}",
                COORDINATION,
            )
        )
    else:
        parts.append(
            PromptPart(
                "coordination_context",
                "ADVISORY COORDINATION CONTEXT. Latent planning memory is available in the KV cache. "
                "Use it only for stage ordering; the authoritative task spec below controls exact columns, files, keys, and formulas.",
                FIXED_PROMPT,
            )
        )
    parts.append(
        PromptPart(
            "coordination_grounding",
            "Use coordination only as an ordering aid. The task spec below is authoritative for exact output schema, column names, required values, and file names.",
            FIXED_PROMPT,
        )
    )
    parts.append(_authoritative_task_part(task))
    return parts


def _write_prompt_audit(run_dir: Path, name: str, parts: list[PromptPart]) -> None:
    (run_dir / name).write_text(render_prompt(parts), encoding="utf-8")


def _generate_phase3_text_reset_repair(
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    task: ToyTask,
    code_text: str,
    exec_result,
    score: ScoreResult,
    run_dir: Path,
    settings: AgentSettings,
    *,
    call_name: str,
) -> str:
    parts = _text_reset_repair_parts(task, _repair_context(code_text, exec_result, score))
    _write_prompt_audit(run_dir, f"{call_name}_prompt.txt", parts)
    return _generate_code(
        backend,
        ledger,
        calls,
        call_name,
        parts,
        max_new_tokens=settings.max_new_tokens_code,
        settings=settings,
    )


def run_phase3_textmas(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings: AgentSettings,
) -> RunRecord:
    run_dir = _prepare_run_dir(run_root, "B_textmas", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results = []

    start = time.perf_counter()
    task.setup(run_dir)

    planner_reports: list[str] = []
    for index, stage in enumerate(task.stage_specs, start=1):
        previous = "\n\n".join(f"Stage {i + 1}: {report}" for i, report in enumerate(planner_reports))
        parts = _planner_prompt_parts(task, index, stage, previous)
        report = _generate_text(
            backend,
            ledger,
            calls,
            f"planner_stage_{index}",
            parts,
            output_category=COORDINATION,
            max_new_tokens=settings.max_new_tokens_plan,
            settings=settings,
        )
        sanitized_report = sanitize_planner_report(report)
        planner_reports.append(sanitized_report)
        (run_dir / f"planner_stage_{index}.txt").write_text(report, encoding="utf-8")
        (run_dir / f"planner_stage_{index}_sanitized.txt").write_text(sanitized_report, encoding="utf-8")

    all_reports = "\n\n".join(f"Stage {i + 1}: {report}" for i, report in enumerate(planner_reports))
    coder_parts = _phase3_coder_parts(task, mode_label="planner -> coder -> critic", coordination_text=all_reports)
    _write_prompt_audit(run_dir, "coder_prompt.txt", coder_parts)
    code_text = _generate_code(
        backend,
        ledger,
        calls,
        "coder",
        coder_parts,
        max_new_tokens=settings.max_new_tokens_code,
        settings=settings,
    )
    exec_result, score = _execute_and_score(code_text, task, run_dir, 1, settings.execution_timeout_seconds)
    exec_results.append(exec_result)

    critic_report = _generate_text(
        backend,
        ledger,
        calls,
        "critic",
        [
            PromptPart(
                "critic_system",
                "You are the critic. Check whether the final script and one execution satisfy all stages. "
                "If it failed, give concise repair instructions. Do not write full code. Begin with CRITIC:",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
            PromptPart("planner_stage_reports", all_reports, COORDINATION),
            PromptPart("code_and_execution", _repair_context(code_text, exec_result, score), TOOL_IO),
        ],
        output_category=COORDINATION,
        max_new_tokens=settings.max_new_tokens_critic,
        settings=settings,
    )
    (run_dir / "critic_report.txt").write_text(critic_report, encoding="utf-8")
    attempts = 1

    if settings.allow_repair and not score.passed:
        repair_code = _generate_phase3_text_reset_repair(
            backend,
            ledger,
            calls,
            task,
            code_text,
            exec_result,
            score,
            run_dir,
            settings,
            call_name="coder_repair",
        )
        exec_result, score = _execute_and_score(repair_code, task, run_dir, 2, settings.execution_timeout_seconds)
        exec_results.append(exec_result)
        attempts = 2

    return _build_record(
        run_id=run_dir.name,
        task=task,
        mode="B_textmas",
        model_id=backend.model_id,
        repeat=repeat,
        score=score,
        wall_latency_s=time.perf_counter() - start,
        exec_results=exec_results,
        calls=calls,
        ledger=ledger,
        attempts=attempts,
        run_dir=run_dir,
        settings=settings,
    )


def run_phase3_latentmas(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings: AgentSettings,
    debug_decode_latent: bool = False,
) -> RunRecord:
    run_dir = _prepare_run_dir(run_root, "C_latentmas", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results = []
    latent = LatentBackend(backend)
    past = None

    start = time.perf_counter()
    task.setup(run_dir)

    for index, stage in enumerate(task.stage_specs, start=1):
        past = _latent_append(
            latent,
            backend,
            ledger,
            calls,
            f"latent_planner_stage_{index}",
            [
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
            latent_steps=settings.latent_steps,
            past_key_values=past,
        )

    coder_parts = _phase3_coder_parts(task, mode_label="latent planner -> coder -> critic")
    _write_prompt_audit(run_dir, "latent_coder_prompt.txt", coder_parts)
    code_text, past = _latent_decode(
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
    exec_result, score = _execute_and_score(code_text, task, run_dir, 1, settings.execution_timeout_seconds)
    exec_results.append(exec_result)
    attempts = 1

    past = _latent_append(
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

    past = _latent_append(
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
            repair_code = _generate_phase3_text_reset_repair(
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

    return _build_record(
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


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def run_record(
    backend: ModelBackend,
    task: ToyTask,
    mode: str,
    run_root: Path,
    repeat: int,
    settings: AgentSettings,
    *,
    debug_decode_latent: bool,
) -> RunRecord:
    if mode == "A":
        return run_single_agent(backend, task, run_root, repeat=repeat, settings=settings)
    if mode == "B":
        return run_phase3_textmas(backend, task, run_root, repeat=repeat, settings=settings)
    if mode == "C":
        return run_phase3_latentmas(backend, task, run_root, repeat=repeat, settings=settings, debug_decode_latent=debug_decode_latent)
    raise ValueError(f"Unknown mode: {mode}")


def validate_reference_scripts(tasks: list[ToyTask], batch_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = runtime_path("runs", f"{batch_id}_reference_fit")
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        run_dir = root / task.task_id
        run_dir.mkdir(parents=True, exist_ok=True)
        task.setup(run_dir)
        execution = execute_python_code(task.reference_script, run_dir, attempt=1, timeout_s=30)
        score = task.score(run_dir) if execution.succeeded else ScoreResult(False, 0.0, "reference execution failed", {})
        rows.append(
            {
                "task_id": task.task_id,
                "passed": bool(score.passed),
                "score": float(score.score),
                "message": score.message,
                "run_dir": str(run_dir),
            }
        )
        if not score.passed:
            raise RuntimeError(f"Reference script failed for {task.task_id}: {score.message}; see {run_dir}")
    return rows


def run_a_qualification(
    args: argparse.Namespace,
    backend: ModelBackend,
    tasks: list[ToyTask],
    budget_table: dict[str, dict[str, Any]],
    batch_id: str,
) -> list[RunRecord]:
    root = runtime_path("runs", f"{batch_id}_a_qualification")
    root.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    for task in tasks:
        settings = make_settings(args, task, budget_table[task.task_id], repeat=1, effective_latent_steps=args.latent_steps, oom_fallback_used=False)
        record = run_single_agent(backend, task, root, repeat=1, settings=settings)
        records.append(record)
        print(
            f"QUALIFY_A task={task.task_id} passed={record.passed} score={record.score:.3f} run_dir={record.run_dir}",
            flush=True,
        )
        if not record.passed:
            raise RuntimeError(f"Mode A qualification failed for {task.task_id}; simplify before full sweep. Run: {record.run_dir}")
    return records


def run_c_long_preflight(
    args: argparse.Namespace,
    backend: ModelBackend,
    tasks: list[ToyTask],
    budget_table: dict[str, dict[str, Any]],
    batch_id: str,
) -> tuple[int, bool, dict[str, Any]]:
    long_tasks = [task for task in tasks if task.horizon_level == "long"]
    if not long_tasks:
        return args.latent_steps, False, {"skipped": True, "reason": "no long tasks selected"}
    preflight_task = long_tasks[0]
    root = runtime_path("runs", f"{batch_id}_c_long_preflight")
    root.mkdir(parents=True, exist_ok=True)

    try:
        settings = make_settings(args, preflight_task, budget_table[preflight_task.task_id], repeat=1, effective_latent_steps=args.latent_steps, oom_fallback_used=False)
        record = run_phase3_latentmas(backend, preflight_task, root, repeat=1, settings=settings, debug_decode_latent=False)
        return args.latent_steps, False, {"task_id": preflight_task.task_id, "passed": record.passed, "peak_vram_mb": record.peak_vram_mb, "run_dir": record.run_dir}
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        _clear_cuda_cache(backend)
        settings = make_settings(args, preflight_task, budget_table[preflight_task.task_id], repeat=1, effective_latent_steps=args.fallback_latent_steps, oom_fallback_used=True)
        record = run_phase3_latentmas(backend, preflight_task, root, repeat=2, settings=settings, debug_decode_latent=False)
        return args.fallback_latent_steps, True, {
            "task_id": preflight_task.task_id,
            "passed": record.passed,
            "peak_vram_mb": record.peak_vram_mb,
            "run_dir": record.run_dir,
            "fallback_reason": str(exc)[:500],
        }


def _clear_cuda_cache(backend: ModelBackend) -> None:
    torch = backend.torch
    if str(backend.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_schedule(tasks: list[ToyTask], modes: list[str], repeat: int, schedule_seed: int) -> list[tuple[int, ToyTask, str]]:
    schedule = [(rep, task, mode) for rep in range(1, repeat + 1) for task in tasks for mode in modes]
    random.Random(schedule_seed).shuffle(schedule)
    return schedule


def _record_key(record: RunRecord) -> tuple[str, str, int]:
    return (record.mode, record.task_id, int(record.repeat))


def _schedule_key(repeat: int, task: ToyTask, mode: str) -> tuple[str, str, int]:
    return (MODE_RECORD_NAMES[mode], task.task_id, int(repeat))


def _run_record_from_json(path: Path) -> RunRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_calls"] = [ModelCallRecord(**call) for call in data.get("model_calls", [])]
    allowed = {field.name for field in fields(RunRecord)}
    return RunRecord(**{key: value for key, value in data.items() if key in allowed})


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _run_record_from_flat_row(row: dict[str, str]) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        task_id=row["task_id"],
        mode=row["mode"],
        model_id=row["model_id"],
        repeat=int(row["repeat"]),
        passed=_coerce_bool(row["passed"]),
        score=float(row["score"]),
        message=row["message"],
        wall_latency_s=float(row["wall_latency_s"]),
        model_latency_ms=float(row["model_latency_ms"]),
        code_exec_latency_s=float(row["code_exec_latency_s"]),
        forward_passes=int(row["forward_passes"]),
        generated_tokens=int(row["generated_tokens"]),
        model_input_tokens=int(row["model_input_tokens"]),
        coordination_tokens=int(row["coordination_tokens"]),
        tool_io_tokens=int(row["tool_io_tokens"]),
        fixed_prompt_tokens=int(row["fixed_prompt_tokens"]),
        coordination_fraction=float(row["coordination_fraction"]),
        peak_vram_mb=float(row["peak_vram_mb"]),
        attempts=int(row["attempts"]),
        run_dir=row["run_dir"],
        latent_steps=int(row.get("latent_steps") or 0),
        latent_repair_strategy=row.get("latent_repair_strategy", ""),
        task_family=row.get("task_family", ""),
        horizon_level=row.get("horizon_level", ""),
        horizon_stages=int(row.get("horizon_stages") or 0),
        coordination_rounds=int(row.get("coordination_rounds") or 0),
        generation_seed=int(row.get("generation_seed") or 0),
        schedule_seed=int(row.get("schedule_seed") or 0),
        temperature=float(row.get("temperature") or 0.0),
        top_p=float(row.get("top_p") or 1.0),
        max_new_tokens_code=int(row.get("max_new_tokens_code") or 0),
        reference_output_tokens=int(row.get("reference_output_tokens") or 0),
        reference_budget_ratio=float(row.get("reference_budget_ratio") or 0.0),
        requested_latent_steps=int(row.get("requested_latent_steps") or 0),
        effective_latent_steps=int(row.get("effective_latent_steps") or 0),
        oom_fallback_used=_coerce_bool(row.get("oom_fallback_used", False)),
    )


def load_resume_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Resume CSV does not exist: {path}")
    records_by_key: dict[tuple[str, str, int], RunRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_dir = Path(row.get("run_dir", ""))
            record_path = run_dir / "run_record.json"
            record = _run_record_from_json(record_path) if record_path.exists() else _run_record_from_flat_row(row)
            records_by_key[_record_key(record)] = record
    return list(records_by_key.values())


def run_schedule(
    args: argparse.Namespace,
    backend: ModelBackend,
    tasks: list[ToyTask],
    modes: list[str],
    budget_table: dict[str, dict[str, Any]],
    batch_id: str,
    *,
    effective_latent_steps: int,
    oom_fallback_used: bool,
    suffix: str = "",
    partial_csv_paths: list[Path] | None = None,
    initial_records: list[RunRecord] | None = None,
) -> list[RunRecord]:
    run_root = runtime_path("runs", f"{batch_id}{suffix}")
    run_root.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = list(initial_records or [])
    completed = {_record_key(record) for record in records}
    new_rows = 0
    for repeat, task, mode in build_schedule(tasks, modes, args.repeat, args.schedule_seed):
        key = _schedule_key(repeat, task, mode)
        if key in completed:
            print(
                f"SKIP_RESUMED repeat={repeat} family={task.task_family} "
                f"horizon={task.horizon_level} task={task.task_id} mode={mode}",
                flush=True,
            )
            continue
        settings = make_settings(
            args,
            task,
            budget_table[task.task_id],
            repeat=repeat,
            effective_latent_steps=effective_latent_steps,
            oom_fallback_used=oom_fallback_used,
        )
        print(
            f"RUN repeat={repeat} seed={settings.generation_seed} family={task.task_family} "
            f"horizon={task.horizon_level} task={task.task_id} mode={mode} code_budget={settings.max_new_tokens_code}",
            flush=True,
        )
        record = run_record(backend, task, mode, run_root, repeat, settings, debug_decode_latent=args.debug_decode_latent)
        records.append(record)
        completed.add(key)
        new_rows += 1
        if partial_csv_paths:
            for path in partial_csv_paths:
                write_csv(path, records)
        print(
            json.dumps(
                {
                    "task_id": record.task_id,
                    "mode": record.mode,
                    "repeat": record.repeat,
                    "passed": record.passed,
                    "score": record.score,
                    "model_latency_ms": round(record.model_latency_ms, 1),
                    "coordination_tokens": record.coordination_tokens,
                    "peak_vram_mb": round(record.peak_vram_mb, 1),
                    "run_dir": record.run_dir,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.max_new_rows > 0 and new_rows >= args.max_new_rows:
            print(
                f"CHUNK_LIMIT_REACHED new_rows={new_rows} total_records={len(records)} max_new_rows={args.max_new_rows}",
                flush=True,
            )
            break
    return records


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return float(ordered[lower] * (1 - frac) + ordered[upper] * frac)


def median_iqr(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(statistics.median(values)), float(_percentile(values, 0.75) - _percentile(values, 0.25))


def summarize_phase3(records: list[RunRecord], budget_table: dict[str, dict[str, Any]], plot_dir: Path) -> str:
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_mode_horizon: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        by_mode_horizon[(record.mode, record.horizon_level)].append(record)

    gap_rows = _gap_rows(by_mode_horizon)
    pass_rows = _pass_rate_rows(by_mode_horizon)
    baseline_rows = _baseline_validity_rows(by_mode_horizon)
    coord_fraction_rows = _coord_fraction_rows(by_mode_horizon)
    vram_rows = _vram_rows(by_mode_horizon)
    long_c_rows = [record for record in records if record.mode == "C_latentmas" and record.horizon_level == "long"]

    plots = {
        "coord_gap": plot_dir / "bc_coordination_token_gap.svg",
        "model_gap": plot_dir / "bc_model_latency_gap.svg",
        "forward_gap": plot_dir / "bc_forward_pass_gap.svg",
        "coord_fraction": plot_dir / "coordination_fraction_by_horizon.svg",
        "pass_rate": plot_dir / "pass_rate_by_horizon.svg",
        "peak_vram": plot_dir / "peak_vram_by_horizon.svg",
    }
    _write_svg_line(plots["coord_gap"], {"B-C decoded coordination tokens": [row["coordination_token_gap"] for row in gap_rows]}, "B-C decoded coordination-token gap vs horizon", "tokens")
    _write_svg_line(plots["model_gap"], {"B-C model latency": [row["model_latency_gap_ms"] for row in gap_rows]}, "B-C model-only latency gap vs horizon", "ms")
    _write_svg_line(plots["forward_gap"], {"B-C forward passes": [row["forward_pass_gap"] for row in gap_rows]}, "B-C forward-pass gap vs horizon", "passes")
    _write_svg_line(
        plots["coord_fraction"],
        {
            mode: [row["coordination_fraction"] for row in coord_fraction_rows if row["mode"] == mode]
            for mode in ("A_single", "B_textmas", "C_latentmas")
        },
        "Coordination-token fraction vs horizon",
        "fraction",
    )
    _write_svg_line(
        plots["pass_rate"],
        {mode: [row["pass_rate"] for row in pass_rows if row["mode"] == mode] for mode in ("A_single", "B_textmas", "C_latentmas")},
        "Pass rate vs horizon",
        "pass rate",
    )
    _write_svg_line(
        plots["peak_vram"],
        {mode: [row["median_peak_vram_mb"] for row in vram_rows if row["mode"] == mode] for mode in ("A_single", "B_textmas", "C_latentmas")},
        "Peak VRAM by mode/horizon",
        "MB",
    )

    lines = [
        "# Phase 3 Horizon Sweep Summary",
        "",
        "This experiment is explicitly a **multi-stage planning-coordination horizon** sweep with **one code execution at the end**. "
        "It tests whether latent coordination saves more decoded coordination work as the planning horizon grows, while holding the execute-once tool loop fixed.",
        "",
        "A per-stage execution variant, `execute -> observe -> continue` at each stage, is the stronger Phase 4 / 8B-tier follow-up test.",
        "",
        f"- Runs summarized: `{len(records)}`",
        f"- Expected full matrix rows: `3 modes * 3 families * 3 horizons * 5 repeats = 135` when defaults are used.",
        "",
        "## Plain-English Readout",
        "",
        _plain_english_readout(gap_rows, pass_rows),
        "",
        "## B-C Gap Vs Horizon",
        "",
        "| Horizon | Stages | Median B coord tokens | Median C coord tokens | B-C coord gap | Median B model ms | Median C model ms | B-C model gap ms | Median B forward | Median C forward | B-C forward gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['horizon_level']} | {row['horizon_stages']} | {row['b_coord']:.0f} | {row['c_coord']:.0f} | {row['coordination_token_gap']:.0f} | "
            f"{row['b_model_ms']:.1f} | {row['c_model_ms']:.1f} | {row['model_latency_gap_ms']:.1f} | "
            f"{row['b_forward']:.0f} | {row['c_forward']:.0f} | {row['forward_pass_gap']:.0f} |"
        )

    lines.extend(["", "## Coordination Fraction", "", "| Mode | Horizon | Runs | Median coordination fraction | IQR |", "|---|---|---:|---:|---:|"])
    for row in coord_fraction_rows:
        lines.append(f"| {row['mode']} | {row['horizon_level']} | {row['runs']} | {row['coordination_fraction']:.3f} | {row['iqr']:.3f} |")

    lines.extend(["", "## Pass Rate", "", "| Mode | Horizon | Runs | Pass rate |", "|---|---|---:|---:|"])
    for row in pass_rows:
        lines.append(f"| {row['mode']} | {row['horizon_level']} | {row['runs']} | {row['pass_rate']:.3f} |")

    lines.extend(
        [
            "",
            "## Text Baseline Validity Gate",
            "",
            "B is a valid text baseline only if it is within `0.15` absolute pass-rate of A for each run horizon and is not `0/N`.",
            "",
            "| Horizon | A runs | A pass rate | B runs | B pass rate | A-B gap | Valid text baseline? |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in baseline_rows:
        valid = "yes" if row["valid"] else ("not run" if not row["has_data"] else "no")
        lines.append(
            f"| {row['horizon_level']} | {row['a_runs']} | {row['a_pass_rate']:.3f} | "
            f"{row['b_runs']} | {row['b_pass_rate']:.3f} | {row['gap']:.3f} | {valid} |"
        )

    lines.extend(["", "## Peak VRAM", "", "| Mode | Horizon | Runs | Median peak VRAM MB | IQR |", "|---|---|---:|---:|---:|"])
    for row in vram_rows:
        lines.append(f"| {row['mode']} | {row['horizon_level']} | {row['runs']} | {row['median_peak_vram_mb']:.1f} | {row['iqr']:.1f} |")

    lines.extend(["", "### 7-Stage Mode C Peak VRAM Runs", "", "| Task | Repeat | Passed | Effective latent steps | Peak VRAM MB | Run dir |", "|---|---:|---:|---:|---:|---|"])
    for record in sorted(long_c_rows, key=lambda r: (r.task_id, r.repeat)):
        lines.append(f"| {record.task_id} | {record.repeat} | {record.passed} | {record.effective_latent_steps} | {record.peak_vram_mb:.1f} | `{record.run_dir}` |")

    lines.extend(["", "## Code Budget Fit", "", "| Task | Family | Horizon | Stages | Reference tokens | Assigned code budget | Ratio |", "|---|---|---|---:|---:|---:|---:|"])
    for row in sorted(budget_table.values(), key=lambda item: (item["task_family"], HORIZON_ORDER[item["horizon_level"]])):
        lines.append(
            f"| {row['task_id']} | {row['task_family']} | {row['horizon_level']} | {row['horizon_stages']} | "
            f"{row['reference_output_tokens']} | {row['max_new_tokens_code']} | {row['reference_budget_ratio']:.3f} |"
        )

    lines.extend(["", "## Plots", ""])
    for label, path in plots.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _gap_rows(by_mode_horizon: dict[tuple[str, str], list[RunRecord]]) -> list[dict[str, float]]:
    rows = []
    for horizon in HORIZON_LABELS:
        b = by_mode_horizon.get(("B_textmas", horizon), [])
        c = by_mode_horizon.get(("C_latentmas", horizon), [])
        b_coord, _ = median_iqr([float(r.coordination_tokens) for r in b])
        c_coord, _ = median_iqr([float(r.coordination_tokens) for r in c])
        b_model, _ = median_iqr([r.model_latency_ms for r in b])
        c_model, _ = median_iqr([r.model_latency_ms for r in c])
        b_forward, _ = median_iqr([float(r.forward_passes) for r in b])
        c_forward, _ = median_iqr([float(r.forward_passes) for r in c])
        rows.append(
            {
                "horizon_level": horizon,
                "horizon_stages": HORIZON_ORDER[horizon],
                "b_coord": b_coord,
                "c_coord": c_coord,
                "coordination_token_gap": b_coord - c_coord,
                "b_model_ms": b_model,
                "c_model_ms": c_model,
                "model_latency_gap_ms": b_model - c_model,
                "b_forward": b_forward,
                "c_forward": c_forward,
                "forward_pass_gap": b_forward - c_forward,
            }
        )
    return rows


def _pass_rate_rows(by_mode_horizon: dict[tuple[str, str], list[RunRecord]]) -> list[dict[str, Any]]:
    rows = []
    for mode in ("A_single", "B_textmas", "C_latentmas"):
        for horizon in HORIZON_LABELS:
            subset = by_mode_horizon.get((mode, horizon), [])
            rows.append({"mode": mode, "horizon_level": horizon, "runs": len(subset), "pass_rate": sum(r.passed for r in subset) / len(subset) if subset else 0.0})
    return rows


def _baseline_validity_rows(by_mode_horizon: dict[tuple[str, str], list[RunRecord]]) -> list[dict[str, Any]]:
    rows = []
    for horizon in HORIZON_LABELS:
        a = by_mode_horizon.get(("A_single", horizon), [])
        b = by_mode_horizon.get(("B_textmas", horizon), [])
        a_rate = sum(r.passed for r in a) / len(a) if a else 0.0
        b_rate = sum(r.passed for r in b) / len(b) if b else 0.0
        gap = a_rate - b_rate
        has_data = bool(a and b)
        rows.append(
            {
                "horizon_level": horizon,
                "a_runs": len(a),
                "a_pass_rate": a_rate,
                "b_runs": len(b),
                "b_pass_rate": b_rate,
                "gap": gap,
                "has_data": has_data,
                "valid": has_data and b_rate > 0.0 and abs(gap) <= 0.15,
            }
        )
    return rows


def _coord_fraction_rows(by_mode_horizon: dict[tuple[str, str], list[RunRecord]]) -> list[dict[str, Any]]:
    rows = []
    for mode in ("A_single", "B_textmas", "C_latentmas"):
        for horizon in HORIZON_LABELS:
            subset = by_mode_horizon.get((mode, horizon), [])
            med, iqr = median_iqr([r.coordination_fraction for r in subset])
            rows.append({"mode": mode, "horizon_level": horizon, "runs": len(subset), "coordination_fraction": med, "iqr": iqr})
    return rows


def _vram_rows(by_mode_horizon: dict[tuple[str, str], list[RunRecord]]) -> list[dict[str, Any]]:
    rows = []
    for mode in ("A_single", "B_textmas", "C_latentmas"):
        for horizon in HORIZON_LABELS:
            subset = by_mode_horizon.get((mode, horizon), [])
            med, iqr = median_iqr([r.peak_vram_mb for r in subset])
            rows.append({"mode": mode, "horizon_level": horizon, "runs": len(subset), "median_peak_vram_mb": med, "iqr": iqr})
    return rows


def _plain_english_readout(gap_rows: list[dict[str, float]], pass_rows: list[dict[str, Any]]) -> str:
    available_gaps = [
        row
        for row in gap_rows
        if any(pass_row["horizon_level"] == row["horizon_level"] and pass_row["runs"] for pass_row in pass_rows)
    ]
    if not available_gaps:
        return "Phase 3 did not produce enough B/C horizon data for a readout."
    short_gap = available_gaps[0]
    widest_gap = available_gaps[-1]
    c_pass_widest = next((row for row in pass_rows if row["mode"] == "C_latentmas" and row["horizon_level"] == widest_gap["horizon_level"]), None)
    pass_lookup = {(row["mode"], row["horizon_level"]): row for row in pass_rows}
    b_validity = []
    for horizon in HORIZON_LABELS:
        a_row = pass_lookup.get(("A_single", horizon), {"runs": 0, "pass_rate": 0.0})
        b_row = pass_lookup.get(("B_textmas", horizon), {"runs": 0, "pass_rate": 0.0})
        a_runs = int(a_row["runs"])
        b_runs = int(b_row["runs"])
        gap = abs(float(a_row["pass_rate"]) - float(b_row["pass_rate"]))
        b_validity.append(
            {
                "horizon_level": horizon,
                "a_runs": a_runs,
                "b_runs": b_runs,
                "valid_baseline": bool(a_runs and b_runs and gap <= 0.15 and float(b_row["pass_rate"]) > 0.0),
            }
        )
    invalid_horizons = [
        row["horizon_level"]
        for row in b_validity
        if row["a_runs"] and row["b_runs"] and not row["valid_baseline"]
    ]
    validity_note = ""
    if invalid_horizons:
        validity_note = (
            f" The text baseline validity gate failed at {', '.join(invalid_horizons)}, "
            "so accuracy should not be interpreted as a fair latent-vs-text channel result yet."
        )
    if len(available_gaps) == 1:
        return (
            "On small local hardware, this short-gate run asks whether the repaired text multi-agent baseline is valid before rerunning "
            "the full horizon sweep. "
            f"At the {short_gap['horizon_level']} ({short_gap['horizon_stages']}-stage) horizon, B used "
            f"{short_gap['b_coord']:.0f} median decoded coordination tokens while C used "
            f"{short_gap['c_coord']:.0f}, and C pass rate was "
            f"{(c_pass_widest or {'pass_rate': 0.0})['pass_rate']:.3f}.{validity_note}"
        )
    return (
        "On small local hardware, the sweep asks whether latent inter-agent planning becomes more valuable as tasks require more dependent stages. "
        f"In this run, the decoded coordination-token gap moved from {short_gap['coordination_token_gap']:.0f} tokens at the short horizon "
        f"to {widest_gap['coordination_token_gap']:.0f} tokens at the {widest_gap['horizon_stages']}-stage horizon, while the latent pass rate there was "
        f"{(c_pass_widest or {'pass_rate': 0.0})['pass_rate']:.3f}. This directly tests the proposed efficiency/accuracy tradeoff before scaling to larger models."
        f"{validity_note}"
    )


def _write_svg_line(path: Path, series: dict[str, list[float]], title: str, ylabel: str) -> None:
    width, height = 720, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 45, 65
    x_labels = HORIZON_LABELS
    usable_w = width - margin_left - margin_right
    usable_h = height - margin_top - margin_bottom
    all_values = [value for values in series.values() for value in values]
    y_min = min(0.0, min(all_values) if all_values else 0.0)
    y_max = max(all_values) if all_values else 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]

    def x_at(i: int) -> float:
        return margin_left + i * usable_w / max(1, len(x_labels) - 1)

    def y_at(value: float) -> float:
        return margin_top + (y_max - value) * usable_h / (y_max - y_min)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='24' text-anchor='middle' font-family='Arial' font-size='18' font-weight='700'>{_xml(title)}</text>",
        f"<text x='18' y='{height/2}' transform='rotate(-90 18 {height/2})' text-anchor='middle' font-family='Arial' font-size='12'>{_xml(ylabel)}</text>",
        f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-margin_right}' y2='{height-margin_bottom}' stroke='#111827'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' stroke='#111827'/>",
    ]
    for i, label in enumerate(x_labels):
        x = x_at(i)
        parts.append(f"<text x='{x}' y='{height-margin_bottom+24}' text-anchor='middle' font-family='Arial' font-size='12'>{label}</text>")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_min + (y_max - y_min) * frac
        y = y_at(value)
        parts.append(f"<line x1='{margin_left}' y1='{y}' x2='{width-margin_right}' y2='{y}' stroke='#e5e7eb'/>")
        parts.append(f"<text x='{margin_left-8}' y='{y+4}' text-anchor='end' font-family='Arial' font-size='11'>{value:.2f}</text>")
    for idx, (name, values) in enumerate(series.items()):
        if not values:
            continue
        color = colors[idx % len(colors)]
        coords = " ".join(f"{x_at(i)},{y_at(value)}" for i, value in enumerate(values[: len(x_labels)]))
        parts.append(f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='3'/>")
        for i, value in enumerate(values[: len(x_labels)]):
            parts.append(f"<circle cx='{x_at(i)}' cy='{y_at(value)}' r='4' fill='{color}'/>")
        legend_y = margin_top + 20 + idx * 22
        parts.append(f"<rect x='{width-230}' y='{legend_y-10}' width='12' height='12' fill='{color}'/>")
        parts.append(f"<text x='{width-212}' y='{legend_y}' font-family='Arial' font-size='12'>{_xml(name)}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    args = parse_args()
    configure_runtime(create=True)
    task_ids = [part.strip() for part in args.tasks.split(",") if part.strip()] or None
    families = [part.strip() for part in args.families.split(",") if part.strip()] or None
    horizons = [part.strip() for part in args.horizons.split(",") if part.strip()] or None
    tasks = selected_horizon_tasks(task_ids=task_ids, families=families, horizons=horizons)
    modes = ["A", "B", "C"] if args.mode == "all" else [args.mode]

    backend = ModelBackend(args.model, quantization=args.quantization)
    batch_id = dt.datetime.now(dt.timezone.utc).strftime("phase3_%Y%m%dT%H%M%SZ")
    budget_table = build_budget_table(tasks, backend.tokenizer)
    reference_rows = validate_reference_scripts(tasks, batch_id)
    print("REFERENCE_FIT_OK", json.dumps(reference_rows, sort_keys=True), flush=True)

    qualification_records: list[RunRecord] = []
    if not args.skip_a_qualification:
        qualification_records = run_a_qualification(args, backend, tasks, budget_table, batch_id)

    effective_latent_steps, oom_fallback_used, preflight = run_c_long_preflight(args, backend, tasks, budget_table, batch_id)
    print(
        "C_LONG_PREFLIGHT",
        json.dumps({"effective_latent_steps": effective_latent_steps, "oom_fallback_used": oom_fallback_used, **preflight}, sort_keys=True),
        flush=True,
    )

    smoke_records: list[RunRecord] = []
    if not args.skip_smoke_matrix and tasks:
        smoke_family = tasks[0].task_family
        smoke_tasks = [task for task in tasks if task.task_family == smoke_family]
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.repeat = 1
        smoke_records = run_schedule(
            smoke_args,
            backend,
            smoke_tasks,
            modes,
            budget_table,
            batch_id,
            effective_latent_steps=effective_latent_steps,
            oom_fallback_used=oom_fallback_used,
            suffix="_smoke",
        )
        if any(not record.passed for record in smoke_records if record.mode == "A_single"):
            raise RuntimeError("Smoke matrix had a Mode A failure; stop before full randomized matrix.")

    records: list[RunRecord] = []
    if not args.skip_full_matrix:
        resume_records: list[RunRecord] = []
        partial_csv_path = project_path("exports", f"{args.output_prefix}_metrics.partial.csv")
        if args.resume_from_partial:
            resume_records = load_resume_records(Path(args.resume_from_partial))
            print(f"RESUME_LOADED rows={len(resume_records)} path={args.resume_from_partial}", flush=True)
        try:
            records = run_schedule(
                args,
                backend,
                tasks,
                modes,
                budget_table,
                batch_id,
                effective_latent_steps=effective_latent_steps,
                oom_fallback_used=oom_fallback_used,
                suffix="",
                partial_csv_paths=[
                    partial_csv_path,
                    runtime_path("exports", f"{batch_id}_metrics.partial.csv"),
                ],
                initial_records=resume_records,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or effective_latent_steps == args.fallback_latent_steps:
                raise
            print(f"FULL_MATRIX_OOM_RETRY fallback_latent_steps={args.fallback_latent_steps}: {exc}", flush=True)
            _clear_cuda_cache(backend)
            effective_latent_steps = args.fallback_latent_steps
            oom_fallback_used = True
            retry_records = load_resume_records(partial_csv_path) if partial_csv_path.exists() else resume_records
            records = run_schedule(
                args,
                backend,
                tasks,
                modes,
                budget_table,
                batch_id,
                effective_latent_steps=effective_latent_steps,
                oom_fallback_used=oom_fallback_used,
                suffix="_fallback2",
                partial_csv_paths=[
                    partial_csv_path,
                    runtime_path("exports", f"{batch_id}_metrics.partial.csv"),
                ],
                initial_records=retry_records,
            )

    prefix = args.output_prefix
    write_csv(project_path("exports", f"{prefix}_metrics.csv"), records)
    write_csv(runtime_path("exports", f"{batch_id}_metrics.csv"), records)
    write_json(project_path("exports", f"{prefix}_budget_fit.json"), budget_table)
    write_json(
        project_path("exports", f"{prefix}_run_manifest.json"),
        {
            "batch_id": batch_id,
            "model": args.model,
            "quantization": args.quantization,
            "modes": modes,
            "repeat": args.repeat,
            "base_generation_seed": args.base_generation_seed,
            "repeat_seed_policy": "generation_seed = base_generation_seed + repeat - 1",
            "schedule_seed": args.schedule_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "requested_latent_steps": args.latent_steps,
            "effective_latent_steps": effective_latent_steps,
            "oom_fallback_used": oom_fallback_used,
            "latent_repair_strategy": args.latent_repair_strategy,
            "max_new_rows": args.max_new_rows,
            "reference_fit": reference_rows,
            "a_qualification_runs": [record.to_flat_dict() for record in qualification_records],
            "c_long_preflight": preflight,
            "smoke_runs": [record.to_flat_dict() for record in smoke_records],
            "full_matrix_rows": len(records),
            "expected_full_matrix_rows": len(tasks) * len(modes) * args.repeat,
            "full_matrix_complete": len(records) >= len(tasks) * len(modes) * args.repeat if not args.skip_full_matrix else False,
            "budget_table": budget_table,
        },
    )
    summary = summarize_phase3(records, budget_table, project_path("reports", "phase3_plots"))
    project_path("reports", f"{prefix}_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
