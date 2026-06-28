from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .executor import ExecutionResult, execute_python_code
from .latent_backend import LatentBackend
from .metrics import ModelCallRecord, RunRecord, write_json
from .models import ModelBackend
from .tasks import ScoreResult, ToyTask
from .token_split import COORDINATION, FIXED_PROMPT, TOOL_IO, PromptPart, TokenLedger, extract_python_code, render_prompt


LATENT_REPAIR_STRATEGIES = ("latent", "text_reset", "text_keep_latent", "latent_reset")


@dataclass
class AgentSettings:
    max_new_tokens_code: int = 640
    max_new_tokens_plan: int = 160
    max_new_tokens_critic: int = 220
    execution_timeout_seconds: int = 30
    allow_repair: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    generation_seed: int | None = None
    schedule_seed: int = 0
    latent_steps: int = 4
    latent_observation_steps: int = 1
    debug_decode_tokens: int = 80
    latent_repair_strategy: str = "latent"
    requested_latent_steps: int = 4
    effective_latent_steps: int = 4
    oom_fallback_used: bool = False
    reference_output_tokens: int = 0
    reference_budget_ratio: float = 0.0
    coordination_rounds: int = 0


def run_single_agent(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings: AgentSettings,
) -> RunRecord:
    run_dir = _prepare_run_dir(run_root, "A_single", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results: list[ExecutionResult] = []

    start = time.perf_counter()
    task.setup(run_dir)

    code_text = _generate_code(
        backend,
        ledger,
        calls,
        "single_code",
        [
            _system_code_part(),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        max_new_tokens=settings.max_new_tokens_code,
        settings=settings,
    )
    exec_result, score = _execute_and_score(code_text, task, run_dir, 1, settings.execution_timeout_seconds)
    exec_results.append(exec_result)
    attempts = 1

    if settings.allow_repair and not score.passed:
        repair_text = _generate_code(
            backend,
            ledger,
            calls,
            "single_repair",
            [
                _system_code_part(),
                PromptPart("task", task.prompt, FIXED_PROMPT),
                PromptPart("repair_context", _repair_context(code_text, exec_result, score), TOOL_IO),
                PromptPart(
                    "repair_instruction",
                    "Use the traceback, stdout/stderr, and scorer message as authoritative. "
                    "Fix the concrete runtime or scoring failure, then audit the entire script against the task output schema. "
                    "Ensure every module used by the corrected script is imported. "
                    "If the corrected script writes JSON, include import json at the top, cast pandas/numpy scalar values with int(), float(), or str(), and write one JSON object with the requested keys. "
                    "Return only the full corrected Python code.",
                    FIXED_PROMPT,
                ),
            ],
            max_new_tokens=settings.max_new_tokens_code,
            settings=settings,
        )
        exec_result, score = _execute_and_score(repair_text, task, run_dir, 2, settings.execution_timeout_seconds)
        exec_results.append(exec_result)
        attempts = 2

    wall_latency_s = time.perf_counter() - start
    return _build_record(
        run_id=run_dir.name,
        task=task,
        mode="A_single",
        model_id=backend.model_id,
        repeat=repeat,
        score=score,
        wall_latency_s=wall_latency_s,
        exec_results=exec_results,
        calls=calls,
        ledger=ledger,
        attempts=attempts,
        run_dir=run_dir,
        settings=settings,
    )


def run_textmas(
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
    exec_results: list[ExecutionResult] = []

    start = time.perf_counter()
    task.setup(run_dir)

    planner_report = _generate_text(
        backend,
        ledger,
        calls,
        "planner",
        [
            PromptPart(
                "planner_system",
                "You are the planner in a planner -> coder -> critic pipeline. "
                "Give a concise plain-English data-analysis plan. Do not write code. "
                "Do not use markdown fences. Do not include imports. Begin with PLAN:",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        output_category=COORDINATION,
        max_new_tokens=settings.max_new_tokens_plan,
        settings=settings,
    )

    (run_dir / "planner_report.txt").write_text(planner_report, encoding="utf-8")

    code_text = _generate_code(
        backend,
        ledger,
        calls,
        "coder",
        [
            _system_code_part("You are the coder in a planner -> coder -> critic pipeline."),
            PromptPart("task", task.prompt, FIXED_PROMPT),
            PromptPart("planner_report", planner_report, COORDINATION),
        ],
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
                "You are the critic in a planner -> coder -> critic pipeline. "
                "Check whether the code and execution satisfy the task. "
                "If it failed, give concise repair instructions. Do not write full code. "
                "Do not use markdown fences. Begin with CRITIC:",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
            PromptPart("planner_report", planner_report, COORDINATION),
            PromptPart("code_and_execution", _repair_context(code_text, exec_result, score), TOOL_IO),
        ],
        output_category=COORDINATION,
        max_new_tokens=settings.max_new_tokens_critic,
        settings=settings,
    )
    (run_dir / "critic_report.txt").write_text(critic_report, encoding="utf-8")
    attempts = 1

    if settings.allow_repair and not score.passed:
        repair_code = _generate_code(
            backend,
            ledger,
            calls,
            "coder_repair",
            [
                _system_code_part("You are the coder repairing your previous script."),
                PromptPart("task", task.prompt, FIXED_PROMPT),
                PromptPart("planner_report", planner_report, COORDINATION),
                PromptPart("critic_report", critic_report, COORDINATION),
                PromptPart("previous_code_and_execution", _repair_context(code_text, exec_result, score), TOOL_IO),
            ],
            max_new_tokens=settings.max_new_tokens_code,
            settings=settings,
        )
        exec_result, score = _execute_and_score(repair_code, task, run_dir, 2, settings.execution_timeout_seconds)
        exec_results.append(exec_result)
        attempts = 2

    wall_latency_s = time.perf_counter() - start
    return _build_record(
        run_id=run_dir.name,
        task=task,
        mode="B_textmas",
        model_id=backend.model_id,
        repeat=repeat,
        score=score,
        wall_latency_s=wall_latency_s,
        exec_results=exec_results,
        calls=calls,
        ledger=ledger,
        attempts=attempts,
        run_dir=run_dir,
        settings=settings,
    )


def run_latentmas_agent(
    backend: ModelBackend,
    task: ToyTask,
    run_root: Path,
    *,
    repeat: int,
    settings: AgentSettings,
    debug_decode_latent: bool = False,
) -> RunRecord:
    if settings.latent_repair_strategy not in LATENT_REPAIR_STRATEGIES:
        raise ValueError(f"Unknown latent repair strategy: {settings.latent_repair_strategy}")

    run_dir = _prepare_run_dir(run_root, "C_latentmas", task, repeat)
    ledger = TokenLedger()
    calls: list[ModelCallRecord] = []
    exec_results: list[ExecutionResult] = []
    latent = LatentBackend(backend)
    past = None

    start = time.perf_counter()
    task.setup(run_dir)

    past = _latent_append(
        latent,
        backend,
        ledger,
        calls,
        "latent_planner",
        [
            PromptPart(
                "planner_system",
                "You are the planner in a planner -> coder -> critic pipeline. "
                "Think through a concise data-analysis plan internally. Do not emit text.",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        latent_steps=settings.latent_steps,
        past_key_values=past,
    )

    code_text, past = _latent_decode(
        latent,
        backend,
        ledger,
        calls,
        "latent_coder_decode",
        [
            _system_code_part("You are the coder in a planner -> coder -> critic latent pipeline."),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
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
                "You are the critic in a planner -> coder -> critic latent pipeline. "
                "Update latent memory with whether the code satisfies the task. Do not emit text.",
                FIXED_PROMPT,
            ),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        latent_steps=settings.latent_steps,
        past_key_values=past,
    )

    if settings.allow_repair and not score.passed:
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

    wall_latency_s = time.perf_counter() - start
    return _build_record(
        run_id=run_dir.name,
        task=task,
        mode="C_latentmas",
        model_id=backend.model_id,
        repeat=repeat,
        score=score,
        wall_latency_s=wall_latency_s,
        exec_results=exec_results,
        calls=calls,
        ledger=ledger,
        attempts=attempts,
        run_dir=run_dir,
        latent_repair_strategy=settings.latent_repair_strategy,
        settings=settings,
    )


def _system_code_part(prefix: str = "You are a Python data-analysis agent.") -> PromptPart:
    return PromptPart(
        "code_system",
        (
            f"{prefix}\n"
            "Write one complete Python script for the task. "
            "The script runs from the current directory. "
            "Use pandas, numpy, and scikit-learn if helpful. "
            "If the task writes a JSON file, include `import json` at the top and cast pandas/numpy scalar values with int(), float(), or str() before json.dump. "
            "Create the required output file exactly as requested. "
            "The task instructions and required output schema are authoritative; if a planner or critic report conflicts, follow the task. "
            "Return only raw Python code, with no markdown fences and no explanation. "
            "The first line must be a valid Python statement such as `import pandas as pd`. "
            "Do not list rules or constraints."
        ),
        FIXED_PROMPT,
    )


def _generate_text(
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    call_name: str,
    parts: list[PromptPart],
    *,
    output_category: str,
    max_new_tokens: int,
    settings: AgentSettings | None = None,
) -> str:
    ledger.add_prompt_parts(backend.tokenizer, call_name, parts)
    result = backend.generate(
        render_prompt(parts),
        max_new_tokens=max_new_tokens,
        temperature=settings.temperature if settings else 0.0,
        top_p=settings.top_p if settings else 1.0,
        generation_seed=settings.generation_seed if settings else None,
    )
    ledger.add_generated(backend.tokenizer, call_name, result.text, output_category)
    calls.append(ModelCallRecord(call_name=call_name, **result.metrics.__dict__))
    return result.text.strip()


def _generate_code(
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    call_name: str,
    parts: list[PromptPart],
    *,
    max_new_tokens: int,
    settings: AgentSettings | None = None,
) -> str:
    text = _generate_text(
        backend,
        ledger,
        calls,
        call_name,
        parts,
        output_category=TOOL_IO,
        max_new_tokens=max_new_tokens,
        settings=settings,
    )
    return extract_python_code(text)


def _latent_append(
    latent: LatentBackend,
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    call_name: str,
    parts: list[PromptPart],
    *,
    latent_steps: int,
    past_key_values,
):
    ledger.add_prompt_parts(backend.tokenizer, call_name, parts)
    result = latent.append_latent(
        render_prompt(parts),
        latent_steps=latent_steps,
        past_key_values=past_key_values,
    )
    calls.append(ModelCallRecord(call_name=call_name, **result.metrics.__dict__))
    return result.past_key_values


def _latent_decode(
    latent: LatentBackend,
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    call_name: str,
    parts: list[PromptPart],
    *,
    output_category: str,
    max_new_tokens: int,
    past_key_values,
    settings: AgentSettings | None = None,
) -> tuple[str, object]:
    ledger.add_prompt_parts(backend.tokenizer, call_name, parts)
    result = latent.decode_from_past(
        render_prompt(parts),
        max_new_tokens=max_new_tokens,
        past_key_values=past_key_values,
        temperature=settings.temperature if settings else 0.0,
        top_p=settings.top_p if settings else 1.0,
        generation_seed=settings.generation_seed if settings else None,
    )
    ledger.add_generated(backend.tokenizer, call_name, result.text, output_category)
    calls.append(ModelCallRecord(call_name=call_name, **result.metrics.__dict__))
    return result.text.strip(), result.past_key_values


def _repair_latent_failure(
    strategy: str,
    latent: LatentBackend,
    backend: ModelBackend,
    ledger: TokenLedger,
    calls: list[ModelCallRecord],
    task: ToyTask,
    code_text: str,
    exec_result: ExecutionResult,
    score: ScoreResult,
    past,
    settings: AgentSettings,
) -> tuple[str, object]:
    repair_context = _repair_context(code_text, exec_result, score)
    if strategy == "latent":
        return _latent_decode(
            latent,
            backend,
            ledger,
            calls,
            "latent_coder_repair_decode",
            [
                _system_code_part("You are the coder repairing your previous script from latent critic memory."),
                PromptPart("task", task.prompt, FIXED_PROMPT),
            ],
            output_category=TOOL_IO,
            max_new_tokens=settings.max_new_tokens_code,
            past_key_values=past,
            settings=settings,
        )

    if strategy == "text_reset":
        repair_code = _generate_code(
            backend,
            ledger,
            calls,
            "latent_text_reset_repair",
            _text_reset_repair_parts(task, repair_context),
            max_new_tokens=settings.max_new_tokens_code,
            settings=settings,
        )
        return repair_code, past

    if strategy == "text_keep_latent":
        return _latent_decode(
            latent,
            backend,
            ledger,
            calls,
            "latent_text_grounded_repair_decode",
            [
                _system_code_part("You are the coder repairing your previous script from latent memory and explicit error text."),
                PromptPart("task", task.prompt, FIXED_PROMPT),
                PromptPart("repair_context", repair_context, TOOL_IO),
                PromptPart(
                    "repair_instruction",
                    "The explicit traceback and scorer feedback are authoritative if they conflict with latent memory. "
                    "After fixing the concrete error, audit the entire script against the task output schema. "
                    "Do not preserve a previous output-writing pattern if it conflicts with the requested file format or keys. "
                    "Ensure every module used by the corrected script is imported. "
                    "If the corrected script uses json.dump or json.dumps, include import json at the top. "
                    "Return only the full corrected Python code.",
                    FIXED_PROMPT,
                ),
            ],
            output_category=TOOL_IO,
            max_new_tokens=settings.max_new_tokens_code,
            past_key_values=past,
            settings=settings,
        )

    if strategy == "latent_reset":
        repair_past = _latent_append(
            latent,
            backend,
            ledger,
            calls,
            "latent_reset_repair_observation",
            [
                PromptPart(
                    "repair_observation_system",
                    "Ingest this failed code execution result into a fresh latent repair memory. Do not emit text.",
                    FIXED_PROMPT,
                ),
                PromptPart("repair_context", repair_context, TOOL_IO),
            ],
            latent_steps=max(1, settings.latent_observation_steps),
            past_key_values=None,
        )
        return _latent_decode(
            latent,
            backend,
            ledger,
            calls,
            "latent_reset_repair_decode",
            [
                _system_code_part("You are the coder repairing a failed script from fresh latent error memory."),
                PromptPart("task", task.prompt, FIXED_PROMPT),
            ],
            output_category=TOOL_IO,
            max_new_tokens=settings.max_new_tokens_code,
            past_key_values=repair_past,
            settings=settings,
        )

    raise ValueError(f"Unknown latent repair strategy: {strategy}")


def _text_reset_repair_parts(task: ToyTask, repair_context: str) -> list[PromptPart]:
    return [
        _system_code_part("You are the coder repairing a failed Python data-analysis script."),
        PromptPart("repair_context", repair_context, TOOL_IO),
        PromptPart(
            "repair_instruction",
            "Use the traceback, stdout/stderr, and scorer message as authoritative. "
            "Fix the concrete runtime or scoring failure, then audit the entire script against the task output schema. "
            "Do not preserve a previous output-writing pattern if it conflicts with the requested file format or keys. "
            "For JSON summary tasks, write a JSON object with the requested keys, not a list of row records. "
            "Ensure every module used by the corrected script is imported. "
            "If the corrected script uses json.dump or json.dumps, include import json at the top. "
            "Return only the full corrected Python code for the authoritative task below.",
            FIXED_PROMPT,
        ),
        PromptPart(
            "authoritative_task",
            "AUTHORITATIVE TASK SPECIFICATION. Follow this exactly, even if previous code, execution output, "
            "planner notes, or critic notes conflict:\n"
            f"{task.prompt}",
            FIXED_PROMPT,
        ),
    ]


def _execute_and_score(
    code: str,
    task: ToyTask,
    run_dir: Path,
    attempt: int,
    timeout_s: int,
) -> tuple[ExecutionResult, ScoreResult]:
    exec_result = execute_python_code(code, run_dir, attempt=attempt, timeout_s=timeout_s)
    if exec_result.succeeded:
        try:
            score = task.score(run_dir)
        except Exception as exc:
            score = ScoreResult(
                False,
                0.0,
                f"scorer failed: {type(exc).__name__}: {exc}",
                {"scorer_exception": repr(exc)},
            )
    else:
        score = ScoreResult(False, 0.0, "script execution failed", {})
    write_json(
        run_dir / f"attempt_{attempt}_execution.json",
        {
            "execution": exec_result.__dict__,
            "score": score.__dict__,
        },
    )
    return exec_result, score


def _repair_context(code: str, execution: ExecutionResult, score: ScoreResult) -> str:
    return (
        "Previous Python code:\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        f"Return code: {execution.returncode}\n"
        f"Timed out: {execution.timed_out}\n"
        f"Stdout:\n{execution.stdout[-2000:]}\n\n"
        f"Stderr:\n{execution.stderr[-2000:]}\n\n"
        f"Scorer message: {score.message}\n"
        f"Scorer details: {score.details}\n"
    )


def _prepare_run_dir(run_root: Path, mode: str, task: ToyTask, repeat: int) -> Path:
    run_dir = run_root / f"{mode}_{task.task_id}_r{repeat}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_record(
    *,
    run_id: str,
    task: ToyTask,
    mode: str,
    model_id: str,
    repeat: int,
    score: ScoreResult,
    wall_latency_s: float,
    exec_results: list[ExecutionResult],
    calls: list[ModelCallRecord],
    ledger: TokenLedger,
    attempts: int,
    run_dir: Path,
    latent_repair_strategy: str = "",
    settings: AgentSettings | None = None,
) -> RunRecord:
    record = RunRecord(
        run_id=run_id,
        task_id=task.task_id,
        mode=mode,
        model_id=model_id,
        repeat=repeat,
        passed=bool(score.passed),
        score=float(score.score),
        message=score.message,
        wall_latency_s=wall_latency_s,
        model_latency_ms=sum(call.model_latency_ms for call in calls),
        code_exec_latency_s=sum(result.wall_latency_s for result in exec_results),
        forward_passes=sum(call.forward_passes for call in calls),
        generated_tokens=sum(call.output_tokens for call in calls),
        model_input_tokens=sum(call.input_tokens for call in calls),
        coordination_tokens=ledger.coordination_tokens,
        tool_io_tokens=ledger.tool_io_tokens,
        fixed_prompt_tokens=ledger.fixed_prompt_tokens,
        coordination_fraction=ledger.coordination_fraction,
        peak_vram_mb=max([call.peak_vram_mb for call in calls] or [0.0]),
        attempts=attempts,
        run_dir=str(run_dir),
        latent_steps=sum(call.latent_steps for call in calls),
        latent_repair_strategy=latent_repair_strategy,
        task_family=str(getattr(task, "task_family", "")),
        horizon_level=str(getattr(task, "horizon_level", "")),
        horizon_stages=int(getattr(task, "horizon_stages", 0) or 0),
        coordination_rounds=(
            int(settings.coordination_rounds)
            if settings and settings.coordination_rounds
            else int(getattr(task, "horizon_stages", 0) or 0)
        ),
        generation_seed=int(settings.generation_seed) if settings and settings.generation_seed is not None else 0,
        schedule_seed=int(settings.schedule_seed) if settings else 0,
        temperature=float(settings.temperature) if settings else 0.0,
        top_p=float(settings.top_p) if settings else 1.0,
        max_new_tokens_code=int(settings.max_new_tokens_code) if settings else 0,
        reference_output_tokens=(
            int(settings.reference_output_tokens)
            if settings
            else int(getattr(task, "reference_output_tokens", 0) or 0)
        ),
        reference_budget_ratio=(
            float(settings.reference_budget_ratio)
            if settings
            else float(getattr(task, "reference_budget_ratio", 0.0) or 0.0)
        ),
        requested_latent_steps=int(settings.requested_latent_steps) if settings else 0,
        effective_latent_steps=int(settings.effective_latent_steps) if settings else 0,
        oom_fallback_used=bool(settings.oom_fallback_used) if settings else False,
        model_calls=calls,
        token_events=ledger.events,
        details={
            "score_details": score.details,
            "latent_repair_strategy": latent_repair_strategy,
        },
    )
    write_json(run_dir / "run_record.json", record.to_dict())
    return record
