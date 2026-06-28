from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.agents import _repair_context, _system_code_part  # noqa: E402
from latent_agent.executor import execute_python_code  # noqa: E402
from latent_agent.latent_backend import LatentBackend, past_length  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402
from latent_agent.tasks import TASKS, ScoreResult  # noqa: E402
from latent_agent.token_split import FIXED_PROMPT, TOOL_IO, PromptPart, extract_python_code, render_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate: latent state survives execute->observe->continue.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--task", default="grouped_sales", choices=sorted(TASKS))
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--fallback-latent-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens-code", type=int, default=320)
    parser.add_argument("--max-new-tokens-continuation", type=int, default=120)
    parser.add_argument("--execution-timeout-seconds", type=int, default=30)
    parser.add_argument("--out", default=str(project_path("exports", "latent_tool_roundtrip.json")))
    return parser.parse_args()


def _append(latent: LatentBackend, parts: list[PromptPart], *, latent_steps: int, past):
    return latent.append_latent(render_prompt(parts), latent_steps=latent_steps, past_key_values=past)


def _decode(latent: LatentBackend, parts: list[PromptPart], *, max_new_tokens: int, past):
    return latent.decode_from_past(render_prompt(parts), max_new_tokens=max_new_tokens, past_key_values=past)


def run_gate(args: argparse.Namespace, latent_steps: int) -> dict[str, object]:
    task = TASKS[args.task]
    backend = ModelBackend(args.model, quantization=args.quantization)
    latent = LatentBackend(backend)
    batch_id = dt.datetime.now(dt.timezone.utc).strftime("phase2_gate_%Y%m%dT%H%M%SZ")
    run_dir = runtime_path("runs", batch_id, f"latent_tool_roundtrip_{task.task_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    task.setup(run_dir)

    start = time.perf_counter()
    past = None
    past_lengths: dict[str, int] = {}

    planner = _append(
        latent,
        [
            PromptPart("planner_system", "Think internally about a short data-analysis plan. Do not emit text.", FIXED_PROMPT),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        latent_steps=latent_steps,
        past=past,
    )
    past = planner.past_key_values
    past_lengths["after_planner"] = past_length(past)

    coder = _decode(
        latent,
        [
            _system_code_part("You are the coder using latent planner memory."),
            PromptPart("task", task.prompt, FIXED_PROMPT),
        ],
        max_new_tokens=args.max_new_tokens_code,
        past=past,
    )
    past = coder.past_key_values
    code = extract_python_code(coder.text)
    (run_dir / "roundtrip_code.py").write_text(code, encoding="utf-8")
    execution = execute_python_code(code, run_dir, attempt=1, timeout_s=args.execution_timeout_seconds)
    score = task.score(run_dir) if execution.succeeded else ScoreResult(False, 0.0, "script execution failed", {})
    past_lengths["after_coder_decode"] = past_length(past)

    observation = _append(
        latent,
        [
            PromptPart(
                "observation_system",
                "Ingest this tool result into latent memory, then preserve whether the task passed. Do not emit text.",
                FIXED_PROMPT,
            ),
            PromptPart("tool_result", _repair_context(code, execution, score), TOOL_IO),
        ],
        latent_steps=1,
        past=past,
    )
    past = observation.past_key_values
    past_lengths["after_tool_observation"] = past_length(past)

    continuation = _decode(
        latent,
        [
            PromptPart(
                "continuation_prompt",
                "Continue from latent memory. If the executed code passed, write 'PASS' and name the output file. "
                "If it failed, return only corrected Python code.",
                FIXED_PROMPT,
            )
        ],
        max_new_tokens=args.max_new_tokens_continuation,
        past=past,
    )
    continuation_text = continuation.text.strip()
    (run_dir / "latent_continuation.txt").write_text(continuation_text, encoding="utf-8")
    past_lengths["after_continuation"] = past_length(continuation.past_key_values)

    final_passed = bool(score.passed)
    repair_score = None
    if not final_passed:
        repair_code = extract_python_code(continuation_text)
        (run_dir / "roundtrip_repair.py").write_text(repair_code, encoding="utf-8")
        repair_execution = execute_python_code(repair_code, run_dir, attempt=2, timeout_s=args.execution_timeout_seconds)
        repair_score = task.score(run_dir) if repair_execution.succeeded else ScoreResult(False, 0.0, "repair execution failed", {})
        final_passed = bool(repair_score.passed)

    coherence_markers = ["pass", "csv", "json", "output", task.task_id.split("_")[0], "python", "import"]
    coherent = bool(continuation_text) and any(marker in continuation_text.lower() for marker in coherence_markers)
    ok = bool(final_passed and coherent and past_lengths["after_tool_observation"] > past_lengths["after_coder_decode"])

    result = {
        "ok": ok,
        "model": args.model,
        "quantization": args.quantization,
        "task_id": task.task_id,
        "latent_steps": latent_steps,
        "run_dir": str(run_dir),
        "initial_passed": bool(score.passed),
        "initial_score": float(score.score),
        "repair_score": None if repair_score is None else float(repair_score.score),
        "final_passed": final_passed,
        "continuation_coherent": coherent,
        "continuation_preview": continuation_text[:500],
        "past_lengths": past_lengths,
        "elapsed_s": time.perf_counter() - start,
        "peak_vram_mb": max(
            planner.metrics.peak_vram_mb,
            coder.metrics.peak_vram_mb,
            observation.metrics.peak_vram_mb,
            continuation.metrics.peak_vram_mb,
        ),
    }
    return result


def main() -> int:
    args = parse_args()
    configure_runtime(create=True)
    result: dict[str, object]
    try:
        result = run_gate(args, args.latent_steps)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or args.latent_steps <= args.fallback_latent_steps:
            raise
        import torch

        torch.cuda.empty_cache()
        result = run_gate(args, args.fallback_latent_steps)
        result["oom_fallback_used"] = True
    else:
        result["oom_fallback_used"] = False

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
