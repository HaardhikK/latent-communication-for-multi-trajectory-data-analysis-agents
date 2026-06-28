from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.agents import AgentSettings, run_single_agent, run_textmas  # noqa: E402
from latent_agent.metrics import summarize_token_split, write_csv, write_json  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402
from latent_agent.tasks import selected_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 text baselines.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", choices=["A", "B", "both"], default="both")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: all toy tasks.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--max-new-tokens-code", type=int, default=640)
    parser.add_argument("--max-new-tokens-plan", type=int, default=160)
    parser.add_argument("--max-new-tokens-critic", type=int, default=220)
    parser.add_argument("--execution-timeout-seconds", type=int, default=30)
    parser.add_argument("--coordination-gate", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    configure_runtime(create=True)

    task_ids = [part.strip() for part in args.tasks.split(",") if part.strip()] or None
    tasks = selected_tasks(task_ids)
    modes = ["A", "B"] if args.mode == "both" else [args.mode]
    settings = AgentSettings(
        max_new_tokens_code=args.max_new_tokens_code,
        max_new_tokens_plan=args.max_new_tokens_plan,
        max_new_tokens_critic=args.max_new_tokens_critic,
        execution_timeout_seconds=args.execution_timeout_seconds,
        allow_repair=not args.no_repair,
    )

    backend = ModelBackend(args.model)
    batch_id = dt.datetime.now(dt.timezone.utc).strftime("phase1_%Y%m%dT%H%M%SZ")
    run_root = runtime_path("runs", batch_id)
    run_root.mkdir(parents=True, exist_ok=True)

    schedule = []
    for repeat in range(1, args.repeat + 1):
        for task in tasks:
            for mode in modes:
                schedule.append((repeat, task, mode))
    random.shuffle(schedule)

    records = []
    for repeat, task, mode in schedule:
        print(f"RUN repeat={repeat} task={task.task_id} mode={mode}", flush=True)
        if mode == "A":
            record = run_single_agent(backend, task, run_root, repeat=repeat, settings=settings)
        else:
            record = run_textmas(backend, task, run_root, repeat=repeat, settings=settings)
        records.append(record)
        print(
            json.dumps(
                {
                    "task_id": record.task_id,
                    "mode": record.mode,
                    "passed": bool(record.passed),
                    "score": float(record.score),
                    "coordination_fraction": float(record.coordination_fraction),
                    "run_dir": record.run_dir,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    exports_csv = project_path("exports", "phase1_metrics.csv")
    runtime_csv = runtime_path("exports", f"{batch_id}_metrics.csv")
    write_csv(exports_csv, records)
    write_csv(runtime_csv, records)

    report = summarize_token_split(records, gate=args.coordination_gate)
    report_path = project_path("reports", "phase1_token_split.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    write_json(project_path("exports", "phase1_run_manifest.json"), {"batch_id": batch_id, "run_root": str(run_root)})
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
