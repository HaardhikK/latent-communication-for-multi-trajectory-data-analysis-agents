from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelCallRecord:
    call_name: str
    input_tokens: int
    output_tokens: int
    forward_passes: int
    model_latency_ms: float
    wall_latency_s: float
    peak_vram_mb: float
    latent_steps: int = 0


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    mode: str
    model_id: str
    repeat: int
    passed: bool
    score: float
    message: str
    wall_latency_s: float
    model_latency_ms: float
    code_exec_latency_s: float
    forward_passes: int
    generated_tokens: int
    model_input_tokens: int
    coordination_tokens: int
    tool_io_tokens: int
    fixed_prompt_tokens: int
    coordination_fraction: float
    peak_vram_mb: float
    attempts: int
    run_dir: str
    latent_steps: int = 0
    latent_repair_strategy: str = ""
    task_family: str = ""
    horizon_level: str = ""
    horizon_stages: int = 0
    coordination_rounds: int = 0
    generation_seed: int = 0
    schedule_seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens_code: int = 0
    reference_output_tokens: int = 0
    reference_budget_ratio: float = 0.0
    requested_latent_steps: int = 0
    effective_latent_steps: int = 0
    oom_fallback_used: bool = False
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    token_events: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = _jsonable(asdict(self))
        data["model_calls"] = [asdict(call) for call in self.model_calls]
        return data

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "mode": self.mode,
            "model_id": self.model_id,
            "repeat": self.repeat,
            "passed": self.passed,
            "score": self.score,
            "message": self.message,
            "wall_latency_s": self.wall_latency_s,
            "model_latency_ms": self.model_latency_ms,
            "code_exec_latency_s": self.code_exec_latency_s,
            "forward_passes": self.forward_passes,
            "generated_tokens": self.generated_tokens,
            "model_input_tokens": self.model_input_tokens,
            "coordination_tokens": self.coordination_tokens,
            "tool_io_tokens": self.tool_io_tokens,
            "fixed_prompt_tokens": self.fixed_prompt_tokens,
            "coordination_fraction": self.coordination_fraction,
            "peak_vram_mb": self.peak_vram_mb,
            "attempts": self.attempts,
            "latent_steps": self.latent_steps,
            "latent_repair_strategy": self.latent_repair_strategy,
            "task_family": self.task_family,
            "horizon_level": self.horizon_level,
            "horizon_stages": self.horizon_stages,
            "coordination_rounds": self.coordination_rounds,
            "generation_seed": self.generation_seed,
            "schedule_seed": self.schedule_seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens_code": self.max_new_tokens_code,
            "reference_output_tokens": self.reference_output_tokens,
            "reference_budget_ratio": self.reference_budget_ratio,
            "requested_latent_steps": self.requested_latent_steps,
            "effective_latent_steps": self.effective_latent_steps,
            "oom_fallback_used": self.oom_fallback_used,
            "run_dir": self.run_dir,
        }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_flat_dict() for record in records]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_token_split(records: list[RunRecord], gate: float = 0.20) -> str:
    coord = sum(record.coordination_tokens for record in records)
    tool = sum(record.tool_io_tokens for record in records)
    fixed = sum(record.fixed_prompt_tokens for record in records)
    denom = coord + tool
    fraction = coord / denom if denom else 0.0
    decision = "STOP before Phase 2" if fraction < gate else "Phase 2 is plausible"

    lines = [
        "# Phase 1 Token Split Report",
        "",
        f"- Runs summarized: {len(records)}",
        f"- Coordination tokens: {coord}",
        f"- Tool/code/execution I/O tokens: {tool}",
        f"- Fixed prompt tokens: {fixed}",
        f"- Coordination fraction: {fraction:.3f}",
        f"- Gate: {gate:.2f}",
        f"- Decision: **{decision}**",
        "",
        "| Mode | Runs | Passes | Coordination tokens | Tool I/O tokens | Coordination fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    modes = sorted({record.mode for record in records})
    for mode in modes:
        subset = [record for record in records if record.mode == mode]
        mode_coord = sum(record.coordination_tokens for record in subset)
        mode_tool = sum(record.tool_io_tokens for record in subset)
        mode_fraction = mode_coord / (mode_coord + mode_tool) if (mode_coord + mode_tool) else 0.0
        passes = sum(record.passed for record in subset)
        lines.append(f"| {mode} | {len(subset)} | {passes} | {mode_coord} | {mode_tool} | {mode_fraction:.3f} |")
    lines.append("")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
