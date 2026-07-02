from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from latent_agent.metrics import RunRecord


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase3  # noqa: E402
import run_phase4  # noqa: E402


def test_phase4_c2_stage_appends_do_not_repeat_full_task_prompt():
    task = _fake_task()
    specs = run_phase4.build_stage_append_plan(task, "C2_dedup", latent_steps=4)
    assert specs[0].call_name == "latent_context_once"
    stage_specs = [spec for spec in specs if spec.stage_index]
    assert len(stage_specs) == 3
    assert all(spec.raw_continuation for spec in stage_specs)
    assert all(task.prompt not in spec.rendered_text for spec in stage_specs)
    assert all("Stage " in spec.rendered_text for spec in stage_specs)


def test_phase4_c1_keeps_current_duplicate_prompt_behavior():
    task = _fake_task()
    specs = run_phase4.build_stage_append_plan(task, "C1_current", latent_steps=4)
    stage_specs = [spec for spec in specs if spec.stage_index]
    assert len(stage_specs) == 3
    assert all(not spec.raw_continuation for spec in stage_specs)
    assert all(task.prompt in spec.rendered_text for spec in stage_specs)


def test_phase4_c3_uses_zero_stage_latent_steps():
    task = _fake_task()
    specs = run_phase4.build_stage_append_plan(task, "C3_no_latent", latent_steps=4)
    stage_specs = [spec for spec in specs if spec.stage_index]
    assert stage_specs
    assert all(spec.latent_steps == 0 for spec in stage_specs)


def test_phase4_schedule_key_includes_c_variant():
    task = _fake_task()
    c1 = run_phase4.ScheduleItem(1, task, "C", "C1_current")
    c2 = run_phase4.ScheduleItem(1, task, "C", "C2_dedup")
    assert run_phase4.schedule_key(c1) != run_phase4.schedule_key(c2)
    assert run_phase4.schedule_key(c1)[3] == "C1_current"


def test_phase4_enrich_forensics_records_cache_and_writes_audits(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _record(run_dir)
    enriched = run_phase4.enrich_forensics(
        record,
        c_variant="C2_dedup",
        first_code="import json\nprint('ok')\n",
        first_attempt_passed=True,
        cache_len_at_decode=123,
        stage_append_audit=[{"call_name": "latent_stage_line_1", "text": "Stage 1/3: clean"}],
        anchor_texts=[],
    )
    assert enriched.c_variant == "C2_dedup"
    assert enriched.first_attempt_passed
    assert enriched.first_attempt_ast_ok
    assert not enriched.first_attempt_empty
    assert enriched.cache_len_at_decode == 123
    assert (run_dir / "stage_append_audit.json").exists()
    assert (run_dir / "first_attempt_code_quality.json").exists()
    assert (run_dir / "run_record.json").exists()


def test_phase3_old_flat_rows_load_with_phase4_defaults():
    row = {
        "run_id": "r",
        "task_id": "orders_kpi_long",
        "mode": "C_latentmas",
        "model_id": "m",
        "repeat": "1",
        "passed": "True",
        "score": "1.0",
        "message": "ok",
        "wall_latency_s": "1.0",
        "model_latency_ms": "2.0",
        "code_exec_latency_s": "0.1",
        "forward_passes": "3",
        "generated_tokens": "4",
        "model_input_tokens": "5",
        "coordination_tokens": "0",
        "tool_io_tokens": "6",
        "fixed_prompt_tokens": "7",
        "coordination_fraction": "0.0",
        "peak_vram_mb": "8.0",
        "attempts": "1",
        "run_dir": "/tmp/r",
    }
    record = run_phase3._run_record_from_flat_row(row)
    assert record.c_variant == ""
    assert not record.first_attempt_passed
    assert record.cache_len_at_decode == 0


def _fake_task():
    return SimpleNamespace(
        task_id="orders_kpi_long",
        prompt="FULL TASK SPEC: write orders_long_report.json with exact keys.",
        stage_specs=["load files", "clean rows", "save report"],
        horizon_stages=3,
        task_family="orders_kpi",
        horizon_level="long",
    )


def _record(run_dir: Path) -> RunRecord:
    return RunRecord(
        run_id="r",
        task_id="orders_kpi_long",
        mode="C_latentmas",
        model_id="m",
        repeat=1,
        passed=True,
        score=1.0,
        message="ok",
        wall_latency_s=1.0,
        model_latency_ms=2.0,
        code_exec_latency_s=0.1,
        forward_passes=3,
        generated_tokens=4,
        model_input_tokens=5,
        coordination_tokens=0,
        tool_io_tokens=6,
        fixed_prompt_tokens=7,
        coordination_fraction=0.0,
        peak_vram_mb=8.0,
        attempts=1,
        run_dir=str(run_dir),
    )
