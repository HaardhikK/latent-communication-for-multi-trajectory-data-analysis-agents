from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_agent.metrics import RunRecord


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase3  # noqa: E402
import run_phase4  # noqa: E402
import write_phase4a_findings  # noqa: E402


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


def test_phase4_c1_phase3_exact_has_no_copied_stage_plan():
    task = _fake_task()
    assert run_phase4.build_stage_append_plan(task, "C1_phase3_exact", latent_steps=4) == []


def test_phase4_c1_phase3_exact_delegates_to_phase3(monkeypatch, tmp_path):
    calls = []

    def fake_latentmas(backend, task, run_root, *, repeat, settings, debug_decode_latent=False, diagnostics=None):
        calls.append((backend, task, run_root, repeat, settings, debug_decode_latent))
        run_dir = tmp_path / "exact"
        run_dir.mkdir()
        (run_dir / "attempt_1.py").write_text("print('ok')\n", encoding="utf-8")
        if diagnostics is not None:
            diagnostics.first_code = "print('ok')\n"
            diagnostics.first_attempt_passed = True
            diagnostics.cache_len_at_decode = 42
            diagnostics.latent_append_audit.append({"call_name": "latent_planner_stage_1"})
        return _record(run_dir)

    monkeypatch.setattr(run_phase3, "run_phase3_latentmas", fake_latentmas)
    record = run_phase4.run_phase4_phase3_exact(
        backend=object(),
        task=_fake_task(),
        run_root=tmp_path,
        repeat=1,
        settings=SimpleNamespace(),
    )
    assert calls
    assert record.c_variant == "C1_phase3_exact"
    assert record.cache_len_at_decode == 42
    assert record.first_attempt_passed


def test_phase4_c5_uses_c2_stage_latent_steps():
    task = _fake_task()
    specs = run_phase4.build_stage_append_plan(task, "C5_anchor", latent_steps=4)
    stage_specs = [spec for spec in specs if spec.stage_index]
    assert stage_specs
    assert all(spec.raw_continuation for spec in stage_specs)
    assert all(spec.latent_steps == 4 for spec in stage_specs)


def test_phase4_c3_uses_zero_stage_latent_steps():
    task = _fake_task()
    specs = run_phase4.build_stage_append_plan(task, "C3_no_latent", latent_steps=4)
    stage_specs = [spec for spec in specs if spec.stage_index]
    assert stage_specs
    assert all(spec.latent_steps == 0 for spec in stage_specs)


def test_phase4_schedule_key_includes_c_variant_and_experiment_part():
    task = _fake_task()
    c1 = run_phase4.ScheduleItem(1, task, "C", "C1_phase3_exact", "session2")
    c2 = run_phase4.ScheduleItem(1, task, "C", "C2_dedup", "session2")
    pilot = run_phase4.ScheduleItem(1, task, "C", "C2_dedup", "pilot")
    assert run_phase4.schedule_key(c1) != run_phase4.schedule_key(c2)
    assert run_phase4.schedule_key(c2) != run_phase4.schedule_key(pilot)
    assert run_phase4.schedule_key(c1)[3] == "C1_phase3_exact"
    assert run_phase4.schedule_key(c1)[4] == "session2"


def test_phase4_baselines_are_scheduled_before_c_variants():
    task = _fake_task()
    schedule = run_phase4.build_schedule(
        [task],
        ["C1_phase3_exact"],
        repeat=1,
        schedule_seed=1701,
        include_baselines=True,
        baseline_modes=["A", "B"],
        experiment_part="session2",
    )
    assert [item.mode for item in schedule[:2]] == ["A", "B"] or [item.mode for item in schedule[:2]] == ["B", "A"]
    assert schedule[2].mode == "C"


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
    assert enriched.experiment_part
    assert enriched.source_commit
    assert enriched.script_hash
    assert enriched.failure_type == "passed"


def test_phase4_failure_analysis_detects_repeated_runtime_exception(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for attempt in (1, 2):
        (run_dir / f"attempt_{attempt}.py").write_text("raise ValueError('bad')\n", encoding="utf-8")
        (run_dir / f"attempt_{attempt}_execution.json").write_text(
            '{"execution":{"returncode":1,"stderr":"Traceback\\nValueError: bad","timed_out":false},"score":{"passed":false}}',
            encoding="utf-8",
        )
    record = _record(run_dir)
    record.passed = False
    analysis = run_phase4.analyze_failure(run_dir, record, quality_empty=False, quality_ast_ok=True)
    assert analysis["failure_type"] == "runtime_bug"
    assert analysis["repeated_exception"]
    assert analysis["repair_similarity"] == 1.0


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
    assert record.experiment_part == ""
    assert record.failure_type == ""


def test_phase4_fisher_exact_known_tables():
    assert run_phase4.fisher_exact_two_sided(11, 15, 3, 15) == pytest.approx(0.00922057, abs=1e-6)
    assert run_phase4.fisher_exact_two_sided(11, 15, 6, 15) == pytest.approx(0.13941976, abs=1e-6)


def test_phase4_summary_prints_preregistered_pair_p_values(tmp_path):
    records = []
    for mode, variant, passes in [
        ("B_textmas", "", 12),
        ("C_latentmas", "C1_phase3_exact", 3),
        ("C_latentmas", "C2_dedup", 11),
        ("C_latentmas", "C3_no_latent", 6),
        ("C_latentmas", "C5_anchor", 8),
    ]:
        for i in range(15):
            record = _record(tmp_path / f"{mode}_{variant}_{i}")
            record.mode = mode
            record.c_variant = variant
            record.repeat = i + 1
            record.passed = i < passes
            record.first_attempt_passed = i < passes
            records.append(record)
    summary = run_phase4.summarize_phase4(records)
    assert "Cache pollution delta C2-C1 final=0.533; Fisher p=0.009" in summary
    assert "Latent-step readout C2-C3 final=0.333; Fisher p=0.139" in summary
    assert "Anchor effect primary delta C5-C2 final=-0.200; Fisher p=0.450" in summary
    assert "Text-baseline comparison delta C2-vs-B final=-0.067; Fisher p=1.000" in summary


def test_phase4_anchor_classifier_flags_polluted_anchor_text():
    assert write_phase4a_findings.classify_anchor("Stage 3/7: Join metadata.Stage 4/7: Compute alerts") == "wrong"
    assert write_phase4a_findings.classify_anchor("Follow the authoritative task specification for this stage.") == "vague"
    assert write_phase4a_findings.classify_anchor("df.groupby('site').agg(alert_count=('alert','sum'))") == "code-like"
    assert write_phase4a_findings.classify_anchor("Aggregate daily site alerts and save the requested summary.") == "faithful"


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
