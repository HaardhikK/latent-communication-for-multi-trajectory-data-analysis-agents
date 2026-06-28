from __future__ import annotations

import sys
from pathlib import Path

import pytest

from latent_agent.executor import execute_python_code
from latent_agent.agents import _repair_context, _text_reset_repair_parts
from latent_agent.executor import ExecutionResult
from latent_agent.horizon_tasks import TASKS
from latent_agent.metrics import RunRecord
from latent_agent.tasks import ScoreResult
from latent_agent.token_split import TOOL_IO, render_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phase3  # noqa: E402


def test_generation_seed_assignment_is_zero_based_repeat_policy():
    assert run_phase3.generation_seed_for_repeat(1) == 17
    assert run_phase3.generation_seed_for_repeat(5) == 21
    assert run_phase3.generation_seed_for_repeat(1, base_seed=99) == 99
    with pytest.raises(ValueError):
        run_phase3.generation_seed_for_repeat(0)


def test_schedule_randomization_does_not_change_seed_assignment():
    tasks = [TASKS["orders_kpi_short"], TASKS["sensor_quality_short"]]
    schedule_a = run_phase3.build_schedule(tasks, ["A", "B", "C"], repeat=2, schedule_seed=1)
    schedule_b = run_phase3.build_schedule(tasks, ["A", "B", "C"], repeat=2, schedule_seed=99)
    seeds_a = sorted((repeat, task.task_id, mode, run_phase3.generation_seed_for_repeat(repeat)) for repeat, task, mode in schedule_a)
    seeds_b = sorted((repeat, task.task_id, mode, run_phase3.generation_seed_for_repeat(repeat)) for repeat, task, mode in schedule_b)
    assert seeds_a == seeds_b
    assert schedule_a != schedule_b


def test_horizon_budget_scaling_enforces_reference_fit():
    budget, ratio = run_phase3.resolve_code_budget(350, "short")
    assert budget == 512
    assert ratio < 0.70

    budget, ratio = run_phase3.resolve_code_budget(600, "medium")
    assert budget == 896
    assert ratio < 0.70

    with pytest.raises(ValueError):
        run_phase3.resolve_code_budget(600, "short")


def test_phase3_reference_scripts_pass_scorers(tmp_path):
    for task in TASKS.values():
        run_dir = tmp_path / task.task_id
        run_dir.mkdir()
        task.setup(run_dir)
        execution = execute_python_code(task.reference_script, run_dir, attempt=1, timeout_s=30)
        score = task.score(run_dir)
        assert execution.succeeded, task.task_id
        assert score.passed, task.task_id


def test_phase3_prompts_include_schema_and_preserve_join_keys():
    orders = TASKS["orders_kpi_short"].prompt
    assert "orders.csv has order_id, customer_id" in orders
    assert "customer_id is a numeric ID" in orders
    assert "Preserve join/id/status columns" in orders
    assert "do not replace the whole orders table with select_dtypes()" in orders
    for snippet in (
        "orders_short_report.json",
        "rows_clean = cleaned order row count",
        "total_net_revenue = sum of net_revenue",
        "top_region = region with the largest summed net_revenue",
        "net_revenue = units * unit_price * (1 - discount_rate)",
    ):
        assert snippet in orders

    sensor = TASKS["sensor_quality_short"].prompt
    assert "readings.csv has timestamp, sensor_id" in sensor
    assert "Preserve timestamp, sensor_id" in sensor
    for snippet in (
        "sensor_short_report.json",
        "exactly one JSON object",
        "not a list and not grouped by sensor_id",
        "rows_clean = count of cleaned reading rows where status == ok and raw_temp is present",
        "mean_adjusted_temp = mean adjusted_temp over those cleaned rows",
        "alert_count = number of cleaned rows where alert is true",
        "adjusted_temp = raw_temp + temp_offset",
        "adjusted_pressure = raw_pressure * pressure_scale",
        "alert = (adjusted_temp > 76) | (adjusted_pressure > 105)",
    ):
        assert snippet in sensor

    campaign = TASKS["campaign_roi_short"].prompt
    assert "campaigns.csv has campaign_id, channel_id" in campaign
    assert "Preserve campaign_id and channel_id" in campaign
    for snippet in (
        "campaign_short_report.json",
        "rows_clean = cleaned campaign row count",
        "mean_roi = mean ROI over cleaned rows",
        "top_campaign = campaign_id with the highest ROI",
        "ROI = (revenue - spend) / spend",
    ):
        assert snippet in campaign


def test_orders_short_fixture_uses_numeric_customer_ids(tmp_path):
    task = TASKS["orders_kpi_short"]
    task.setup(tmp_path)
    orders_text = (tmp_path / "orders.csv").read_text(encoding="utf-8")
    customers_text = (tmp_path / "customers.csv").read_text(encoding="utf-8")
    assert "C1" not in orders_text
    assert "C1" not in customers_text
    assert "customer_id" in orders_text


def test_run_record_flat_dict_includes_phase3_fields():
    record = RunRecord(
        run_id="r1",
        task_id="orders_kpi_long",
        mode="C_latentmas",
        model_id="fake",
        repeat=5,
        passed=True,
        score=1.0,
        message="ok",
        wall_latency_s=1.0,
        model_latency_ms=10.0,
        code_exec_latency_s=0.1,
        forward_passes=7,
        generated_tokens=20,
        model_input_tokens=30,
        coordination_tokens=0,
        tool_io_tokens=20,
        fixed_prompt_tokens=30,
        coordination_fraction=0.0,
        peak_vram_mb=4096.0,
        attempts=1,
        run_dir="/tmp/run",
        task_family="orders_kpi",
        horizon_level="long",
        horizon_stages=7,
        coordination_rounds=7,
        generation_seed=21,
        schedule_seed=1701,
        temperature=0.2,
        top_p=0.95,
        max_new_tokens_code=1152,
        reference_output_tokens=420,
        reference_budget_ratio=0.365,
        requested_latent_steps=4,
        effective_latent_steps=2,
        oom_fallback_used=True,
    )
    flat = record.to_flat_dict()
    for key in (
        "generation_seed",
        "schedule_seed",
        "temperature",
        "horizon_level",
        "max_new_tokens_code",
        "reference_output_tokens",
        "reference_budget_ratio",
        "oom_fallback_used",
        "peak_vram_mb",
    ):
        assert key in flat
    assert flat["generation_seed"] == 21
    assert flat["effective_latent_steps"] == 2


def test_synthetic_phase3_summary_has_required_tables(tmp_path):
    records = []
    for horizon in ("short", "medium", "long"):
        stages = {"short": 3, "medium": 5, "long": 7}[horizon]
        for mode in ("A_single", "B_textmas", "C_latentmas"):
            records.append(
                RunRecord(
                    run_id=f"{mode}_{horizon}",
                    task_id=f"task_{horizon}",
                    mode=mode,
                    model_id="fake",
                    repeat=1,
                    passed=True,
                    score=1.0,
                    message="ok",
                    wall_latency_s=1.0,
                    model_latency_ms=100.0 if mode == "B_textmas" else 25.0,
                    code_exec_latency_s=0.1,
                    forward_passes=100 if mode == "B_textmas" else 30,
                    generated_tokens=50,
                    model_input_tokens=50,
                    coordination_tokens=stages * 20 if mode == "B_textmas" else 0,
                    tool_io_tokens=100,
                    fixed_prompt_tokens=100,
                    coordination_fraction=0.4 if mode == "B_textmas" else 0.0,
                    peak_vram_mb=3000,
                    attempts=1,
                    run_dir="/tmp/run",
                    task_family="family",
                    horizon_level=horizon,
                    horizon_stages=stages,
                )
            )
    budget_table = {
        f"task_{h}": {
            "task_id": f"task_{h}",
            "task_family": "family",
            "horizon_level": h,
            "horizon_stages": {"short": 3, "medium": 5, "long": 7}[h],
            "reference_output_tokens": 100,
            "max_new_tokens_code": 512,
            "reference_budget_ratio": 0.2,
        }
        for h in ("short", "medium", "long")
    }
    summary = run_phase3.summarize_phase3(records, budget_table, tmp_path / "plots")
    assert "multi-stage planning-coordination horizon" in summary
    assert "one code execution at the end" in summary
    assert "B-C Gap Vs Horizon" in summary
    assert "Text Baseline Validity Gate" in summary
    assert (tmp_path / "plots" / "bc_coordination_token_gap.svg").exists()


def test_planner_sanitizer_removes_code_sections_and_fragments():
    raw = """PLAN:
- Read both files.

CODE:
import pandas as pd
campaigns = campaigns[['campaign_id', 'channel_id']]
channels =

NOTE: Follow the task spec.
"""
    cleaned = run_phase3.sanitize_planner_report(raw)
    assert "Read both files" in cleaned
    assert "import pandas" not in cleaned
    assert "campaigns[[" not in cleaned
    assert "channels =" not in cleaned
    assert "Follow the task spec" not in cleaned


def test_phase3_b_and_c_coder_prompts_end_with_same_authoritative_task():
    task = TASKS["orders_kpi_short"]
    b_parts = run_phase3._phase3_coder_parts(
        task,
        mode_label="planner -> coder -> critic",
        coordination_text="Stage 1: read files",
    )
    c_parts = run_phase3._phase3_coder_parts(task, mode_label="latent planner -> coder -> critic")
    assert b_parts[-1].label == "authoritative_task"
    assert c_parts[-1].label == "authoritative_task"
    assert b_parts[-1].text == c_parts[-1].text
    assert "orders_short_report.json" in b_parts[-1].text
    rendered_b = render_prompt(b_parts)
    assert rendered_b.rfind("AUTHORITATIVE TASK SPECIFICATION") > rendered_b.find("ADVISORY COORDINATION CONTEXT")


def test_text_reset_repair_template_counts_error_context_as_tool_io():
    task = TASKS["campaign_roi_short"]
    execution = ExecutionResult(
        code_path="/tmp/attempt.py",
        returncode=1,
        stdout="",
        stderr="KeyError: 'clicks'",
        wall_latency_s=0.1,
        timed_out=False,
    )
    score = ScoreResult(False, 0.0, "script execution failed", {})
    parts = _text_reset_repair_parts(task, _repair_context("bad code", execution, score))
    labels = [part.label for part in parts]
    assert labels == ["code_system", "repair_context", "repair_instruction", "authoritative_task"]
    assert next(part for part in parts if part.label == "repair_context").category == TOOL_IO
    assert parts[-1].text.startswith("AUTHORITATIVE TASK SPECIFICATION")
