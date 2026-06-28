from __future__ import annotations

import sys
from pathlib import Path

from latent_agent.metrics import RunRecord


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_tier2_full_sweep as full  # noqa: E402


def test_full_sweep_expected_row_counts():
    assert full.expected_primary_rows(5) == 135
    assert full.expected_ablation_rows(5) == 30


def test_combined_labels_separate_primary_c_and_pure_latent():
    text_reset = full._combined_row(_record("C_latentmas", "medium", True, repair="text_reset"), analysis_mode="C_text_reset", experiment_part="primary")
    pure = full._combined_row(_record("C_latentmas", "medium", False, repair="latent"), analysis_mode="C_pure_latent", experiment_part="repair_ablation")
    assert text_reset["analysis_mode"] == "C_text_reset"
    assert text_reset["experiment_part"] == "primary"
    assert pure["analysis_mode"] == "C_pure_latent"
    assert pure["repair_strategy_scope"] == "pure_latent"


def test_a_vs_multi_classification_beats_matches_trails():
    pass_rows = [
        {"mode": "A_single", "horizon": "short", "runs": 10, "passes": 8, "pass_rate": 0.8},
        {"mode": "B_textmas", "horizon": "short", "runs": 10, "passes": 8, "pass_rate": 0.8},
        {"mode": "C_text_reset", "horizon": "short", "runs": 10, "passes": 9, "pass_rate": 0.9},
        {"mode": "C_pure_latent", "horizon": "short", "runs": 10, "passes": 7, "pass_rate": 0.7},
    ]
    rows = {(row["mode"], row["horizon"]): row for row in full.a_vs_multi_rows(pass_rows)}
    assert rows[("B_textmas", "short")]["relation"] == "matches"
    assert rows[("C_text_reset", "short")]["relation"] == "beats"
    assert rows[("C_pure_latent", "short")]["relation"] == "trails"


def test_full_summary_writes_required_sections_and_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "SUMMARY_PATH", tmp_path / "tier2_phase3_full_summary.md")
    monkeypatch.setattr(full, "PLOT_DIR", tmp_path / "plots")
    records = []
    for horizon in ("short", "medium", "long"):
        for mode in ("A_single", "B_textmas", "C_text_reset"):
            for repeat in range(1, 3):
                records.append(full._combined_row(_record(mode.replace("C_text_reset", "C_latentmas"), horizon, True, repeat=repeat), analysis_mode=mode, experiment_part="primary"))
    for horizon in ("medium", "long"):
        for repeat in range(1, 3):
            records.append(full._combined_row(_record("C_latentmas", horizon, repeat == 1, repeat=repeat, repair="latent"), analysis_mode="C_pure_latent", experiment_part="repair_ablation"))
    full.write_full_summary(records)
    text = (tmp_path / "tier2_phase3_full_summary.md").read_text(encoding="utf-8")
    assert "Research Readout" in text
    assert "A Vs Multi-Agent" in text
    assert "C Repair Ablation" in text
    assert (tmp_path / "plots" / "pass_rate_vs_horizon.svg").exists()


def _record(mode: str, horizon: str, passed: bool, *, repeat: int = 1, repair: str = "text_reset") -> RunRecord:
    return RunRecord(
        run_id=f"{mode}_{horizon}_r{repeat}",
        task_id=f"orders_kpi_{horizon}",
        mode=mode,
        model_id="Qwen/Qwen3-8B",
        repeat=repeat,
        passed=passed,
        score=1.0 if passed else 0.0,
        message="ok" if passed else "failed",
        wall_latency_s=1.0,
        model_latency_ms=100.0 if mode != "B_textmas" else 200.0,
        code_exec_latency_s=0.1,
        forward_passes=10 if mode != "B_textmas" else 20,
        generated_tokens=100,
        model_input_tokens=200,
        coordination_tokens=0 if mode.startswith("C_") or mode == "C_latentmas" else 50,
        tool_io_tokens=100,
        fixed_prompt_tokens=200,
        coordination_fraction=0.0 if mode.startswith("C_") or mode == "C_latentmas" else 0.333,
        peak_vram_mb=8500.0 if mode == "C_latentmas" else 6500.0,
        attempts=1,
        run_dir=f"/tmp/{mode}_{horizon}_r{repeat}",
        latent_steps=4 if mode == "C_latentmas" else 0,
        latent_repair_strategy=repair if mode == "C_latentmas" else "",
        task_family="orders_kpi",
        horizon_level=horizon,
        horizon_stages={"short": 3, "medium": 5, "long": 7}[horizon],
        coordination_rounds={"short": 3, "medium": 5, "long": 7}[horizon],
        generation_seed=16 + repeat,
        schedule_seed=1701,
        temperature=0.2,
        top_p=0.95,
        max_new_tokens_code=512,
        reference_output_tokens=100,
        reference_budget_ratio=0.2,
        requested_latent_steps=4,
        effective_latent_steps=4,
        oom_fallback_used=False,
    )
