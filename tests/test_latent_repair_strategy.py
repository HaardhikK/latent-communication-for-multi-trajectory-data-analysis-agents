from __future__ import annotations

from types import SimpleNamespace

from latent_agent import agents
from latent_agent.agents import AgentSettings
from latent_agent.executor import ExecutionResult
from latent_agent.tasks import ScoreResult
from latent_agent.token_split import TokenLedger


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeBackend:
    model_id = "fake"

    def __init__(self):
        self.tokenizer = ToyTokenizer()
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int, **kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(
            text="print('fixed')",
            metrics=SimpleNamespace(
                input_tokens=len(prompt.split()),
                output_tokens=2,
                forward_passes=2,
                model_latency_ms=3.0,
                wall_latency_s=0.1,
                peak_vram_mb=4.0,
            ),
        )


class FakeTask:
    prompt = "Fix input.csv and write summary.json."


def _execution_failure() -> ExecutionResult:
    return ExecutionResult(
        code_path="/tmp/attempt_1.py",
        returncode=1,
        stdout="",
        stderr="Traceback: AttributeError: module pandas has no attribute to_numeric_dtype",
        wall_latency_s=0.1,
        timed_out=False,
    )


def test_text_reset_repair_uses_text_context_as_tool_io():
    backend = FakeBackend()
    ledger = TokenLedger()
    calls = []
    settings = AgentSettings(max_new_tokens_code=10, latent_repair_strategy="text_reset")

    repair_code, returned_past = agents._repair_latent_failure(
        "text_reset",
        object(),
        backend,
        ledger,
        calls,
        FakeTask(),
        "bad code",
        _execution_failure(),
        ScoreResult(False, 0.0, "script execution failed", {}),
        "OLD_PAST",
        settings,
    )

    assert repair_code == "print('fixed')"
    assert returned_past == "OLD_PAST"
    assert "to_numeric_dtype" in backend.prompts[0]
    assert ledger.tool_io_tokens > 0
    assert ledger.coordination_tokens == 0
    assert calls[0].call_name == "latent_text_reset_repair"


def test_latent_repair_strategy_keeps_current_latent_path(monkeypatch):
    seen = {}

    def fake_latent_decode(latent, backend, ledger, calls, call_name, parts, *, output_category, max_new_tokens, past_key_values, settings=None):
        seen["call_name"] = call_name
        seen["labels"] = [part.label for part in parts]
        seen["past"] = past_key_values
        return "latent repair", "NEXT_PAST"

    monkeypatch.setattr(agents, "_latent_decode", fake_latent_decode)

    repair_code, returned_past = agents._repair_latent_failure(
        "latent",
        object(),
        FakeBackend(),
        TokenLedger(),
        [],
        FakeTask(),
        "bad code",
        _execution_failure(),
        ScoreResult(False, 0.0, "script execution failed", {}),
        "OLD_PAST",
        AgentSettings(latent_repair_strategy="latent"),
    )

    assert repair_code == "latent repair"
    assert returned_past == "NEXT_PAST"
    assert seen == {
        "call_name": "latent_coder_repair_decode",
        "labels": ["code_system", "task"],
        "past": "OLD_PAST",
    }


def test_text_keep_latent_injects_repair_context_and_keeps_past(monkeypatch):
    seen = {}

    def fake_latent_decode(latent, backend, ledger, calls, call_name, parts, *, output_category, max_new_tokens, past_key_values, settings=None):
        seen["call_name"] = call_name
        seen["labels"] = [part.label for part in parts]
        seen["past"] = past_key_values
        return "grounded repair", "NEXT_PAST"

    monkeypatch.setattr(agents, "_latent_decode", fake_latent_decode)

    agents._repair_latent_failure(
        "text_keep_latent",
        object(),
        FakeBackend(),
        TokenLedger(),
        [],
        FakeTask(),
        "bad code",
        _execution_failure(),
        ScoreResult(False, 0.0, "script execution failed", {}),
        "OLD_PAST",
        AgentSettings(latent_repair_strategy="text_keep_latent"),
    )

    assert seen["call_name"] == "latent_text_grounded_repair_decode"
    assert "repair_context" in seen["labels"]
    assert seen["past"] == "OLD_PAST"


def test_latent_reset_rebuilds_repair_memory_without_old_past(monkeypatch):
    seen = {}

    def fake_latent_append(latent, backend, ledger, calls, call_name, parts, *, latent_steps, past_key_values):
        seen["append_call_name"] = call_name
        seen["append_labels"] = [part.label for part in parts]
        seen["append_past"] = past_key_values
        return "REPAIR_PAST"

    def fake_latent_decode(latent, backend, ledger, calls, call_name, parts, *, output_category, max_new_tokens, past_key_values, settings=None):
        seen["decode_call_name"] = call_name
        seen["decode_past"] = past_key_values
        return "reset repair", "NEXT_PAST"

    monkeypatch.setattr(agents, "_latent_append", fake_latent_append)
    monkeypatch.setattr(agents, "_latent_decode", fake_latent_decode)

    agents._repair_latent_failure(
        "latent_reset",
        object(),
        FakeBackend(),
        TokenLedger(),
        [],
        FakeTask(),
        "bad code",
        _execution_failure(),
        ScoreResult(False, 0.0, "script execution failed", {}),
        "OLD_PAST",
        AgentSettings(latent_repair_strategy="latent_reset"),
    )

    assert seen["append_call_name"] == "latent_reset_repair_observation"
    assert "repair_context" in seen["append_labels"]
    assert seen["append_past"] is None
    assert seen["decode_call_name"] == "latent_reset_repair_decode"
    assert seen["decode_past"] == "REPAIR_PAST"
