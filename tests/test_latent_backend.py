from __future__ import annotations

import torch

from latent_agent.latent_backend import LatentBackend, past_length
from latent_agent.metrics import ModelCallRecord, RunRecord, write_json


class FakeCache:
    def __init__(self, length: int):
        self.length = length

    def get_seq_length(self):
        return self.length


def test_past_length_supports_tuple_cache():
    key = torch.zeros(1, 2, 7, 3)
    value = torch.zeros(1, 2, 7, 3)
    assert past_length(((key, value),)) == 7


def test_past_length_supports_dynamic_cache_like_object():
    assert past_length(FakeCache(11)) == 11


def test_run_record_serializes_latent_steps(tmp_path):
    record = RunRecord(
        run_id="r",
        task_id="t",
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
        run_dir="/tmp/r",
        latent_steps=4,
        model_calls=[ModelCallRecord("latent", 1, 0, 2, 3.0, 0.1, 9.0, latent_steps=4)],
    )
    path = tmp_path / "record.json"
    write_json(path, record.to_dict())
    text = path.read_text(encoding="utf-8")
    assert '"latent_steps": 4' in text


def test_latent_backend_raw_continuation_skips_chat_template():
    calls = []

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None, **kwargs):
            calls.append({"text": text, "kwargs": kwargs})
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    class FakeBackend:
        tokenizer = FakeTokenizer()

        @staticmethod
        def _format_prompt(prompt: str) -> str:
            return f"CHAT::{prompt}"

    latent = LatentBackend.__new__(LatentBackend)
    latent.backend = FakeBackend()
    latent.tokenizer = latent.backend.tokenizer
    latent.device = "cpu"

    raw = latent._encode_prompt("Stage 1: clean", raw_continuation=True)
    normal = latent._encode_prompt("Stage 1: clean", raw_continuation=False)

    assert raw["input_ids"].shape == (1, 3)
    assert calls[0]["text"] == "Stage 1: clean"
    assert calls[0]["kwargs"]["add_special_tokens"] is False
    assert calls[1]["text"] == "CHAT::Stage 1: clean"
