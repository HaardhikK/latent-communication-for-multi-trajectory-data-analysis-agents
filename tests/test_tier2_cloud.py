from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

from latent_agent import cloud
from latent_agent.models import ModelBackend


def test_cloud_platform_detection_honors_kaggle_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("COLAB_GPU", raising=False)
    detected = cloud.detect_cloud_platform()
    assert detected.name == "kaggle"
    assert detected.runtime_root.as_posix() == "/kaggle/working/latent-agent-runtime"


def test_cloud_dependency_guard_reports_versions(monkeypatch):
    def fake_version(package: str) -> str:
        return cloud.CLOUD_DEPENDENCY_PINS[package]

    monkeypatch.setattr(cloud.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(cloud.importlib, "import_module", lambda package: object())
    report = cloud.assert_cloud_dependencies(strict_versions=True)
    assert report["ok"]
    assert report["packages"]["bitsandbytes"]["version"] == cloud.CLOUD_DEPENDENCY_PINS["bitsandbytes"]


def test_gpu_guard_accepts_t4_and_rejects_p100():
    assert cloud.gpu_is_supported("Tesla T4", (7, 5))
    assert cloud.gpu_is_supported("NVIDIA L4", (8, 9))
    assert not cloud.gpu_is_supported("Tesla P100-PCIE-16GB", (6, 0))


def test_gpu_guard_requires_single_supported_device():
    fake = _fake_torch_cuda([("Tesla T4", (7, 5), 16 * 1024**3)])
    report = cloud.assert_supported_single_gpu(fake)
    assert report["ok"]
    assert report["devices"][0]["name"] == "Tesla T4"

    with pytest.raises(RuntimeError, match="Unsupported GPU"):
        cloud.assert_supported_single_gpu(_fake_torch_cuda([("Tesla P100", (6, 0), 16 * 1024**3)]))

    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        cloud.assert_supported_single_gpu(
            _fake_torch_cuda([("Tesla T4", (7, 5), 16 * 1024**3), ("Tesla T4", (7, 5), 16 * 1024**3)])
        )


def test_latent_signal_check_requires_divergence():
    good = cloud.analyze_latent_signal(
        normal_text="HIDDEN_SIGNAL_ALPHA active",
        ablated_text="",
        normal_top_token_id=1,
        ablated_top_token_id=2,
        logit_max_abs_diff=0.25,
        logit_mean_abs_diff=0.01,
        threshold=1e-4,
    )
    assert good["ok"]

    dead = cloud.analyze_latent_signal(
        normal_text="same",
        ablated_text="same",
        normal_top_token_id=1,
        ablated_top_token_id=1,
        logit_max_abs_diff=0.0,
        logit_mean_abs_diff=0.0,
        threshold=1e-4,
    )
    assert not dead["ok"]


def test_hidden_smoke_preserves_cache_object_when_cloning_and_zeroing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import latent_hidden_smoke

    key = torch.ones(1, 2, 5, 3)
    value = torch.full((1, 2, 5, 3), 2.0)
    cache = _FakeDynamicCache.from_legacy_cache(((key, value),))

    cloned = latent_hidden_smoke._clone_past_key_values(cache)
    ablated = latent_hidden_smoke._zero_past_key_values(cache)

    assert isinstance(cloned, _FakeDynamicCache)
    assert isinstance(ablated, _FakeDynamicCache)
    assert cloned is not cache
    assert torch.equal(cloned.key_cache[0], key)
    assert torch.equal(cloned.value_cache[0], value)
    assert torch.equal(ablated.key_cache[0], torch.zeros_like(key))
    assert torch.equal(ablated.value_cache[0], torch.zeros_like(value))
    key.zero_()
    assert torch.equal(cloned.key_cache[0], torch.ones_like(cloned.key_cache[0]))


def test_model_backend_4bit_load_uses_bitsandbytes_and_skips_to(monkeypatch):
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch(types.SimpleNamespace):
        pass

    fake_torch = FakeTorch(cuda=FakeCuda(), float16="float16", float32="float32")

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 2
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return cls()

    class FakeModel:
        def to(self, device):
            calls["to_called"] = device
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls["bnb_config"] = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = ModelBackend("Qwen/Qwen3-8B", quantization="4bit")
    assert backend.quantization == "4bit"
    assert calls["bnb_config"]["load_in_4bit"] is True
    assert calls["model"][1]["device_map"] == {"": 0}
    assert "to_called" not in calls


def test_tier2_gate_analyzer_marks_validity():
    records = []
    for mode, passes in (("A_single", 10), ("B_textmas", 8), ("C_latentmas", 7)):
        for index in range(15):
            records.append(types.SimpleNamespace(mode=mode, task_family="family", passed=index < passes))
    result = cloud.summarize_gate(records)
    assert result["valid_text_baseline"]
    assert result["a_b_gap"] <= 0.15


def test_package_cloud_excludes_old_outputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import package_cloud

    packaged = [path.relative_to(package_cloud.PROJECT_ROOT).as_posix() for path in package_cloud.collect_package_files()]
    assert "src/latent_agent/cloud.py" in packaged
    assert "notebooks/tier2_short_gate.ipynb" in packaged
    assert "README.md" in packaged
    assert not any(path.endswith(".md") and path != "README.md" for path in packaged)
    assert "scripts/run_tier2_full_sweep.py" in packaged
    assert not any(path.startswith("exports/") for path in packaged)
    assert not any(path.startswith("reports/") for path in packaged)
    assert not any("secret" in path.lower() or "hf_token" in path.lower() for path in packaged)


def test_notebook_contains_required_tier2_cells():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "tier2_short_gate.ipynb").read_text())
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "CUDA_VISIBLE_DEVICES" in joined
    assert "torchvision==0.21.0" in joined
    assert "torchaudio==2.6.0" in joined
    assert "bitsandbytes==0.46.1" in joined
    assert "scripts/run_tier2_gate.py" in joined
    assert "Qwen/Qwen3-8B" in joined
    assert "tier2_short_gate_results.zip" in joined


def test_import_tier2_results_recomputes_valid_gate(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import import_tier2_results

    result_zip = tmp_path / "tier2_short_gate_results.zip"
    _write_fake_tier2_zip(result_zip, a_passes=10, b_passes=8, c_passes=7)
    audit = import_tier2_results.import_tier2_results(result_zip, tmp_path / "imported")
    assert audit["metrics_rows"] == 45
    assert audit["hidden_smoke_ok"]
    assert audit["tool_roundtrip_ok"]
    assert audit["gate"]["valid_text_baseline"]
    assert (tmp_path / "imported" / "tier2_import_audit.md").exists()


def test_import_tier2_results_rejects_forbidden_files(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import import_tier2_results

    result_zip = tmp_path / "bad.zip"
    _write_fake_tier2_zip(result_zip, a_passes=10, b_passes=8, c_passes=7, extra_files={"project/hf-cache/model.bin": "bad"})
    with pytest.raises(RuntimeError, match="forbidden"):
        import_tier2_results.import_tier2_results(result_zip, tmp_path / "imported")


def test_import_tier2_results_fails_missing_required_artifact(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import import_tier2_results

    result_zip = tmp_path / "missing.zip"
    _write_fake_tier2_zip(result_zip, a_passes=10, b_passes=8, c_passes=7, omit={"reports/tier2_short_gate_summary.md"})
    with pytest.raises(FileNotFoundError, match="tier2_short_gate_summary"):
        import_tier2_results.import_tier2_results(result_zip, tmp_path / "imported")


def _fake_torch_cuda(devices):
    class Props:
        def __init__(self, total_memory):
            self.total_memory = total_memory

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return len(devices)

        @staticmethod
        def get_device_name(index):
            return devices[index][0]

        @staticmethod
        def get_device_capability(index):
            return devices[index][1]

        @staticmethod
        def get_device_properties(index):
            return Props(devices[index][2])

    return types.SimpleNamespace(cuda=Cuda())


class _FakeDynamicCache:
    def __init__(self, legacy=None):
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        if legacy is not None:
            for key_states, value_states in legacy:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            if self.key_cache:
                self._seen_tokens = int(self.key_cache[0].shape[-2])

    @classmethod
    def from_legacy_cache(cls, legacy):
        return cls(legacy)

    def to_legacy_cache(self):
        return tuple(zip(self.key_cache, self.value_cache))

    def get_seq_length(self):
        return self._seen_tokens


def _write_fake_tier2_zip(
    path: Path,
    *,
    a_passes: int,
    b_passes: int,
    c_passes: int,
    extra_files: dict[str, str] | None = None,
    omit: set[str] | None = None,
) -> None:
    omit = omit or set()
    extra_files = extra_files or {}
    required: dict[str, str] = {
        "exports/tier2_environment.json": json.dumps({"dependency_report": {"ok": True}, "gpu_report": {"ok": True}}),
        "exports/tier2_latent_hidden_smoke.json": json.dumps({"ok": True, "signal_check": {"ok": True}}),
        "exports/tier2_latent_tool_roundtrip.json": json.dumps({"ok": True}),
        "exports/tier2_short_gate_metrics.csv": _fake_metrics_csv(a_passes, b_passes, c_passes),
        "exports/tier2_short_gate_gate.json": json.dumps({"valid_text_baseline": abs(a_passes / 15 - b_passes / 15) <= 0.15 and b_passes > 0}),
        "reports/tier2_short_gate_summary.md": "# Tier 2 Short Gate\n\ncoarse screen\n",
    }
    with zipfile.ZipFile(path, "w") as zf:
        for suffix, content in required.items():
            if suffix in omit:
                continue
            zf.writestr(f"project/{suffix}", content)
        for name, content in extra_files.items():
            zf.writestr(name, content)


def _fake_metrics_csv(a_passes: int, b_passes: int, c_passes: int) -> str:
    header = [
        "run_id",
        "task_id",
        "mode",
        "repeat",
        "passed",
        "score",
        "message",
        "task_family",
    ]
    lines = [",".join(header)]
    families = ["campaign_roi", "orders_kpi", "sensor_quality"]
    counters = {"A_single": 0, "B_textmas": 0, "C_latentmas": 0}
    pass_limits = {"A_single": a_passes, "B_textmas": b_passes, "C_latentmas": c_passes}
    for family in families:
        for repeat in range(1, 6):
            for mode in ("A_single", "B_textmas", "C_latentmas"):
                counters[mode] += 1
                passed = counters[mode] <= pass_limits[mode]
                lines.append(
                    ",".join(
                        [
                            f"{mode}_{family}_r{repeat}",
                            f"{family}_short",
                            mode,
                            str(repeat),
                            str(passed),
                            "1.0" if passed else "0.0",
                            "ok" if passed else "failed",
                            family,
                        ]
                    )
                )
    return "\n".join(lines) + "\n"
