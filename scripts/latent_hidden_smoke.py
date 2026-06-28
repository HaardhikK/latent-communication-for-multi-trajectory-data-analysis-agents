from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.cloud import analyze_latent_signal, assert_supported_single_gpu, force_single_gpu_env  # noqa: E402
from latent_agent.latent_backend import LatentBackend, past_length  # noqa: E402
from latent_agent.metrics import write_json  # noqa: E402
from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2 hidden-state smoke with normal-vs-ablated latent signal check.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--logit-threshold", type=float, default=1e-4)
    parser.add_argument("--skip-gpu-guard", action="store_true")
    parser.add_argument("--out", default=str(project_path("exports", "tier2_latent_hidden_smoke.json")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_single_gpu_env()
    configure_runtime(create=True)

    import torch

    gpu = None
    if not args.skip_gpu_guard and args.quantization == "4bit":
        gpu = assert_supported_single_gpu(torch)

    started = time.perf_counter()
    backend = ModelBackend(args.model, quantization=args.quantization)
    latent = LatentBackend(backend)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    planner_prompt = (
        "Think internally only. Store this exact marker in latent memory: HIDDEN_SIGNAL_ALPHA. "
        "The continuation should later mention that marker and say the latent channel is active."
    )
    appended = latent.append_latent(planner_prompt, latent_steps=args.latent_steps)
    seed_past = _clone_past_key_values(appended.past_key_values)
    ablated_past = _zero_past_key_values(seed_past)

    continuation_prompt = "Continue from latent memory. Reply with the remembered marker and one short sentence."
    normal_logits = _first_step_logits(backend, latent, continuation_prompt, _clone_past_key_values(seed_past))
    ablated_logits = _first_step_logits(backend, latent, continuation_prompt, _clone_past_key_values(ablated_past))
    diff = (normal_logits - ablated_logits).abs()
    normal_top = int(normal_logits.argmax(dim=-1).item())
    ablated_top = int(ablated_logits.argmax(dim=-1).item())

    normal = latent.decode_from_past(
        continuation_prompt,
        max_new_tokens=args.max_new_tokens,
        past_key_values=_clone_past_key_values(seed_past),
    )
    ablated = latent.decode_from_past(
        continuation_prompt,
        max_new_tokens=args.max_new_tokens,
        past_key_values=_clone_past_key_values(ablated_past),
    )
    signal = analyze_latent_signal(
        normal_text=normal.text,
        ablated_text=ablated.text,
        normal_top_token_id=normal_top,
        ablated_top_token_id=ablated_top,
        logit_max_abs_diff=float(diff.max().item()),
        logit_mean_abs_diff=float(diff.mean().item()),
        threshold=args.logit_threshold,
    )
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    result = {
        "ok": bool(
            signal["ok"]
            and past_length(seed_past) > 0
            and past_length(seed_past) > appended.metrics.input_tokens
            and normal.text.strip()
        ),
        "model": args.model,
        "quantization": args.quantization,
        "latent_steps": args.latent_steps,
        "prompt_input_tokens": appended.metrics.input_tokens,
        "past_length": past_length(seed_past),
        "normal_text_preview": normal.text[:500],
        "ablated_text_preview": ablated.text[:500],
        "signal_check": signal,
        "gpu": gpu,
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - started,
        "deferred_disambiguator": (
            "If 8B/4-bit C is ambiguous, run a later 2xT4/fp16 confirmation to separate NF4 hidden-state signal loss "
            "from latent-mechanism failure. Do not run that disambiguator in the short gate."
        ),
    }
    out_path = Path(args.out)
    write_json(out_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _clone_past_key_values(past_key_values: Any) -> Any:
    """Clone a KV cache while preserving modern Transformers Cache objects.

    Qwen3 in Transformers 4.53+ rejects legacy tuple caches, so the smoke signal
    check must not call ``to_legacy_cache()`` unless it converts back to Cache.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return _clone_key_value_cache_object(past_key_values, zero=False)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = _clone_legacy_layers(past_key_values.to_legacy_cache(), zero=False)
        if hasattr(type(past_key_values), "from_legacy_cache") and type(past_key_values).from_legacy_cache is not None:
            return type(past_key_values).from_legacy_cache(legacy)
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:
            pass
    if isinstance(past_key_values, tuple):
        return _clone_legacy_layers(past_key_values, zero=False)
    return copy.deepcopy(past_key_values)


def _zero_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return _clone_key_value_cache_object(past_key_values, zero=True)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = _clone_legacy_layers(past_key_values.to_legacy_cache(), zero=True)
        if hasattr(type(past_key_values), "from_legacy_cache") and type(past_key_values).from_legacy_cache is not None:
            return type(past_key_values).from_legacy_cache(legacy)
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception:
            pass
    if isinstance(past_key_values, tuple):
        return _clone_legacy_layers(past_key_values, zero=True)
    clone = copy.deepcopy(past_key_values)
    _zero_cache_tensors_in_place(clone)
    return clone


def _clone_key_value_cache_object(past_key_values: Any, *, zero: bool) -> Any:
    legacy = tuple(
        (
            _clone_tensor(key_states, zero=zero),
            _clone_tensor(value_states, zero=zero),
        )
        for key_states, value_states in zip(past_key_values.key_cache, past_key_values.value_cache)
    )
    if hasattr(type(past_key_values), "from_legacy_cache") and type(past_key_values).from_legacy_cache is not None:
        cloned = type(past_key_values).from_legacy_cache(legacy)
    else:
        try:
            cloned = type(past_key_values)(legacy)
        except TypeError:
            cloned = copy.deepcopy(past_key_values)
            cloned.key_cache = [layer[0] for layer in legacy]
            cloned.value_cache = [layer[1] for layer in legacy]
    if hasattr(cloned, "_seen_tokens") and hasattr(past_key_values, "_seen_tokens"):
        cloned._seen_tokens = int(getattr(past_key_values, "_seen_tokens"))
    return cloned


def _clone_legacy_layers(past_key_values: Any, *, zero: bool) -> Any:
    return tuple(tuple(_clone_tensor(item, zero=zero) for item in layer) for layer in past_key_values)


def _clone_tensor(tensor: Any, *, zero: bool) -> Any:
    cloned = tensor.detach().clone()
    if zero:
        cloned.zero_()
    return cloned


def _zero_cache_tensors_in_place(cache: Any) -> None:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for tensor in list(cache.key_cache) + list(cache.value_cache):
            if hasattr(tensor, "zero_"):
                tensor.zero_()


def _first_step_logits(backend: ModelBackend, latent: LatentBackend, prompt: str, past_key_values: Any):
    encoded = latent._encode_prompt(prompt)
    attention_mask = latent._extend_attention_mask(encoded["attention_mask"], past_key_values)
    with backend.torch.no_grad():
        outputs = backend.model(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return outputs.logits[:, -1, :].detach().float().cpu()


if __name__ == "__main__":
    raise SystemExit(main())
