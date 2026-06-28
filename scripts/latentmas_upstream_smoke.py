from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.runtime import configure_runtime, project_path, runtime_path  # noqa: E402


def _load_latentmas_modelwrapper(repo_path: Path):
    models_path = repo_path / "models.py"
    if not models_path.exists():
        raise FileNotFoundError(f"LatentMAS models.py not found at {models_path}")
    spec = importlib.util.spec_from_file_location("latentmas_upstream_models", models_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {models_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._past_length = _cache_length_compat
    return module.ModelWrapper, module._past_length


def _cache_length_compat(past_key_values) -> int:
    if not past_key_values:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    k = past_key_values[0][0]
    return int(k.shape[-2])


def _encode_prompt(wrapper, prompt: str):
    rendered = wrapper.render_chat(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    encoded = wrapper.tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return rendered, encoded["input_ids"].to(wrapper.device), encoded["attention_mask"].to(wrapper.device)


def _decode_from_past_compat(wrapper, input_ids, attention_mask, *, past_key_values, max_new_tokens: int):
    import torch

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=wrapper.device)
    else:
        attention_mask = attention_mask.to(wrapper.device)
    past_len = _cache_length_compat(past_key_values)
    if past_len > 0:
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    generated_ids = []
    with torch.no_grad():
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=-1,
            )
            outputs = wrapper.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    if not generated_ids:
        return [""], past
    generated_tensor = torch.cat(generated_ids, dim=-1)
    text = wrapper.tokenizer.decode(generated_tensor[0], skip_special_tokens=True).strip()
    return [text], past


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct-import LatentMAS hidden-state smoke test.")
    parser.add_argument("--repo", default=str(runtime_path("third_party", "LatentMAS")))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--latent-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default=str(project_path("exports", "latentmas_upstream_smoke.json")))
    args = parser.parse_args()

    configure_runtime(create=True)

    import torch

    repo_path = Path(args.repo)
    ModelWrapper, past_length = _load_latentmas_modelwrapper(repo_path)
    wrapper_args = SimpleNamespace(latent_space_realign=False)
    start = time.perf_counter()
    wrapper = ModelWrapper(args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"), use_vllm=False, args=wrapper_args)

    prompt = (
        "You are testing latent memory. Remember the word LATENT_OK and answer briefly after latent thinking."
    )
    rendered, input_ids, attention_mask = _encode_prompt(wrapper, prompt)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    before_len = 0
    past = wrapper.generate_latent_batch(
        input_ids,
        attention_mask=attention_mask,
        latent_steps=args.latent_steps,
        past_key_values=None,
    )
    after_latent_len = int(past_length(past))
    decode_prompt = "Now answer with the remembered marker and one short sentence."
    _, decode_ids, decode_mask = _encode_prompt(wrapper, decode_prompt)
    generated, decoded_past = _decode_from_past_compat(
        wrapper,
        decode_ids,
        decode_mask,
        max_new_tokens=args.max_new_tokens,
        past_key_values=past,
    )
    after_decode_len = int(past_length(decoded_past))
    text = generated[0].strip() if generated else ""
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    elapsed_s = time.perf_counter() - start

    result = {
        "ok": bool(text and after_latent_len > before_len and after_decode_len >= after_latent_len),
        "model": args.model,
        "repo": str(repo_path),
        "latent_steps": args.latent_steps,
        "before_past_len": before_len,
        "after_latent_past_len": after_latent_len,
        "after_decode_past_len": after_decode_len,
        "text_preview": text[:500],
        "rendered_prompt_tokens": int(input_ids.shape[-1]),
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "compat_notes": [
            "Patched upstream _past_length for Transformers DynamicCache.",
            "Decoded from upstream latent past with manual forward loop because model.generate rejected cache_position.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
