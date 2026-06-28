from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .models import ModelBackend


# Adapted from Gen-Verse/LatentMAS (Apache-2.0), especially models.py:
# - latent KV-cache growth via last-layer hidden states
# - hidden-state-to-input-embedding norm alignment
# - cached decoding from latent past_key_values
#
# This local version adds compatibility for Transformers DynamicCache and uses a
# manual greedy decode loop because Qwen/Qwen3-1.7B with the installed
# Transformers rejects LatentMAS's cache_position kwarg in model.generate().


@dataclass
class LatentCallMetrics:
    input_tokens: int
    output_tokens: int
    forward_passes: int
    model_latency_ms: float
    wall_latency_s: float
    peak_vram_mb: float
    latent_steps: int = 0


@dataclass
class LatentCallResult:
    past_key_values: Any
    text: str
    metrics: LatentCallMetrics


def past_length(past_key_values: Any) -> int:
    if not past_key_values:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    k = past_key_values[0][0]
    return int(k.shape[-2])


class LatentBackend:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.device = backend.device
        self.torch = backend.torch
        self._target_embedding_norm: torch.Tensor | None = None

    def append_latent(
        self,
        prompt: str,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentCallResult:
        torch = self.torch
        encoded = self._encode_prompt(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_tokens = int(attention_mask.sum().item())

        forward_passes = 0
        use_cuda_events = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda_events:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        else:
            start_event = end_event = None

        wall_start = time.perf_counter()
        with torch.no_grad():
            full_attention_mask = self._extend_attention_mask(attention_mask, past_key_values)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            forward_passes += 1
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            for _ in range(latent_steps):
                latent_vec = self._align_hidden_to_embedding(last_hidden)
                latent_embed = latent_vec.unsqueeze(1)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_length(past) + 1),
                    dtype=torch.long,
                    device=latent_embed.device,
                )
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                forward_passes += 1
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]

        model_latency_ms, peak_vram_mb = self._finish_timing(use_cuda_events, start_event, end_event, wall_start)
        return LatentCallResult(
            past_key_values=past,
            text="",
            metrics=LatentCallMetrics(
                input_tokens=input_tokens,
                output_tokens=0,
                forward_passes=forward_passes,
                model_latency_ms=model_latency_ms,
                wall_latency_s=time.perf_counter() - wall_start,
                peak_vram_mb=peak_vram_mb,
                latent_steps=latent_steps,
            ),
        )

    def decode_from_past(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        past_key_values: Any = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generation_seed: int | None = None,
    ) -> LatentCallResult:
        torch = self.torch
        if generation_seed is not None:
            torch.manual_seed(int(generation_seed))
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(generation_seed))
        encoded = self._encode_prompt(prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_tokens = int(attention_mask.sum().item())

        forward_passes = 0
        generated_ids: list[torch.Tensor] = []
        use_cuda_events = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda_events:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        else:
            start_event = end_event = None

        wall_start = time.perf_counter()
        with torch.no_grad():
            full_attention_mask = self._extend_attention_mask(attention_mask, past_key_values)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            forward_passes += 1
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for _ in range(max_new_tokens):
                next_token = self._next_token(logits, temperature=temperature, top_p=top_p)
                token_id = int(next_token.item())
                if self.tokenizer.eos_token_id is not None and token_id == int(self.tokenizer.eos_token_id):
                    break
                generated_ids.append(next_token)
                full_attention_mask = torch.cat(
                    [
                        full_attention_mask,
                        torch.ones(
                            (full_attention_mask.shape[0], 1),
                            dtype=full_attention_mask.dtype,
                            device=full_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=full_attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                forward_passes += 1
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

        text = ""
        if generated_ids:
            generated_tensor = torch.cat(generated_ids, dim=-1)
            text = self.tokenizer.decode(generated_tensor[0], skip_special_tokens=True).strip()
        model_latency_ms, peak_vram_mb = self._finish_timing(use_cuda_events, start_event, end_event, wall_start)
        return LatentCallResult(
            past_key_values=past,
            text=text,
            metrics=LatentCallMetrics(
                input_tokens=input_tokens,
                output_tokens=len(generated_ids),
                forward_passes=forward_passes,
                model_latency_ms=model_latency_ms,
                wall_latency_s=time.perf_counter() - wall_start,
                peak_vram_mb=peak_vram_mb,
                latent_steps=0,
            ),
        )

    def _next_token(self, logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled = logits / max(float(temperature), 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if 0 < top_p < 1:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > float(top_p)
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(dim=-1, index=sampled)
        return torch.multinomial(probs, num_samples=1)

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        model_prompt = self.backend._format_prompt(prompt)
        encoded = self.tokenizer(model_prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _extend_attention_mask(self, attention_mask: torch.Tensor, past_key_values: Any) -> torch.Tensor:
        past_len = past_length(past_key_values)
        if past_len <= 0:
            return attention_mask
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([past_mask, attention_mask], dim=-1)

    def _align_hidden_to_embedding(self, hidden: torch.Tensor) -> torch.Tensor:
        target_norm = self._mean_input_embedding_norm(hidden.device, hidden.dtype)
        aligned = hidden.to(torch.float32)
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm.to(aligned.device, aligned.dtype) / aligned_norm)
        return aligned.to(hidden.dtype)

    def _mean_input_embedding_norm(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        if self._target_embedding_norm is None:
            weight = self.model.get_input_embeddings().weight.detach().to(device=device, dtype=torch.float32)
            self._target_embedding_norm = weight.norm(dim=1).mean().detach()
        return self._target_embedding_norm.to(device=device, dtype=dtype)

    def _finish_timing(self, use_cuda_events: bool, start_event: Any, end_event: Any, wall_start: float) -> tuple[float, float]:
        torch = self.torch
        peak_vram_mb = 0.0
        if use_cuda_events and start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            model_latency_ms = float(start_event.elapsed_time(end_event))
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        else:
            model_latency_ms = (time.perf_counter() - wall_start) * 1000.0
        return model_latency_ms, peak_vram_mb
