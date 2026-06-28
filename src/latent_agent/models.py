from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationMetrics:
    input_tokens: int
    output_tokens: int
    forward_passes: int
    model_latency_ms: float
    wall_latency_s: float
    peak_vram_mb: float


@dataclass
class GenerationResult:
    text: str
    metrics: GenerationMetrics


class ModelBackend:
    def __init__(self, model_id: str, device: str | None = None, *, quantization: str = "none") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.quantization = quantization
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if quantization not in {"none", "4bit"}:
            raise ValueError(f"Unsupported quantization: {quantization}")
        self.is_cuda = str(self.device).startswith("cuda")
        if quantization == "4bit" and not self.is_cuda:
            raise RuntimeError("4-bit quantization requires CUDA for this prototype.")
        dtype = torch.float16 if self.is_cuda else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": {"": 0},
                }
            )
        else:
            load_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if quantization == "none":
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generation_seed: int | None = None,
    ) -> GenerationResult:
        torch = self.torch
        if generation_seed is not None:
            torch.manual_seed(int(generation_seed))
            if self.is_cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(generation_seed))
        model_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(model_prompt, return_tensors="pt").to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = 0
        forward_passes = 0
        peak_vram_mb = 0.0

        original_forward = self.model.forward

        def counted_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal forward_passes
            forward_passes += 1
            return original_forward(*args, **kwargs)

        self.model.forward = counted_forward  # type: ignore[method-assign]

        use_cuda_events = self.is_cuda and torch.cuda.is_available()
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
        try:
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )
        finally:
            self.model.forward = original_forward  # type: ignore[method-assign]

        if use_cuda_events and start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            model_latency_ms = float(start_event.elapsed_time(end_event))
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        else:
            model_latency_ms = (time.perf_counter() - wall_start) * 1000.0

        wall_latency_s = time.perf_counter() - wall_start
        output_ids = generated[0][input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                forward_passes=forward_passes,
                model_latency_ms=model_latency_ms,
                wall_latency_s=wall_latency_s,
                peak_vram_mb=peak_vram_mb,
            ),
        )

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
