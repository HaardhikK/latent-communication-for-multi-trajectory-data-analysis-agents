from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


FIXED_PROMPT = "fixed_prompt"
COORDINATION = "coordination"
TOOL_IO = "tool_io"
TOKEN_CATEGORIES = (FIXED_PROMPT, COORDINATION, TOOL_IO)


@dataclass(frozen=True)
class PromptPart:
    label: str
    text: str
    category: str = FIXED_PROMPT

    def __post_init__(self) -> None:
        if self.category not in TOKEN_CATEGORIES:
            raise ValueError(f"Unknown token category: {self.category}")


@dataclass
class TokenLedger:
    input_tokens: dict[str, int] = field(default_factory=lambda: {c: 0 for c in TOKEN_CATEGORIES})
    output_tokens: dict[str, int] = field(default_factory=lambda: {c: 0 for c in TOKEN_CATEGORIES})
    events: list[dict[str, Any]] = field(default_factory=list)

    def add_prompt_parts(self, tokenizer: Any, call_name: str, parts: list[PromptPart]) -> int:
        total = 0
        for part in parts:
            count = count_tokens(tokenizer, part.text)
            self.input_tokens[part.category] += count
            total += count
            self.events.append(
                {
                    "call_name": call_name,
                    "direction": "input",
                    "label": part.label,
                    "category": part.category,
                    "tokens": count,
                }
            )
        return total

    def add_generated(self, tokenizer: Any, call_name: str, text: str, category: str) -> int:
        if category not in TOKEN_CATEGORIES:
            raise ValueError(f"Unknown token category: {category}")
        count = count_tokens(tokenizer, text)
        self.output_tokens[category] += count
        self.events.append(
            {
                "call_name": call_name,
                "direction": "output",
                "label": "generated",
                "category": category,
                "tokens": count,
            }
        )
        return count

    @property
    def coordination_tokens(self) -> int:
        return self.input_tokens[COORDINATION] + self.output_tokens[COORDINATION]

    @property
    def tool_io_tokens(self) -> int:
        return self.input_tokens[TOOL_IO] + self.output_tokens[TOOL_IO]

    @property
    def fixed_prompt_tokens(self) -> int:
        return self.input_tokens[FIXED_PROMPT] + self.output_tokens[FIXED_PROMPT]

    @property
    def coordination_fraction(self) -> float:
        denom = self.coordination_tokens + self.tool_io_tokens
        return self.coordination_tokens / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": dict(self.input_tokens),
            "output_tokens": dict(self.output_tokens),
            "coordination_tokens": self.coordination_tokens,
            "tool_io_tokens": self.tool_io_tokens,
            "fixed_prompt_tokens": self.fixed_prompt_tokens,
            "coordination_fraction": self.coordination_fraction,
            "events": list(self.events),
        }


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(parts: list[PromptPart]) -> str:
    return "\n\n".join(part.text.rstrip() for part in parts if part.text is not None).strip() + "\n"


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return max((block.strip() for block in fenced), key=len)
    cleaned = text.strip()
    if cleaned.lower().startswith("python\n"):
        cleaned = cleaned.split("\n", 1)[1]
    return cleaned.strip()
