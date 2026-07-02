from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CodeQuality:
    ast_ok: bool
    empty: bool
    repetition_ratio: float
    ast_error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ast_parse_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code or "")
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, ""


def is_empty_code(code: str) -> bool:
    stripped = (code or "").strip()
    if not stripped:
        return True
    substantive = [
        line
        for line in stripped.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not substantive:
        return True
    ok, _ = ast_parse_check(stripped)
    if not ok:
        return len(_code_tokens(stripped)) < 5
    try:
        return len(ast.parse(stripped).body) == 0
    except SyntaxError:
        return False


def repetition_ratio(code: str, *, window: int = 8) -> float:
    tokens = _code_tokens(code)
    if len(tokens) < window * 2:
        return 0.0
    windows = [tuple(tokens[index : index + window]) for index in range(0, len(tokens) - window + 1)]
    if not windows:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (len(set(windows)) / len(windows))))


def assess_code_quality(code: str) -> CodeQuality:
    ast_ok, ast_error = ast_parse_check(code)
    return CodeQuality(
        ast_ok=ast_ok,
        empty=is_empty_code(code),
        repetition_ratio=repetition_ratio(code),
        ast_error=ast_error,
    )


def _code_tokens(code: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%=()[\]{}.,:]", code or "")
