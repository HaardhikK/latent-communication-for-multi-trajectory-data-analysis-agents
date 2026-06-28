from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.models import ModelBackend  # noqa: E402
from latent_agent.runtime import configure_runtime, project_path  # noqa: E402


PRIMARY_MODEL = "Qwen/Qwen3-1.7B"
FALLBACK_MODEL = "Qwen/Qwen3-0.6B"


def try_model(model_id: str) -> dict[str, object]:
    backend = ModelBackend(model_id)
    result = backend.generate(
        "Write a one-line Python program that prints the word ready. Return only code.",
        max_new_tokens=64,
    )
    return {
        "model_id": model_id,
        "ok": True,
        "text_preview": result.text[:200],
        "metrics": result.metrics.__dict__,
    }


def main() -> int:
    configure_runtime(create=True)
    results: list[dict[str, object]] = []
    for model_id in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            report = try_model(model_id)
            results.append(report)
            out = {"selected_model": model_id, "attempts": results}
            break
        except Exception as exc:
            results.append(
                {
                    "model_id": model_id,
                    "ok": False,
                    "error": repr(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                }
            )
    else:
        out = {"selected_model": None, "attempts": results}

    path = project_path("exports", "smoke_generate.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["selected_model"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
