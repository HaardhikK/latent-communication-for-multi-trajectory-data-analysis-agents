from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_agent.metrics import write_json  # noqa: E402


REQUIRED_SUFFIXES = (
    "exports/tier2_environment.json",
    "exports/tier2_latent_hidden_smoke.json",
    "exports/tier2_latent_tool_roundtrip.json",
    "exports/tier2_short_gate_metrics.csv",
    "exports/tier2_short_gate_gate.json",
    "reports/tier2_short_gate_summary.md",
)
FORBIDDEN_SUBSTRINGS = (
    "hf-cache",
    "huggingface",
    "kaggle.json",
    "hf_token",
    "kgat",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
)
FORBIDDEN_SECRET_WORDS = ("secret", "token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and audit a Tier 2 short-gate result zip.")
    parser.add_argument("--zip", required=True, help="Path to tier2_short_gate_results.zip from Kaggle/Colab.")
    parser.add_argument("--out-dir", default="", help="Optional import folder. Defaults to imports/tier2_short_gate_<UTC timestamp>.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_import_dir()
    audit = import_tier2_results(zip_path, out_dir)
    print((out_dir / "tier2_import_audit.md").read_text(encoding="utf-8"))
    print(json.dumps({"import_dir": str(out_dir), "valid_text_baseline": audit["gate"]["valid_text_baseline"]}, indent=2))
    return 0


def import_tier2_results(zip_path: Path, out_dir: Path) -> dict[str, Any]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Tier 2 result zip not found: {zip_path}")
    _validate_zip_names(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    artifacts = _find_required_artifacts(extract_dir)
    env = _load_json(artifacts["exports/tier2_environment.json"])
    smoke = _load_json(artifacts["exports/tier2_latent_hidden_smoke.json"])
    roundtrip = _load_json(artifacts["exports/tier2_latent_tool_roundtrip.json"])
    cloud_gate = _load_json(artifacts["exports/tier2_short_gate_gate.json"])
    rows = _load_metrics(artifacts["exports/tier2_short_gate_metrics.csv"])
    gate = _recompute_gate(rows)

    failures = []
    if not _nested_ok(env.get("dependency_report"), default=True):
        failures.append("dependency guard did not pass")
    if not _nested_ok(env.get("gpu_report"), default=False):
        failures.append("GPU guard did not pass")
    if not smoke.get("ok"):
        failures.append("hidden smoke ok=false")
    if not smoke.get("signal_check", {}).get("ok"):
        failures.append("hidden smoke signal_check.ok=false")
    if not roundtrip.get("ok"):
        failures.append("tool-roundtrip ok=false")
    if len(rows) != 45:
        failures.append(f"metrics row count is {len(rows)}, expected 45")
    if cloud_gate.get("valid_text_baseline") != gate["valid_text_baseline"]:
        failures.append("cloud gate validity disagrees with recomputed gate")

    audit = {
        "zip_path": str(zip_path),
        "import_dir": str(out_dir),
        "required_artifacts": {key: str(path) for key, path in artifacts.items()},
        "environment_ok": not any(item for item in failures if "dependency" in item or "GPU" in item),
        "hidden_smoke_ok": bool(smoke.get("ok") and smoke.get("signal_check", {}).get("ok")),
        "tool_roundtrip_ok": bool(roundtrip.get("ok")),
        "metrics_rows": len(rows),
        "gate": gate,
        "cloud_gate": cloud_gate,
        "failures": failures,
    }
    write_json(out_dir / "tier2_import_audit.json", audit)
    (out_dir / "tier2_import_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def _default_import_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "imports" / f"tier2_short_gate_{stamp}"


def _validate_zip_names(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    bad = []
    for name in names:
        normalized = name.replace("\\", "/").lower()
        if any(part in normalized for part in FORBIDDEN_SUBSTRINGS):
            bad.append(name)
            continue
        filename = Path(normalized).name
        if any(word in filename for word in FORBIDDEN_SECRET_WORDS) and not filename.startswith("test_token_split"):
            bad.append(name)
    if bad:
        preview = "\n".join(f"- {name}" for name in bad[:20])
        raise RuntimeError(f"Result zip contains forbidden cache/model/secret-like files:\n{preview}")


def _find_required_artifacts(root: Path) -> dict[str, Path]:
    all_files = [path for path in root.rglob("*") if path.is_file()]
    found: dict[str, Path] = {}
    for suffix in REQUIRED_SUFFIXES:
        suffix_norm = suffix.replace("\\", "/")
        matches = [path for path in all_files if path.as_posix().endswith(suffix_norm)]
        if not matches:
            raise FileNotFoundError(f"Required Tier 2 artifact not found in import zip: {suffix}")
        found[suffix] = matches[0]
    return found


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _recompute_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, str]]] = {}
    by_family_mode: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
        by_family_mode.setdefault((row.get("task_family", ""), row["mode"]), []).append(row)

    def passed(row: dict[str, str]) -> bool:
        return str(row.get("passed", "")).lower() in {"1", "true", "yes"}

    def rate(items: list[dict[str, str]]) -> float:
        return sum(passed(row) for row in items) / len(items) if items else 0.0

    a = by_mode.get("A_single", [])
    b = by_mode.get("B_textmas", [])
    c = by_mode.get("C_latentmas", [])
    a_rate = rate(a)
    b_rate = rate(b)
    c_rate = rate(c)
    gap = abs(a_rate - b_rate)
    family_rows = []
    for family in sorted({row.get("task_family", "") for row in rows}):
        for mode in ("A_single", "B_textmas", "C_latentmas"):
            items = by_family_mode.get((family, mode), [])
            family_rows.append(
                {
                    "family": family,
                    "mode": mode,
                    "runs": len(items),
                    "passes": sum(passed(row) for row in items),
                    "pass_rate": rate(items),
                }
            )
    return {
        "valid_text_baseline": bool(a and b and b_rate > 0 and gap <= 0.15),
        "a_runs": len(a),
        "b_runs": len(b),
        "c_runs": len(c),
        "a_pass_rate": a_rate,
        "b_pass_rate": b_rate,
        "c_pass_rate": c_rate,
        "a_b_gap": gap,
        "family_rows": family_rows,
        "coarse_screen_note": "15 runs/mode gives pass-rate resolution of about 0.067; this is a go/no-go screen, not final accuracy proof.",
    }


def _nested_ok(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, dict):
        return bool(value.get("ok", default))
    return bool(value)


def _audit_markdown(audit: dict[str, Any]) -> str:
    gate = audit["gate"]
    status = "PASSED" if gate["valid_text_baseline"] else "FAILED"
    lines = [
        "# Tier 2 Short-Gate Import Audit",
        "",
        f"- Result zip: `{audit['zip_path']}`",
        f"- Metrics rows: `{audit['metrics_rows']}`",
        f"- Dependency/GPU environment OK: `{audit['environment_ok']}`",
        f"- Hidden-signal smoke OK: `{audit['hidden_smoke_ok']}`",
        f"- Tool-roundtrip OK: `{audit['tool_roundtrip_ok']}`",
        f"- Text-baseline gate: **{status}**",
        f"- A pass rate: `{gate['a_pass_rate']:.3f}` over `{gate['a_runs']}` runs",
        f"- B pass rate: `{gate['b_pass_rate']:.3f}` over `{gate['b_runs']}` runs",
        f"- C pass rate: `{gate['c_pass_rate']:.3f}` over `{gate['c_runs']}` runs",
        f"- A-B gap: `{gate['a_b_gap']:.3f}`",
        "",
        gate["coarse_screen_note"],
        "",
        "## Family Pass Rates",
        "",
        "| Family | Mode | Runs | Passes | Pass rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in gate["family_rows"]:
        lines.append(f"| {row['family']} | {row['mode']} | {row['runs']} | {row['passes']} | {row['pass_rate']:.3f} |")
    if audit["failures"]:
        lines.extend(["", "## Validation Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
    lines.append("")
    if gate["valid_text_baseline"]:
        lines.append("Decision: Tier 2 text baseline is valid as a coarse screen; prepare the approved full 8B horizon sweep plan with C repair ablation.")
    else:
        lines.append("Decision: stop before the full sweep and diagnose A-pass/B-fail artifacts.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
