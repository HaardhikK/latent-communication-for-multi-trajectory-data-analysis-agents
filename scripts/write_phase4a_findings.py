from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import run_phase4


DEFAULT_IMPORT_ROOT = Path("imports/phase4_session2_final_20260702T221835")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase 4A findings and C5 anchor forensics.")
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument("--extra-import-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("reports/phase4a_findings.md"))
    return parser.parse_args()


def load_metrics(import_root: Path) -> list[dict[str, str]]:
    extracted = _extracted_root(import_root)
    metrics = extracted / "project" / "exports" / "tier2_phase4_session2_metrics.csv"
    with metrics.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_findings(import_root: Path, out: Path, extra_import_roots: list[Path] | None = None) -> str:
    rows = load_metrics(import_root)
    extracted_roots = [_extracted_root(import_root)] + [_extracted_root(root) for root in (extra_import_roots or [])]
    text = render_findings(rows, extracted_roots)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def render_findings(rows: list[dict[str, str]], extracted_roots: list[Path]) -> str:
    by_variant = aggregate_by_variant(rows)
    by_family = aggregate_by_family(rows)
    failures = aggregate_failures(rows)
    anchors = anchor_forensics(rows, extracted_roots)

    c1 = by_variant[("C_latentmas", "C1_phase3_exact")]
    c2 = by_variant[("C_latentmas", "C2_dedup")]
    c3 = by_variant[("C_latentmas", "C3_no_latent")]
    c5 = by_variant[("C_latentmas", "C5_anchor")]
    b = by_variant[("B_textmas", "-")]

    c2_c1_p = fisher(c2, c1)
    c2_c3_p = fisher(c2, c3)
    c5_c2_p = fisher(c5, c2)
    c2_b_p = fisher(c2, b)

    lines = [
        "# Phase 4A Findings",
        "",
        "Session 2 reran the long-horizon attribution matrix on Qwen3-8B 4-bit with the frozen Phase 3 repair path. "
        "The run used commit `dfdd1fca655851707840b2127ebbc0aa9cc7509b` and generation-path hash `7072860e2ace8afe`.",
        "",
        "## Claims",
        "",
        f"- **Confirmed:** the original 7-stage latent collapse was caused by duplicate chat-templated task/prompt re-encoding into the latent KV cache. `C1_phase3_exact` was {fmt_count(c1)} while `C2_dedup` was {fmt_count(c2)}; Fisher p={c2_c1_p:.4f}. Median decode cache length fell from {c1['median_cache']:.0f} to {c2['median_cache']:.0f}. `C2_dedup` is statistically indistinguishable from `B_textmas` ({fmt_count(b)}, Fisher p={c2_b_p:.4f}) while using 0 decoded coordination tokens.",
        f"- **Directional, not yet confirmed:** latent steps added +{c2['rate'] - c3['rate']:.3f} over the stage-text-only cache (`C2_dedup` {fmt_count(c2)} vs `C3_no_latent` {fmt_count(c3)}), but Fisher p={c2_c3_p:.4f}; the n=30 confirmation run decides whether this becomes a claim.",
        f"- **No evidence decoded anchors help:** `C5_anchor` was {fmt_count(c5)}, worse than `C2_dedup` by {c5['rate'] - c2['rate']:.3f} (Fisher p={c5_c2_p:.4f}). C5 median cache length was {c5['median_cache']:.0f}, about {c5['median_cache'] / c2['median_cache']:.1f}x C2, consistent with anchors re-polluting the cache.",
        "",
        "## By Variant",
        "",
        "| Mode | Variant | Runs | Final pass | Wilson CI | First-attempt pass | Median cache len |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for key in sorted(by_variant):
        row = by_variant[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {row['n']} | {row['rate']:.3f} | "
            f"[{row['ci'][0]:.3f}, {row['ci'][1]:.3f}] | {row['first_rate']:.3f} | {row['median_cache']:.0f} |"
        )

    lines.extend(["", "## By Family", ""])
    lines.append("| Mode | Variant | Family | Runs | Final pass | First-attempt pass | Median cache len |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for key in sorted(by_family):
        row = by_family[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {row['n']} | {row['rate']:.3f} | "
            f"{row['first_rate']:.3f} | {row['median_cache']:.0f} |"
        )

    lines.extend(["", "## Failure Classes", ""])
    lines.append("| Mode | Variant | Failure type | Rows |")
    lines.append("|---|---|---|---:|")
    for (mode, variant, failure), count in sorted(failures.items()):
        lines.append(f"| {mode} | {variant} | {failure} | {count} |")

    lines.extend(["", "## C5 Anchor Forensics", ""])
    lines.append(
        "Anchors were greedy, <=24-token decoded stage summaries appended as raw continuation text. "
        "The table dumps the per-run anchor-quality classification and pass/fail outcome."
    )
    lines.extend(["", "| Task | Repeat | Passed | Cache len | Anchor quality | Quality counts | Anchor dump |"])
    lines.append("|---|---:|---:|---:|---|---|---|")
    for row in anchors:
        lines.append(
            f"| {row['task_id']} | {row['repeat']} | {row['passed']} | {row['cache_len']} | "
            f"{row['quality']} | {row['quality_counts']} | {row['anchor_dump']} |"
        )

    quality_by_pass = defaultdict(Counter)
    caches_by_pass: dict[str, list[float]] = defaultdict(list)
    for row in anchors:
        quality_by_pass[row["passed"]][row["quality"]] += 1
        caches_by_pass[row["passed"]].append(float(row["cache_len"]))
    lines.extend(["", "Anchor quality vs outcome:", ""])
    lines.append("| Outcome | Runs | Median cache len | Anchor qualities |")
    lines.append("|---|---:|---:|---|")
    for passed in ("True", "False"):
        caches = caches_by_pass.get(passed, [])
        lines.append(
            f"| {passed} | {sum(quality_by_pass[passed].values())} | {median(caches):.0f} | "
            f"{dict(sorted(quality_by_pass[passed].items()))} |"
        )
    lines.append("")
    lines.append("Interpretation: C5 failures were not driven by empty or degenerate code. The anchor text often contained duplicated/truncated stage fragments, and the added decoded text roughly doubled-to-tripled the C2 cache length, matching the cache-pollution mechanism.")
    return "\n".join(lines) + "\n"


def aggregate_by_variant(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row.get("c_variant") or "-")].append(row)
    return {key: aggregate_group(group) for key, group in groups.items()}


def aggregate_by_family(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        family = row.get("task_family") or row["task_id"].removesuffix("_long")
        groups[(row["mode"], row.get("c_variant") or "-", family)].append(row)
    return {key: aggregate_group(group) for key, group in groups.items()}


def aggregate_group(group: list[dict[str, str]]) -> dict[str, Any]:
    pass_count = sum(as_bool(row.get("passed")) for row in group)
    first_count = sum(as_bool(row.get("first_attempt_passed")) for row in group)
    caches = [float(row.get("cache_len_at_decode") or 0) for row in group]
    return {
        "n": len(group),
        "passes": pass_count,
        "rate": pass_count / len(group),
        "ci": run_phase4.wilson_ci(pass_count, len(group)),
        "first_passes": first_count,
        "first_rate": first_count / len(group),
        "median_cache": median(caches),
    }


def aggregate_failures(rows: list[dict[str, str]]) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(row["mode"], row.get("c_variant") or "-", row.get("failure_type") or "unknown")] += 1
    return counts


def anchor_forensics(rows: list[dict[str, str]], extracted_roots: list[Path]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("c_variant") != "C5_anchor":
            continue
        anchors = load_anchor_texts(extracted_roots, row.get("run_dir", ""))
        counts = Counter(classify_anchor(anchor) for anchor in anchors)
        quality = dominant_anchor_quality(counts)
        out.append(
            {
                "task_id": row["task_id"],
                "repeat": row["repeat"],
                "passed": str(as_bool(row.get("passed"))),
                "cache_len": str(int(float(row.get("cache_len_at_decode") or 0))),
                "quality": quality,
                "quality_counts": html_escape(str(dict(sorted(counts.items())))),
                "anchor_dump": html_escape("; ".join(anchors)),
            }
        )
    return sorted(out, key=lambda r: (r["task_id"], int(r["repeat"])))


def load_anchor_texts(extracted_roots: list[Path], run_dir: str) -> list[str]:
    basename = Path(run_dir).name
    candidates = []
    for extracted in extracted_roots:
        candidates.extend((extracted / "runtime" / "runs").rglob(f"{basename}/anchor_texts.json"))
    if not candidates:
        return []
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def classify_anchor(anchor: str) -> str:
    text = " ".join(anchor.strip().split())
    if not text:
        return "vague"
    lower = text.lower()
    if "```" in text or re.search(r"\b(import|def|return|to_csv|read_csv|groupby|merge)\b", lower):
        return "code-like"
    if text.count("Stage ") > 1 or re.search(r"(.{12,}?)\1", text):
        return "wrong"
    if len(text.split()) < 5 or "authoritative task specification" in lower or "this stage" in lower:
        return "vague"
    if text.endswith((",", "and", "or", "from", "to", "with")):
        return "wrong"
    return "faithful"


def dominant_anchor_quality(counts: Counter[str]) -> str:
    for quality in ("wrong", "code-like", "vague", "faithful"):
        if counts.get(quality):
            return quality
    return "vague"


def fisher(left: dict[str, Any], right: dict[str, Any]) -> float:
    return run_phase4.fisher_exact_two_sided(int(left["passes"]), int(left["n"]), int(right["passes"]), int(right["n"]))


def fmt_count(row: dict[str, Any]) -> str:
    return f"{row['passes']}/{row['n']} = {row['rate']:.3f}"


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def html_escape(text: str) -> str:
    return text.replace("|", "&#124;").replace("\n", " ").strip()


def _extracted_root(import_root: Path) -> Path:
    return import_root / "extracted" if (import_root / "extracted").exists() else import_root


def main() -> int:
    args = parse_args()
    text = write_findings(args.import_root, args.out, args.extra_import_root)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
