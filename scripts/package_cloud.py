from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".agents",
    ".codex",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "exports",
    "reports",
    "runs",
    "logs",
    "hf-cache",
    "datasets",
    "models",
    "tmp",
    ".venv",
    "imports",
}
EXCLUDE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.zip",
    "*.parquet",
    "*.jsonl",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.token",
    "*hf_token*",
    "*secret*",
    "kaggle.json",
]
INCLUDE_SUFFIXES = {".py", ".toml", ".txt", ".md", ".yaml", ".yml", ".ipynb", ".json"}
PUBLIC_MARKDOWN = {"README.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean Tier 2 cloud zip.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dist" / "latent-agent-prototype-tier2.zip"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_package_files(root: Path = PROJECT_ROOT) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if any(part.startswith(".pytest_tmp") for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        if path.suffix.lower() == ".md" and rel.as_posix() not in PUBLIC_MARKDOWN:
            continue
        rel_posix = rel.as_posix().lower()
        if any(fnmatch.fnmatch(rel_posix, pattern.lower()) for pattern in EXCLUDE_PATTERNS):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    args = parse_args()
    files = collect_package_files()
    if args.dry_run:
        for file in files:
            print(file.relative_to(PROJECT_ROOT).as_posix())
        print(f"FILES {len(files)}", file=sys.stderr)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file, Path("latent-agent-prototype") / file.relative_to(PROJECT_ROOT))
    print(f"WROTE {out} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
