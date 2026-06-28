from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    code_path: str
    returncode: int
    stdout: str
    stderr: str
    wall_latency_s: float
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def execute_python_code(code: str, work_dir: Path, *, attempt: int, timeout_s: int = 30) -> ExecutionResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / f"attempt_{attempt}.py"
    code_path.write_text(code, encoding="utf-8")

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=str(work_dir),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {timeout_s}s"

    return ExecutionResult(
        code_path=str(code_path),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        wall_latency_s=time.perf_counter() - start,
        timed_out=timed_out,
    )
