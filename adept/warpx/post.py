"""Post-processing for WarpX runs — M1 scope.

After ``BaseWarpX.__call__`` finishes, ``collect`` copies the provenance
files (rendered ``inputs``, ``warpx_used_inputs``, stdout/stderr) and the
step-cadence reduced-diagnostic text files into the adept temp dir so MLflow
uploads them, and returns scalar metrics. The openPMD→NetCDF layer, units
conversion, and canned plots land in M2 (see dev_docs/warpx-wrapper-plan.md).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

# WarpX per-step progress lines look like "STEP 100 ends. TIME = ..." — the
# last one gives the final completed step even if the run died early.
_STEP_RE = re.compile(r"^STEP (\d+) ends", re.MULTILINE)


def _final_step(stdout_log: Path) -> int:
    if not stdout_log.is_file():
        return -1
    best = -1
    for m in _STEP_RE.finditer(stdout_log.read_text(errors="replace")):
        best = max(best, int(m.group(1)))
    return best


def collect(run_output: dict, cfg: dict, td: str) -> dict[str, Any]:
    """Post-process a finished WarpX run.

    Side effects: copies ``inputs``, ``warpx_used_inputs``, ``stdout.log``,
    ``stderr.log``, and every reduced-diagnostic ``.txt`` under
    ``diags/reducedfiles/`` into ``td`` for MLflow upload. Returns
    ``{"metrics": {...}}`` for adept to log.
    """
    solver = run_output["solver result"]
    run_dir = Path(solver["run_dir"])
    td = Path(td)

    metrics: dict[str, float] = {
        "wall_time_s": float(solver["wall_time"]),
        "exit_code": float(solver["exit_code"]),
        "final_step": float(_final_step(run_dir / "stdout.log")),
    }

    for fname in ("inputs", "warpx_used_inputs", "stdout.log", "stderr.log"):
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, td / fname)

    reduced = run_dir / "diags" / "reducedfiles"
    if reduced.is_dir():
        out = td / "reducedfiles"
        out.mkdir(exist_ok=True)
        for p in sorted(reduced.glob("*.txt")):
            shutil.copy(p, out / p.name)

    return {"metrics": metrics}
