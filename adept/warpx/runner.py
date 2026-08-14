"""Subprocess driver for WarpX.

WarpX takes its inputs file as the first positional argument. This module
sets up a per-run work directory, writes the rendered inputs there, invokes
the configured launcher (``srun`` by default — the team runs on
Perlmutter/Slurm; override with ``mpi_launcher: mpirun`` for a local MPI)
when ``mpi_ranks > 1`` — or runs the binary directly when ``mpi_ranks == 1``
— and captures stdout/stderr to files for later artifact upload.

Error handling differs from the OSIRIS runner on purpose: WarpX/AMReX fails
loudly (``amrex::Abort`` and failed assertions exit non-zero), so the exit
code is the primary signal and there is no exit-0 stderr fuzzing. The
salvage-if-output-exists behavior is kept: a crash after diagnostics were
written still lets post-processing run on the partial data.
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

INPUTS_FILENAME = "inputs"

# Tokens WarpX/AMReX print on aborts/assertions/signal handlers, matched
# lowercased. "### error" is WarpX's own WARPX_ABORT/ASSERT banner prefix
# ("### ERROR   : ", ablastr TextMsg); the rest come from AMReX. Only
# consulted for the failure message detail — the exit code decides
# success/failure.
_AMREX_ERR_TOKENS = ("### error", "amrex::abort", "amrex error", "assertion", "sigsegv", "sigfpe", "backtrace")


def _stream_to_file_and_buffer(stream, file_path: Path, tail: list[str], tail_max: int = 200) -> None:
    """Tee a subprocess stream to disk and a bounded in-memory tail."""
    with file_path.open("w") as fh:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            tail.append(line)
            if len(tail) > tail_max:
                del tail[: len(tail) - tail_max]


def _make_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{stamp}_{uuid.uuid4().hex[:8]}"
    rd = run_root / name
    rd.mkdir()
    return rd


def _run_produced_output(run_dir: Path) -> bool:
    """True if the run wrote any salvageable diagnostic output.

    WarpX writes everything under the run directory: openPMD/plotfile dumps
    under the diag prefixes (default ``diags/``), reduced diagnostics under
    ``diags/reducedfiles/``. Any file beyond the ones the runner itself wrote
    counts as output worth keeping.
    """
    ours = {INPUTS_FILENAME, "stdout.log", "stderr.log"}
    for p in run_dir.rglob("*"):
        if p.is_file() and p.name not in ours:
            return True
    return False


def run_warpx(
    deck_text: str,
    *,
    binary: str | Path,
    mpi_ranks: int = 1,
    run_root: str | Path = "./checkpoints",
    env: dict[str, str] | None = None,
    launcher: str = "srun",
    extra_mpi_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run WarpX and return run metadata.

    Returns a dict with keys ``run_dir`` (Path), ``exit_code`` (int),
    ``crashed`` (bool), ``wall_time`` (float, seconds), and ``cmd``
    (list[str]).

    Raises ``RuntimeError`` on a non-zero exit code that left no output
    behind; a crash *with* output on disk is logged and salvaged so the
    caller still consolidates and plots what was written.
    """
    binary = Path(binary).expanduser().resolve()
    if not binary.exists():
        raise FileNotFoundError(f"WarpX binary not found: {binary}")

    run_dir = _make_run_dir(Path(run_root).expanduser().resolve())
    (run_dir / INPUTS_FILENAME).write_text(deck_text)

    if mpi_ranks > 1:
        cmd = [launcher, "-n", str(mpi_ranks)]
        if extra_mpi_args:
            cmd.extend(extra_mpi_args)
        cmd.append(str(binary))
    else:
        cmd = [str(binary)]
    cmd.append(INPUTS_FILENAME)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_tail: list[str] = []
    stderr_tail: list[str] = []

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=run_dir,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    t_out = threading.Thread(
        target=_stream_to_file_and_buffer,
        args=(proc.stdout, stdout_path, stdout_tail),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_stream_to_file_and_buffer,
        args=(proc.stderr, stderr_path, stderr_tail),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()
    wall_time = time.time() - t0

    crashed = rc != 0
    if crashed:
        # Pull the AMReX abort/assert lines to the front of the message so the
        # cause is visible without opening the logs; fall back to the raw tail.
        err_lines = [ln for ln in (stdout_tail + stderr_tail) if any(tok in ln.lower() for tok in _AMREX_ERR_TOKENS)]
        detail = "".join(err_lines[-20:]) or "".join(stderr_tail[-50:]) or "(empty stderr)"
        failure_msg = f"WarpX exited with status {rc}.\n  cmd: {shlex.join(cmd)}\n  cwd: {run_dir}\n  detail:\n{detail}"
        if _run_produced_output(run_dir):
            print(f"[warpx] WARNING: {failure_msg}")
            print(
                "[warpx] the run produced output files despite the error above — "
                "consolidating and post-processing the (possibly partial) data "
                "anyway. Verify results carefully."
            )
        else:
            raise RuntimeError(f"{failure_msg}\n  (no output files were written — nothing to salvage.)")

    return {
        "run_dir": run_dir,
        "exit_code": rc,
        "crashed": crashed,
        "wall_time": wall_time,
        "cmd": cmd,
    }


def discover_binary(cfg_binary: str | None, *, dim: int | None = None) -> Path:
    """Resolve the WarpX binary path.

    Precedence: explicit ``cfg_binary`` > ``WARPX_BIN_<dim>D`` env var >
    ``WARPX_BIN`` env var. Returns an existing Path or raises.
    """
    candidates: list[str] = []
    if cfg_binary:
        candidates.append(cfg_binary)
    if dim is not None:
        env_key = f"WARPX_BIN_{dim}D"
        if env_key in os.environ:
            candidates.append(os.environ[env_key])
    if "WARPX_BIN" in os.environ:
        candidates.append(os.environ["WARPX_BIN"])

    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "No WarpX binary found. Set warpx.binary in the manifest or "
        "WARPX_BIN / WARPX_BIN_<dim>D in the environment. Tried: "
        f"{candidates}"
    )
