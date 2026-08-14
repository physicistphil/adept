"""Smoke tests for the WarpX subprocess runner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adept.warpx import runner

# Resolved from the same env vars the runner itself honors, so the live-binary
# smoke test below runs wherever WarpX is built and skips cleanly otherwise.
WARPX_BIN_1D = os.environ.get("WARPX_BIN_1D") or os.environ.get("WARPX_BIN")

DECKS_DIR = Path(__file__).parent / "decks"
SMOKE_DECK = DECKS_DIR / "warpx-1d-smoke"


def test_discover_binary_explicit_wins(tmp_path: Path) -> None:
    fake = tmp_path / "fake-warpx"
    fake.write_text("")
    out = runner.discover_binary(str(fake))
    assert out == fake.resolve()


def test_discover_binary_env_fallback(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "fake-warpx-1d"
    fake.write_text("")
    monkeypatch.setenv("WARPX_BIN_1D", str(fake))
    out = runner.discover_binary(None, dim=1)
    assert out == fake.resolve()


def test_discover_binary_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        runner.discover_binary("/no/such/path/exists", dim=1)


def test_run_warpx_missing_binary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.run_warpx(
            "max_step = 1\n",
            binary="/no/such/binary",
            mpi_ranks=1,
            run_root=tmp_path,
        )


def _write_fake_binary(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)
    return path


def test_run_warpx_clean_exit(tmp_path: Path) -> None:
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        'test -f "$1" && echo "STEP 1 ends" && exit 0',
    )
    result = runner.run_warpx("max_step = 1\n", binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert result["exit_code"] == 0
    assert result["crashed"] is False
    # The rendered inputs file is the first positional argument, in the run dir.
    assert (result["run_dir"] / runner.INPUTS_FILENAME).read_text() == "max_step = 1\n"
    assert "STEP 1 ends" in (result["run_dir"] / "stdout.log").read_text()


def test_run_warpx_crash_with_output_is_salvaged(tmp_path: Path) -> None:
    # A binary that writes a diagnostic then dies (e.g. an abort partway
    # through the run) must NOT raise: the run produced data, so the runner
    # salvages it and lets the caller post-process what was written.
    fake = _write_fake_binary(
        tmp_path / "fake-warpx",
        "mkdir -p diags/reducedfiles && echo '#step time' > diags/reducedfiles/fieldenergy.txt"
        " && echo 'amrex::Abort::0::boom' >&2 && exit 6",
    )
    result = runner.run_warpx("max_step = 1\n", binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert result["exit_code"] == 6
    assert result["crashed"] is True
    assert (result["run_dir"] / "diags" / "reducedfiles" / "fieldenergy.txt").exists()


def test_run_warpx_crash_no_output_raises(tmp_path: Path) -> None:
    # A binary that exits non-zero WITHOUT writing anything is a hard failure —
    # there is nothing to salvage, so the runner raises with the AMReX detail.
    fake = _write_fake_binary(tmp_path / "fake-warpx", "echo 'amrex::Abort::0::bad inputs' >&2 && exit 1")
    with pytest.raises(RuntimeError) as excinfo:
        runner.run_warpx("max_step = 1\n", binary=str(fake), mpi_ranks=1, run_root=tmp_path)
    assert "nothing to salvage" in str(excinfo.value)
    assert "amrex::Abort" in str(excinfo.value)


@pytest.mark.skipif(
    not (WARPX_BIN_1D and Path(WARPX_BIN_1D).exists()),
    reason="set WARPX_BIN_1D (or WARPX_BIN) to a built 1D WarpX executable to run",
)
def test_run_warpx_smoke_deck_live(tmp_path: Path) -> None:
    result = runner.run_warpx(
        SMOKE_DECK.read_text(),
        binary=WARPX_BIN_1D,
        mpi_ranks=1,
        run_root=tmp_path,
        launcher="mpirun",
    )
    assert result["exit_code"] == 0
    assert (result["run_dir"] / "warpx_used_inputs").exists()
