"""Checkpoint / restart (LPSE --restart): a run split in two reproduces the unbroken run.

Tolerances (fixed before the tests were first run): step counts exact; resumed vs unbroken fields and
energies rtol 1e-9 (the two runs take the same steps with the same noise keys, so they agree to round-off).
"""

import math
from copy import deepcopy

import numpy as np
import pytest
import yaml


def _cfg(tmax_fs, checkpoint=None, restart=None, run="restart-test", dt_fs=1):
    with open("tests/test_lpse2d_cpp/configs/epw.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {"tmax": f"{tmax_fs}fs", "dt": f"{dt_fs}fs", "xmax": "8um", "dx": "0.1um", "ymax": "0.2um", "ymin": "-0.2um"}
    )
    cfg["save"]["fields"]["t"].update({"tmin": "0fs", "tmax": f"{tmax_fs}fs", "dt": "5fs"})
    cfg["terms"]["epw"]["source"]["noise"] = True
    cfg["terms"]["epw"]["source"]["noise_seed"] = 7
    cfg["mlflow"]["run"] = run
    if checkpoint is not None:
        cfg["save"]["checkpoint"] = checkpoint
    if restart is not None:
        cfg["restart"] = {"file": restart}
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    _, ppo, _ = exo(modules)
    return ppo


def test_split_run_matches_unbroken_run(tmp_path):
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_cfg(10, checkpoint=ckpt, run="restart-first"))
    saved = np.load(ckpt)
    t_ckpt = float(saved["t"])
    assert 0.0099 < t_ckpt < 0.0121 and "epw" in saved.files  # the solver's end time (tmax rounded to steps)

    resumed = _run(_cfg(20, restart=ckpt, run="restart-second"))
    unbroken = _run(_cfg(20, run="restart-unbroken"))

    t_res = resumed["series"]["t (ps)"].values
    assert t_res[0] >= t_ckpt - 1e-12  # the series starts at the restart time
    for name in ("phi", "ex"):
        a = resumed["x"][name].isel({"t (ps)": -1}).values
        b = unbroken["x"][name].isel({"t (ps)": -1}).values
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))
    e_res = float(resumed["series"]["epw_energy"].values[-1])
    e_unb = float(unbroken["series"]["epw_energy"].values[-1])
    assert e_unb > 0.0
    np.testing.assert_allclose(e_res, e_unb, rtol=1e-9)


def test_restart_past_the_last_scheduled_field_save(tmp_path):
    """restart_t0 < save.fields.t.tmax < next scheduled save: the resumed segment saves no fields but
    runs, keeps the cadence (no off-schedule save) and reproduces the unbroken run's series."""
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_cfg(12, checkpoint=ckpt, run="restart-exhausted-first"))  # fields saved at 0, 5, 10 fs
    t_ckpt = float(np.load(ckpt)["t"])  # the solver's end time, tmax rounded up to steps
    next_save = 0.005 * np.ceil(t_ckpt / 0.005 - 1e-9)
    assert t_ckpt < 0.014 < next_save  # restart_t0 < save.fields.t.tmax < next scheduled save

    resumed = _run(_cfg(14, restart=ckpt, run="restart-exhausted-second"))
    unbroken = _run(_cfg(14, run="restart-exhausted-unbroken"))

    assert resumed["x"] is None and resumed["k"] is None
    np.testing.assert_allclose(unbroken["x"]["phi"].coords["t (ps)"].values, [0.0, 0.005, 0.010], atol=1e-12)
    t_res = resumed["series"]["t (ps)"].values
    t_unb = unbroken["series"]["t (ps)"].values
    assert t_res[0] >= t_ckpt - 1e-12 and abs(t_res[-1] - t_unb[-1]) < 1e-12
    e_res = float(resumed["series"]["epw_energy"].values[-1])
    e_unb = float(unbroken["series"]["epw_energy"].values[-1])
    assert e_unb > 0.0
    np.testing.assert_allclose(e_res, e_unb, rtol=1e-9)


def _diffrax_step_count(t0, t1, dt0):
    """The number of steps diffrax's ConstantStepSize takes over [t0, t1] (vectorised over t0, dt0)."""
    import diffrax
    import jax

    jax.config.update("jax_enable_x64", True)

    def count(a, h):
        return diffrax.ConstantStepSize().init(None, a, t1, None, h, None, None, None)[1][1]

    return np.asarray(jax.vmap(count)(np.asarray(t0, dtype=np.float64), np.asarray(dt0, dtype=np.float64)))


@pytest.mark.parametrize("dt_fs, nt", [(1.0, 23291), (0.6, 4000), (1.0 / 3.0, 3000)])
def test_resumed_segment_takes_whole_steps_from_every_restart_point(dt_fs, nt):
    """diffrax takes ceil((t1 - t0) / nextafter(dt0)) steps; with dt0 = grid.dt a segment resumed at
    t0 = n0 dt can take n + 1 steps (adept's Stepper then overruns tmax by a step). solver_dt0 makes the
    count nt - n0 from every restart step."""
    from adept._lpse2d_cpp.helpers import solver_dt0

    dt = dt_fs * 1.0e-3
    t1 = dt * nt  # grid.tmax = dt * nt
    n0 = np.arange(nt)
    t0 = n0 * dt
    expected = nt - n0
    old = _diffrax_step_count(t0, t1, np.full(nt, dt))
    assert np.count_nonzero(old != expected) > 0  # the case is sensitive: dt0 = dt miscounts somewhere
    new = _diffrax_step_count(t0, t1, [solver_dt0(a, t1, dt) for a in t0])
    np.testing.assert_array_equal(new, expected)


@pytest.mark.parametrize(
    "tmax_first_fs, tmax_fs",
    [
        (6, 8),  # resumed at step 11 of 14: dt0 = dt took 4 steps instead of 3
        (6, 12),  # resumed at step 11 of 21: truncated t / dt reused noise keys, differently in the two runs
    ],
)
def test_split_run_matches_unbroken_run_at_awkward_step_times(tmp_path, tmax_first_fs, tmax_fs):
    """At dt = 0.6 fs the solver's step times sit off n dt by round-off, which the step count and the
    per-step noise key must not see."""
    dt_fs = 0.6
    ckpt = str(tmp_path / "ckpt.npz")
    _run(_cfg(tmax_first_fs, checkpoint=ckpt, run="restart-awkward-first", dt_fs=dt_fs))
    resumed = _run(_cfg(tmax_fs, restart=ckpt, run="restart-awkward-second", dt_fs=dt_fs))
    unbroken = _run(_cfg(tmax_fs, run="restart-awkward-unbroken", dt_fs=dt_fs))

    nt = int(tmax_fs / dt_fs + 1)
    assert math.isclose(float(unbroken["series"]["t (ps)"].values[-1]), nt * dt_fs * 1.0e-3, rel_tol=1e-12)
    e_res = float(resumed["series"]["epw_energy"].values[-1])
    e_unb = float(unbroken["series"]["epw_energy"].values[-1])
    assert e_unb > 0.0
    np.testing.assert_allclose(e_res, e_unb, rtol=1e-9)
    if resumed["x"] is not None:
        a = resumed["x"]["phi"].isel({"t (ps)": -1}).values
        b = unbroken["x"]["phi"].isel({"t (ps)": -1}).values
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * np.max(np.abs(b)))
