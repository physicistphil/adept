"""Translation of original-LPSE decks and reading of LPSE outputs (lpse_deck.py).

The ZAK-unit port is checked against the derived quantities LPSE prints in its own
parameter summary for the test_025 deck; the deck parser and translator are checked on
the shipped example decks; the output readers on a reference run when one is present.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from adept._lpse2d_cpp.parity import lpse_root, reference_run_dir

# ``$LPSE_ROOT`` or the ``original-lpse`` checkout beside this repository (or its worktree
# parents); the shipped decks need the checkout, the reference-run tests fall back to the
# ``lpse_reference/`` artifact tree on MLflow ``lpse-parity`` when the local run is absent.
LPSE_ROOT = lpse_root() or Path("original-lpse-not-found")
DECKS = LPSE_ROOT / "examples" / "testRuns"


def _finish(cfg):
    from adept._lpse2d_cpp.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    return cfg


def test_zak_units_match_lpse_printed_summary():
    """test_025: Z = 1, Te = 2 keV, Ti = 1 keV, Mi/Me = 1836, n0/nc = 0.2, 0.351 um. LPSE prints
    critical density 9.0491e+21 1/cm^3, envelope density 1.8098e+21, plasma frequency
    2.4000e+15 rad/s, Debye length 7.8148e-07 cm (runs/test_025/lpse.out)."""
    from adept._lpse2d_cpp.lpse_deck import ZakUnits

    zak = ZakUnits(2.0, 1.0, 1.0, 1836.0, 0.2, 0.351)
    assert zak.nc == pytest.approx(9.0491e21, rel=2e-4)
    assert zak.n0 == pytest.approx(1.8098e21, rel=2e-4)
    assert zak.plasma_frequency == pytest.approx(2.4000e15, rel=2e-4)
    assert zak.debye_length_cm == pytest.approx(7.8148e-7, rel=2e-4)
    # the dispersion coefficient 3 ve^2 / (2 wpe) is exactly 1 in ZAK units
    assert 3.0 * zak.ve**2 / (2.0 * zak.wpe) == pytest.approx(1.0)
    # ZAK k unit vs the Debye wavenumber: kde = wpe / ve
    assert zak.kde == pytest.approx(zak.wpe / zak.ve)
    assert zak.zak_per_ps > 0 and zak.zak_per_um > 0 and np.isfinite(zak.potential_zak_to_adept)


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_parse_deck_with_include():
    from adept._lpse2d_cpp.lpse_deck import parse_parms

    parms = parse_parms(DECKS / "test_006" / "lpse.parms")
    assert parms["lw.solver"] == "spectral" and parms["raman.solver"] == "spectral"
    assert parms["laser.1.intensity"] == "1e+15"  # from laser_include.txt
    assert parms["grid.nodes"] == "360 360" and parms["densityProfile.NminLocation"] == "-10 0"


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_translate_test_006_builds_a_runnable_config():
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    parms = parse_parms(DECKS / "test_006" / "lpse.parms")
    cfg, report = translate_parms(parms, run="test_006")
    assert cfg["terms"]["epw"]["source"] == {
        "noise": True,
        "noise_model": "thermal",
        "noise_debye_factor": False,
        "noise_calibrate": False,
        "noise_amplitude": pytest.approx(cfg["terms"]["epw"]["source"]["noise_amplitude"]),
        "noise_seed": 1,
        "tpd": False,
        "tpd_form": "lpse",
        "srs": True,
        "srs_k_filter": False,  # LPSE lw.kFilter is off by default
        "srs_k_filter_scale": 1.2,
    }
    assert cfg["terms"]["light"]["solver"] == "spectral" and cfg["terms"]["light"]["pump_depletion"] is False
    assert cfg["terms"]["light"]["max_wavenumber"] == 1.0
    assert cfg["terms"]["epw"]["damping"]["landau_form"] == "relativistic"
    assert cfg["terms"]["epw"]["damping"]["collisions"] == 0.1
    # EPW dt = min(lw.spectral.dt, maxLightStepsPerStep * raman.dt) = min(0.002, 4 * 0.0001)
    assert cfg["grid"]["dt"] == "0.0004ps" and cfg["grid"]["light_substeps"] == 4
    assert cfg["grid"]["dealias"] == "rectangular" and cfg["grid"]["low_pass_filter"] == pytest.approx(0.666)
    assert cfg["units"]["laser intensity"] == "1e+15W/cm^2"
    assert not report["unsupported"]

    cfg = _finish(cfg)
    grid = cfg["grid"]
    assert grid["nx"] == 360 and grid["ny"] == 360
    assert grid["dx"] == pytest.approx(20.0 / 359.0)
    n = np.asarray(grid["background_density"])[:, 0]
    # LPSE's node i sits at i h from the N_min location (x = 0) to N_max (x = 20 um, node 359); index i
    # carries that node's value (A33), linear from 0.2 to 0.28
    x_nodes = np.asarray(grid["x"]) - 0.5 * grid["dx"]
    np.testing.assert_allclose(n, 0.2 + 0.08 * x_nodes / 20.0, rtol=1e-12)
    assert n.min() >= 0.2 - 1e-12 and n.max() <= 0.28 + 1e-12
    # rectangular anti-aliasing, LPSE's index rounding (A28): 360 points at range 0.334 keep |m| <= 120
    band = np.asarray(grid["low_pass_filter_grid"])
    m = np.abs(np.fft.fftfreq(360, d=1.0 / 360)).astype(int)
    expected = (m[:, None] <= 120) & (m[None, :] <= 120)
    np.testing.assert_array_equal(band > 0, expected)


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_translate_combined_iaw_deck():
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    cfg, report = translate_parms(parse_parms(DECKS / "test_032" / "lpse.parms"), run="test_032")
    assert cfg["terms"]["epw"]["solver"] == "combined"
    assert cfg["terms"]["epw"]["source"]["tpd"] and cfg["terms"]["epw"]["source"]["srs"]
    assert cfg["terms"]["iaw"]["active"] and cfg["terms"]["iaw"]["solver"] == "spectral"
    assert cfg["terms"]["light"]["solver"] == "spectral"
    cfg = _finish(cfg)
    assert cfg["terms"]["iaw"]["damping"]["landau"] > 0.0


def test_translator_disabled_laser_is_no_pump():
    """LPSE laser.enable = false switches the E0 light off whatever beams the deck keeps (PR #372
    review): the translation carries no intensity, no beams and no depletion even with both
    pumpDepletion flags on, and the static pump it sets up is exactly zero. Exact checks."""
    from adept._lpse2d_cpp.core.laser import Light
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "lw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.solver": "static",
        "laser.nBeams": "1",
        "laser.1.intensity": "1e15",
        "laser.1.frequencyShift": "0.01",
        "laser.pumpDepletion.TPD.enable": "true",
        "laser.pumpDepletion.SRS.enable": "true",
    }
    on, _ = translate_parms({**base, "laser.enable": "true"}, run="y")
    assert on["units"]["laser intensity"] == "1e+15W/cm^2" and on["terms"]["light"]["pump_depletion"] is True

    for parms in ({**base, "laser.enable": "false"}, base):  # LPSE's default is false
        cfg, report = translate_parms(parms, run="y")
        assert cfg["units"]["laser intensity"] == "0W/cm^2"
        assert "beams" not in cfg["drivers"]["E0"]
        assert cfg["terms"]["light"]["pump_depletion"] is False
        assert any("laser.enable = false" in note for note in report["notes"])

        cfg = _finish(cfg)
        ny = cfg["grid"]["ny"]
        wave = {"delta_omega": np.array([0.0]), "intensities": np.ones((1, ny)), "phases": np.zeros((1, ny))}
        assert np.max(np.abs(np.asarray(Light(cfg).laser_update(0.05, None, wave)))) == 0.0

    with pytest.raises(ValueError, match=r"thermalFil\.laser\.enable"):
        translate_parms({**base, "iaw.enable": "true", "thermalFil.laser.enable": "true"}, run="y")


def test_translator_runs_lpse_square_cells():
    """LPSE runs square cells h = min(size / (nodes - 1)) over the deck's axes, rescales the box to
    h (nodes - 1) and refuses a mismatch above 10 % of h (ParameterManager.cpp:1258-1281; PR #372
    review). Exact checks: a deck finer in y runs at h = dy on both axes (the old translation used
    dx), a deck coarser in y keeps h = dx with LPSE's rescaled y box, each with a report note, and a
    100 % mismatch raises."""
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    base = {
        "laser.enable": "true",
        "lw.enable": "true",
        "lw.spectral.dt": "0.005",
        "simulation.time.end": "1",
        "laser.nBeams": "1",
        "laser.1.intensity": "1e15",
        "grid.nodes": "201 50",
    }
    # the translated grid (adept's own setup may still round nx to an FFT-friendly size)
    for size_y, h in (("4.851", 0.099), ("5.049", 0.1)):  # dy = 0.099 < dx = 0.1, dy = 0.10304 > dx
        cfg, report = translate_parms({**base, "grid.sizes": f"20 {size_y}"}, run="y")
        assert any("square cells" in note for note in report["notes"])
        grid = cfg["grid"]
        assert float(grid["dx"].removesuffix("um")) == pytest.approx(h, rel=1e-12)
        assert float(grid["xmax"].removesuffix("um")) == pytest.approx(201 * h, rel=1e-12)
        assert float(grid["ymax"].removesuffix("um")) == pytest.approx(0.5 * 50 * h, rel=1e-12)
        assert float(grid["ymin"].removesuffix("um")) == pytest.approx(-0.5 * 50 * h, rel=1e-12)
    cfg, report = translate_parms({**base, "grid.sizes": "20 4.9"}, run="y")  # dy = dx: no note
    assert not any("square cells" in note for note in report["notes"])
    with pytest.raises(ValueError, match="square-cell"):
        translate_parms({**base, "grid.sizes": "20 9.8"}, run="y")  # dy = 2 dx


def test_lpse_exp_profile_and_thermal_noise_without_debye_factor():
    import yaml

    from adept._lpse2d_cpp.core.epw import noise_kick_spectrum

    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["density"] = {"basis": "lpse-exp", "min": 0.2, "max": 0.28, "min_location": "2um", "max_location": "8um"}
    cfg["grid"].update({"xmax": "10um", "dx": "0.1um", "ymax": "0.2um", "ymin": "-0.2um", "tmax": "10fs", "dt": "1fs"})
    cfg["terms"]["epw"]["source"].update(
        {"noise": True, "noise_model": "thermal", "noise_debye_factor": False, "tpd": False}
    )
    cfg["terms"]["epw"]["damping"]["collisions"] = 1.0
    cfg = _finish(cfg)
    n = np.asarray(cfg["grid"]["background_density"])[:, 0]
    x = np.asarray(cfg["grid"]["x"])
    ln = 6.0 / np.log(0.28 / 0.2)
    inside = (x > 2.0) & (x < 8.0)
    np.testing.assert_allclose(n[inside], 0.28 * np.exp(-(8.0 - x[inside]) / ln), rtol=1e-12)
    assert n[x < 2.0].max() == pytest.approx(0.2) and n[x > 8.0].min() == pytest.approx(0.28)

    kick = np.asarray(noise_kick_spectrum(cfg))
    cfg["terms"]["epw"]["source"]["noise_debye_factor"] = True
    kick_debye = np.asarray(noise_kick_spectrum(cfg))
    derived = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    lam_sq = derived["vte_sq"] / derived["wp0"] ** 2
    band = kick > 0
    np.testing.assert_allclose(kick[band] / kick_debye[band], np.sqrt(1.0 + (kx**2 + ky**2) * lam_sq)[band], rtol=1e-12)


_RUN_025 = reference_run_dir("test_025")
RUN_025 = _RUN_025 / "data" if _RUN_025 is not None else Path("test_025-not-found")


@pytest.mark.skipif(not RUN_025.exists(), reason="no LPSE reference run present (local or MLflow)")
def test_read_lpse_metrics_and_frames():
    from adept._lpse2d_cpp.lpse_deck import read_frames, read_metrics

    metrics = read_metrics(RUN_025 / "lpse.metrics")
    assert metrics["time"][0] == pytest.approx(0.01) and metrics["time"][-1] >= 1.0  # one sample past the end
    assert "EPW_energy" in metrics and np.all(metrics["EPW_energy"] > 0)
    frames = read_frames(RUN_025 / "lpse.pots")
    assert len(frames) == 2
    header, pots = frames[0]
    assert header["Nx"] == 720 and header["Ny"] == 90 and header["time"] == pytest.approx(0.5)
    assert pots.shape == (720, 90) and np.iscomplexobj(pots) and np.all(np.isfinite(pots))
    assert frames[1][0]["time"] == pytest.approx(1.0)
    # the x-space field is smooth along x on the scale of one cell but not constant
    assert np.abs(pots).std() > 0


@pytest.mark.skipif(not RUN_025.exists(), reason="no LPSE reference run present (local or MLflow)")
def test_read_frames_layout_is_c_order_x_then_y():
    """test_025 (40 x 5 um, 720 x 90 nodes, ``grid.downSampleFactors = 1 4``): LPSE writes
    ``(Nx, Ny)`` in C order (y fastest). The ``downSample_4`` frame is the full frame
    subsampled by 4 along both axes under that layout and nothing recognisable under the
    x-fastest one; the 2 um ``lw.Labc`` skirts along x (``Labc.y = 0``) lower ``<|phi|>`` in
    the outer 36 cells of axis 0."""
    from adept._lpse2d_cpp.lpse_deck import read_frames

    (header, pots), _ = read_frames(RUN_025 / "lpse.pots")
    (dheader, down), _ = read_frames(RUN_025 / "lpse.pots.downSample_4")
    assert (dheader["Nx"], dheader["Ny"]) == (180, 23) and down.shape == (180, 23)
    sub = pots[::4, ::4][:180, :23]
    np.testing.assert_allclose(down, sub, rtol=1e-6, atol=1e-6 * np.abs(pots).max())
    # under the wrong (transposed) layout the same subsample does not match at all
    wrong = pots.ravel().reshape((90, 720)).T
    overlap = abs(np.vdot(down.ravel(), wrong[::4, ::4][:180, :23].ravel())) / (
        np.linalg.norm(down) * np.linalg.norm(sub)
    )
    assert overlap < 0.6
    profile = np.abs(pots).mean(axis=1)
    edges = np.concatenate([profile[:36], profile[-36:]]).mean()
    assert edges < 0.9 * profile[100:-100].mean()
    # no absorber along y: the y-profile is flat to the noise level
    yprof = np.abs(pots).mean(axis=0)
    assert yprof.std() / yprof.mean() < 0.1


# ------------------------------------------------ LPSE thermal-noise constant (plan 2 N.3) --


def test_lpse_noise_constant_matches_the_value_lpse_prints():
    """test_010 (Te 2, Ti 1, Z 1, Mi/Me 1836, n_env 0.25, 0.351 um; 20 x 10 um on 360 x 180
    nodes, so h = 20/359 um) prints ``calcNoiseAmp_K0: 1.7885e+00`` (runs/test_010/lpse.out)."""
    from adept._lpse2d_cpp.lpse_deck import ZakUnits

    zak = ZakUnits(2.0, 1.0, 1.0, 1836.0, 0.25, 0.351)
    assert zak.noise_amp_k0(20.0 / 359.0, 360, 180) == pytest.approx(1.7885, rel=1e-4)
    # the normalization LPSE applies to its saved potential frames: e phi / (m_e c^2) per ZAK
    assert 0.0 < zak.potential_normalization_factor < 1.0


def _noise_cfg(calibrate, amplitude=1.0):
    import yaml

    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    # test_010's plasma; a 64 x 32 box of square 0.1 um cells
    cfg["units"].update(
        {
            "atomic number": 1836.0 / 1836.15,
            "ionization state": 1,
            "envelope density": 0.25,
            "reference electron temperature": "2keV",
            "reference ion temperature": "1keV",
        }
    )
    cfg["density"] = {"basis": "uniform", "val": 0.25}
    cfg["grid"].update({"xmax": "6.4um", "dx": "0.1um", "ymax": "1.6um", "ymin": "-1.6um", "tmax": "10fs", "dt": "2fs"})
    cfg["terms"]["epw"]["source"].update(
        {
            "noise": True,
            "noise_model": "thermal",
            "noise_calibrate": calibrate,
            "noise_amplitude": amplitude,
            "tpd": False,
        }
    )
    cfg["terms"]["epw"]["damping"]["collisions"] = 1.0
    return _finish(cfg)


def test_noise_calibrate_lpse_reproduces_lpse_kick_expression_mode_by_mode():
    """``noise_calibrate: lpse`` must give, per retained mode, LPSE's k-space kick
    ``A calcNoiseAmp_K0 / sqrt(1 + K^2 Lambda_d^2) sqrt(1 - exp(-2 gamma dt)) / |K|`` (ZAK, K in
    1/zak) converted to this code's k-space potential (``potential_zak_to_adept``), and differ
    from the equipartition calibration by a k-independent factor."""
    from adept._lpse2d_cpp.core.epw import analytic_landau_rate, noise_kick_spectrum
    from adept._lpse2d_cpp.lpse_deck import ZakUnits

    cfg = _noise_cfg("lpse", amplitude=0.7)
    kick = np.asarray(noise_kick_spectrum(cfg))
    grid, derived = cfg["grid"], cfg["units"]["derived"]
    zak = ZakUnits.from_cfg(cfg)
    kx = np.asarray(grid["kx"])[:, None]
    ky = np.asarray(grid["ky"])[None, :]
    k_sq = kx**2 + ky**2
    retained = kick > 0
    assert retained.sum() > 100
    gamma = np.asarray(analytic_landau_rate(cfg)) + derived["nu_coll"]
    c0 = zak.noise_amp_k0(grid["dx"], grid["nx"], grid["ny"])
    big_k = np.sqrt(np.where(k_sq > 0, k_sq, 1.0)) / zak.zak_per_um  # 1/zak
    lambda_d = 1.0 / zak.kde
    lpse_kick = 0.7 * c0 / np.sqrt(1.0 + big_k**2 * lambda_d**2) * np.sqrt(-np.expm1(-2.0 * gamma * grid["dt"])) / big_k
    expected = lpse_kick * zak.potential_zak_to_adept
    np.testing.assert_allclose(kick[retained], expected[retained], rtol=1e-6)
    # LPSE's Debye length equals this code's vte/wp0 (both use v_te^2 = kT/m)
    assert lambda_d / zak.zak_per_um == pytest.approx(np.sqrt(derived["vte_sq"]) / derived["wp0"], rel=1e-4)
    # equipartition and lpse differ by a constant factor across the band
    equi = np.asarray(noise_kick_spectrum(_noise_cfg("equipartition", amplitude=0.7)))
    ratio = kick[retained] / equi[retained]
    assert ratio.std() < 1e-9 * ratio.mean()
    # noise_calibrate: true is the equipartition calibration (unchanged behaviour)
    np.testing.assert_array_equal(np.asarray(noise_kick_spectrum(_noise_cfg(True, amplitude=0.7))), equi)
    with pytest.raises(ValueError, match="noise_calibrate"):
        noise_kick_spectrum(_noise_cfg("bogus"))


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
def test_translator_maps_is_calculated_to_noise_calibrate_lpse():
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    cfg, report = translate_parms(parse_parms(DECKS / "test_010" / "lpse.parms"), run="test_010")
    source = cfg["terms"]["epw"]["source"]
    assert source["noise_calibrate"] == "lpse" and source["noise_debye_factor"] is True
    assert source["noise_amplitude"] == 1.0  # the deck's lw.noise.amplitude multiplies LPSE's constant
    assert not any("isCalculated" in n for n in report["notes"])


@pytest.mark.skipif(not RUN_025.exists(), reason="no LPSE reference run present (local or MLflow)")
def test_025_noise_floor_matches_the_fluctuation_dissipation_prediction():
    """test_025 (EPW only, ``lw.noise.amplitude = 1``, collisional rate 1/ps + Landau, no
    isCalculated): the interior ``<|phi|^2>`` of LPSE's potential frames at 0.5 and 1.0 ps
    equals adept's expectation for the translated deck, ``sum_k D_k^2 (1 - exp(-2 gamma_k t))
    / (1 - exp(-2 gamma_k dt)) / N^2`` (Parseval; the frame holds e phi / (m_e c^2)), within
    the plan's 10 % -- with no adept run. Without the frame normalization the same comparison
    reads 3e-4, without the transient 0.6 / 0.8: it is sensitive to every factor."""
    from adept._lpse2d_cpp.core.epw import analytic_landau_rate, noise_kick_spectrum
    from adept._lpse2d_cpp.lpse_deck import ZakUnits, parse_parms, read_frames, translate_parms

    cfg, _ = translate_parms(parse_parms(RUN_025.parent / "lpse.parms"), run="test_025")
    cfg = _finish(cfg)
    grid, derived = cfg["grid"], cfg["units"]["derived"]
    nx, ny, dt = grid["nx"], grid["ny"], grid["dt"]
    kick = np.asarray(noise_kick_spectrum(cfg))
    kx = np.asarray(grid["kx"])[:, None]
    ky = np.asarray(grid["ky"])[None, :]
    k_sq = kx**2 + ky**2
    gamma = np.asarray(analytic_landau_rate(cfg)) + derived["nu_coll"] * np.where(k_sq > 0, 1.0, 0.0)
    retained = kick > 0
    steady = np.where(retained, kick**2 / np.where(retained, -np.expm1(-2.0 * gamma * dt), 1.0), 0.0)
    zak = ZakUnits.from_cfg(cfg)
    x = np.asarray(grid["x"])
    interior = (x > 6.0) & (x < 34.0)  # beyond the 2 um lw.Labc skirts
    ratios = []
    for header, pots in read_frames(RUN_025 / "lpse.pots"):
        t = float(header["time"])
        phi = pots / zak.potential_normalization_factor * zak.potential_zak_to_adept
        expected = np.sum(steady * -np.expm1(-2.0 * gamma * t) * retained) / (nx * ny) ** 2
        ratios.append(np.mean(np.abs(phi[interior, :]) ** 2) / expected)
    assert len(ratios) == 2
    for r in ratios:
        assert r == pytest.approx(1.0, abs=0.10), ratios


def test_translator_reads_pulse_shape_files_and_bracketed_vectors(tmp_path):
    """LPSE's keys are laser.pulseShape.{enable, file} (a two-column t_ps / scale table,
    relative to the deck) and get_floatVector accepts ``[1,0,0]`` (test_012's direction)."""
    from adept._lpse2d_cpp.lpse_deck import _floats, parse_parms, translate_parms

    assert _floats("[1,0,0]") == [1.0, 0.0, 0.0] and _floats("0.98 0.17 0") == [0.98, 0.17, 0.0]
    deck = tmp_path / "lpse.parms"
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "pulseShape.dat").write_text("# t scale\n0 0\n0.2 1\n2 1\n")
    deck.write_text(
        "grid.sizes = 20 10;\ngrid.nodes = 200 100;\nsimulation.time.end = 1;\nlaser.enable = true;\n"
        "laser.nBeams = 1;\nlaser.1.intensity = 1e15;\nlaser.pulseShape.enable = true;\n"
        "laser.pulseShape.file = ./data/pulseShape.dat;\ninitialPerturbation.enable = true;\n"
        "initialPerturbation.field = E0_z;\ninitialPerturbation.direction = [1,0,0];\n"
    )
    parms = parse_parms(deck)
    cfg, report = translate_parms(parms, run="x")
    assert cfg["drivers"]["E0"]["pulse_file"] == str(tmp_path / "data" / "pulseShape.dat")
    assert cfg["initial_perturbation"]["direction"] == [1.0, 0.0]
    assert not any("pulse" in u for u in report["unsupported"])


def _translate_minimal(tmp_path, extra: str = ""):
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    deck = tmp_path / "lpse.parms"
    deck.write_text(
        "grid.sizes = 20 10;\ngrid.nodes = 200 100;\nsimulation.time.end = 1;\nlaser.enable = true;\n"
        "laser.nBeams = 1;\nlaser.1.intensity = 1e15;\n" + extra
    )
    return translate_parms(parse_parms(deck), run="x")


@pytest.mark.parametrize(
    "line, retained",
    [
        ("", 1.0 - 0.3334),  # key omitted: ParameterManager.cpp:241 sets 0.3334
        ("grid.antiAliasing.range = 0;\n", 1.0),
        ("grid.antiAliasing.range = 0.4;\n", 0.6),
        ("grid.antiAliasing.range = [0.25];\n", 0.75),
    ],
)
def test_translator_anti_aliasing_range_follows_lpse_default(tmp_path, line, retained):
    """LPSE anti-aliases every run unless the deck sets grid.antiAliasing.range = 0: the
    six-face range vector defaults to 0.3334 (ParameterManager.cpp:237-244)."""
    cfg, report = _translate_minimal(tmp_path, line)
    assert cfg["grid"]["dealias"] == "rectangular"
    assert cfg["grid"]["low_pass_filter"] == pytest.approx(retained)
    assert not any("antiAliasing" in u for u in report["unsupported"])


def test_translator_reports_anti_aliasing_it_cannot_represent(tmp_path):
    """Per-face ranges and a solver-specific lw./iaw. range (LwSolver.cpp:159-169) have no
    adept counterpart: one rectangular mask serves every field."""
    _, report = _translate_minimal(tmp_path, "grid.antiAliasing.range = 0.3 0.2;\n")
    assert any("per-face" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, "grid.antiAliasing.range = 0.3;\nlw.antiAliasing.range = 0.5;\n")
    assert any("lw.antiAliasing.range" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, "grid.antiAliasing.range = 0.3;\nlw.antiAliasing.range = 0.3;\n")
    assert not any("antiAliasing" in u for u in report["unsupported"])


@pytest.mark.parametrize("line, feedback", [("", False), ("hpe.landauDampingEvolution.enable = true;\n", True)])
def test_translator_maps_hpe_landau_damping_evolution_with_lpse_default(tmp_path, line, feedback):
    """ParameterManager.cpp:272-279: useLDE defaults to false -- the particles then leave the
    Landau rate Maxwellian (ElectronTracker.cu:335)."""
    cfg, _ = _translate_minimal(tmp_path, "lw.enable = true;\nlw.spectral.dt = 0.002;\nhpe.enable = true;\n" + line)
    assert cfg["terms"]["hpe"]["feedback"] is feedback


def test_translator_small_keys_follow_lpse(tmp_path):
    """LightSolver.cpp:1716 (phase in degrees), ParameterManager.cpp:487 (float Z), 610/619
    (maxBackgroundDensity up to 1000), 782-787 (lw.landauDamping off by default), 346 / LightSolver.cpp:195
    (static raman with its lw source refused), 268 (laser.enable default false)."""
    cfg, _ = _translate_minimal(
        tmp_path,
        "physical.Z = 3.5;\nlaser.1.phase = 90;\ndensityProfile.maxBackgroundDensity = 4;\nlaser.nBeams = 2;\n"
        "laser.2.intensity = 1e15;\nlaser.2.phase = 180;\n",
    )
    assert cfg["units"]["ionization state"] == 3.5
    assert [b["phase"] for b in cfg["drivers"]["E0"]["beams"]] == pytest.approx([np.pi / 2, np.pi])
    assert cfg["density"]["max_density"] == 4.0
    assert cfg["terms"]["epw"]["damping"]["landau"] is False
    with pytest.raises(ValueError, match=r"raman\.solver = static"):
        _translate_minimal(tmp_path, "raman.enable = true;\n")


@pytest.mark.skipif(not DECKS.exists(), reason="original-lpse example decks not available")
@pytest.mark.parametrize("deck", ["test_036", "test_082"])
def test_translated_grid_keeps_lpse_node_counts(deck):
    """LPSE runs on grid.nodes (ParameterManager.cpp:1285-1300); adept's 5-smooth resizing grew test_036
    504x102 -> 512x108 and test_082 528x104 -> 540x108. The translator turns it off."""
    from adept._lpse2d_cpp.helpers import get_derived_quantities, get_solver_quantities, write_units
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    parms = parse_parms(DECKS / deck / "lpse.parms")
    cfg, _ = translate_parms(parms, run=deck)
    assert cfg["grid"]["smooth_fft_size"] is False
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    grid = get_solver_quantities(cfg)
    nodes = [int(v) for v in parms["grid.nodes"].split()[:2]]
    assert (grid["nx"], grid["ny"]) == tuple(nodes)


_PRINTED_SOLVERS = {"lightSolver": "laser", "ramanSolver": "raman", "lwSolver": "lw", "iawSolver": "iaw"}
_REFERENCE_RUNS = sorted(
    p.parent.name for p in (LPSE_ROOT / "runs").glob("*/lpse.out") if (p.parent / "lpse.parms").exists()
)


def _printed_time_steps(out_path: Path) -> dict[str, float]:
    """The solver steps (fs) of LPSE's "Time step sizes" block (Lpse.cpp printSolverTimeSteps)."""
    block = out_path.read_text(errors="replace").split("Time step sizes:", 1)[1].split("}", 1)[0]
    matches = re.finditer(r"(\w+)-timeStep:\s*([0-9.eE+-]+) fs", block)
    return {_PRINTED_SOLVERS[m.group(1)]: float(m.group(2)) for m in matches if m.group(1) in _PRINTED_SOLVERS}


@pytest.mark.skipif(not _REFERENCE_RUNS, reason="original-lpse reference runs (runs/*/lpse.out) not available")
@pytest.mark.parametrize("run", _REFERENCE_RUNS)
def test_translator_reproduces_lpse_time_steps(run):
    """A12: the translator's port of Lpse::computeMicroTimestep / setupSolverTimeSteps and the solvers'
    Tstep gives the steps each reference run printed (rtol 2e-3, fixed in advance: the printout keeps 4-6
    significant figures), and adept's schedule built from them (grid.dt, light sub-steps, IAW stride)
    passes adept's own light-stability check with the same steps."""
    from adept._lpse2d_cpp.helpers import get_derived_quantities, get_solver_quantities, write_units
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    run_dir = LPSE_ROOT / "runs" / run
    printed = _printed_time_steps(run_dir / "lpse.out")
    cfg, report = translate_parms(parse_parms(run_dir / "lpse.parms"), run=run)
    got = report["lpse_time_steps_fs"]
    assert set(got) == set(printed)
    for cls, step in printed.items():
        assert got[cls] == pytest.approx(step, rel=2e-3), cls
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    grid = get_solver_quantities(cfg)
    dt_fs = grid["dt"] * 1e3
    assert any(dt_fs == pytest.approx(s, rel=2e-3) for s in printed.values())
    if cfg["grid"].get("light_substeps"):
        light = [printed[c] for c in ("laser", "raman") if c in printed]
        assert dt_fs / grid["light_substeps"] == pytest.approx(min(light), rel=2e-3)
    if "iaw" in printed and (cfg["terms"].get("iaw") or {}).get("stride", 1) > 1:
        assert dt_fs * cfg["terms"]["iaw"]["stride"] == pytest.approx(printed["iaw"], rel=2e-3)


def test_translator_maps_interpolate_sources_in_time(tmp_path):
    """ParameterManager.cpp:330, 367, 756: {laser|raman|lw}.interpolateSourcesInTime default true; the
    evolved Raman light's flag sets terms.light.interpolate_sources, a differing pump flag is noted."""
    base = "lw.enable = true;\nlw.spectral.dt = 0.002;\nraman.enable = true;\nraman.solver = spectral;\n"
    cfg, _ = _translate_minimal(tmp_path, base)
    assert cfg["terms"]["light"]["interpolate_sources"] is True
    assert cfg["terms"]["epw"]["interpolate_sources"] is True
    cfg, report = _translate_minimal(
        tmp_path,
        base
        + "laser.solver = spectral;\nraman.interpolateSourcesInTime = false;\n"
        + "lw.interpolateSourcesInTime = false;\n",
    )
    assert cfg["terms"]["light"]["interpolate_sources"] is False
    assert cfg["terms"]["epw"]["interpolate_sources"] is False
    assert any("interpolateSourcesInTime differs" in n for n in report["notes"])


@pytest.mark.parametrize(
    "n, aa, kept",
    [
        # worked by hand from Lpse::setupAntiAliasingRange: x0 = int(360 * 0.333) = 119 -> 118 -> +2 = 120,
        # x1 = int(360 * 0.667) = 240; zeroed 120 < l < 240
        (360, 0.334, list(range(0, 121)) + list(range(-120, 0))),
        # odd N: x0 = 119 -> 118 -> 120, x1 = int(239.45) = 239 -> 238; zeroed 121..237 -> m = -121 kept
        (359, 0.334, list(range(0, 121)) + list(range(-121, 0))),
        # range 0.5 on 64: x0 = 16, x1 = 48; 17 below, 16 above -> no step; zeroed 17..47
        (64, 0.5, list(range(0, 17)) + list(range(-16, 0))),
    ],
)
def test_rectangular_dealias_keeps_lpse_index_range(n, aa, kept):
    """Inventory A28: LPSE's anti-aliasing indices (float32 products, even rounding, symmetry step;
    insideAntiAliasRegion x0 < l < x1) keep one or two more modes than |k| < (1 - aa) k_Nyquist. The
    kept sets are exact, worked by hand from the C++ above (no tolerance)."""
    from adept._lpse2d_cpp.helpers import lpse_antialias_retained

    m = np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(int)
    np.testing.assert_array_equal(np.sort(m[lpse_antialias_retained(n, aa)]), np.sort(kept))


def _lpse_zak_per_ps(te=2.0, ti=1.0, z=1.0, mi_over_me=1836.0, n_env=0.25, wavelength_um=0.351):
    """LPSE's zakUnitsPerPicosecond, transcribed from ParameterManager.cpp:1137-1163 independently of
    ``ZakUnits``: ``1e-12 plasmaFrequency / wpe`` with ``plasmaFrequency = w0 sqrt(n_env)`` (``No = NoOverNc
    Nc``, ``Nc = w0^2 me / (4 pi e^2)``) and the ZAK plasma frequency ``wpe = 1.5 M``,
    ``M = (Mi/Me) / (eta Z)``, ``eta = 1 + 3 Ti / (Z Te)``. 2.436 for LPSE's default plasma."""
    w0 = 2.0 * np.pi * 2.99792458e10 / (wavelength_um * 1.0e-4)
    eta = 1.0 + 3.0 * ti / (z * te)
    return 1.0e-12 * w0 * np.sqrt(n_env) / (1.5 * mi_over_me / (eta * z))


def test_translator_converts_the_landau_lower_threshold_from_zak_units(tmp_path):
    """Review 2026-11-16 B3: LPSE compares lw.landauDamping.lowerThreshold with its Landau rate in ZAK
    units, unconverted (ParameterManager.cpp:788; LwSolver.cpp:1291, ZakharovSolver.cpp:2702, :2883), so
    the deck value is a ZAK rate: 1/ps = value * zakUnitsPerPicosecond. Tolerance rel 1e-12 (the same
    double-precision expression evaluated two ways); the old pass-through gave 1.5 (a factor 2.44 low)."""
    base = "lw.enable = true;\nlw.spectral.dt = 0.002;\nlw.landauDamping.enable = true;\n"
    cfg, _ = _translate_minimal(tmp_path, base + "lw.landauDamping.lowerThreshold = 1.5;\n")
    zak_per_ps = _lpse_zak_per_ps()
    assert zak_per_ps == pytest.approx(2.436, rel=1e-3)
    assert cfg["terms"]["epw"]["damping"]["landau_lower_threshold"] == pytest.approx(1.5 * zak_per_ps, rel=1e-12)
    cfg, _ = _translate_minimal(tmp_path, base + "physical.Te = 4;\nphysical.Z = 3.5;\nphysical.MiOverMe = 11926;\n")
    assert cfg["terms"]["epw"]["damping"]["landau_lower_threshold"] == 0.0
    cfg, _ = _translate_minimal(
        tmp_path,
        base + "physical.Te = 4;\nphysical.Z = 3.5;\nphysical.MiOverMe = 11926;\nlw.envelopeDensity = 0.1;\n"
        "lw.landauDamping.lowerThreshold = 1.5;\n",
    )
    expected = 1.5 * _lpse_zak_per_ps(te=4.0, z=3.5, mi_over_me=11926.0, n_env=0.1)
    assert cfg["terms"]["epw"]["damping"]["landau_lower_threshold"] == pytest.approx(expected, rel=1e-12)


def test_translator_maps_the_iaw_noise_amplitude_from_zak_units(tmp_path):
    """Review 2026-11-16 B1: LPSE's iaw.noise.amplitude (default 1, ParameterManager.cpp:950) multiplies the
    kick ``A N sqrt(exp(2 dt gamma) - 1)`` on the velocity divergence's modes (IawSolver::addNoise), a ZAK
    rate, so adept's terms.iaw.noise_amplitude (1/ps) = A * zakUnitsPerPicosecond. The old translator did not
    write it (adept's default 1 /ps: 1/2.436 of LPSE's level here). Tolerance rel 1e-12. A non-positive
    amplitude adds no noise (IawSolver.cpp:2553); noise without IAW damping is refused (:2557) except on the
    fd path with a zero simplified rate, whose k-space step (noise included) LPSE skips (:2428)."""
    base = (
        "lw.enable = true;\nlw.spectral.dt = 0.002;\niaw.enable = true;\niaw.spectral.dt = 0.01;\n"
        "iaw.landauDampingRate = 0.1;\niaw.noise.enable = true;\n"
    )
    zak_per_ps = _lpse_zak_per_ps()
    cfg, _ = _translate_minimal(tmp_path, base)
    assert cfg["terms"]["iaw"]["noise"] is True
    assert cfg["terms"]["iaw"]["noise_amplitude"] == pytest.approx(1.0 * zak_per_ps, rel=1e-12)
    cfg, _ = _translate_minimal(tmp_path, base + "iaw.noise.amplitude = 3e-4;\n")
    assert cfg["terms"]["iaw"]["noise_amplitude"] == pytest.approx(3.0e-4 * zak_per_ps, rel=1e-12)
    cfg, _ = _translate_minimal(tmp_path, base + "iaw.noise.amplitude = 0;\n")
    assert cfg["terms"]["iaw"]["noise"] is False
    no_damping = base.replace("iaw.landauDampingRate = 0.1;\n", "")
    with pytest.raises(ValueError, match=r"iaw\.noise\.enable needs"):
        _translate_minimal(tmp_path, no_damping)
    cfg, report = _translate_minimal(tmp_path, no_damping + "iaw.solver = fd;\n")
    assert cfg["terms"]["iaw"]["noise"] is False
    assert any("adds no noise" in n for n in report["notes"])


def test_translator_keeps_each_light_class_absorber(tmp_path):
    """Review 2026-11-16 B18: {laser|raman}.evolution.abc.maxDampingRate are per class (LightSolver.cpp:1317-1319)
    and adept keeps them apart (terms.light.boundary_max_rate / raman_boundary_max_rate); the translator
    overwrote the pump's with the Raman light's. Exact values (no tolerance). The per-class abc.type, of
    which adept has one, is taken from the evolved classes and a difference is reported."""
    base = (
        "lw.enable = true;\nlw.spectral.dt = 0.002;\nlw.SRS.enable = true;\nraman.enable = true;\nraman.solver = fd;\n"
        "laser.evolution.Labc = 1;\nraman.evolution.Labc = 1;\n"
    )
    cfg, report = _translate_minimal(
        tmp_path,
        base
        + "laser.solver = fd;\nlaser.evolution.abc.maxDampingRate = 3000;\nraman.evolution.abc.maxDampingRate = 800;\n",
    )
    assert cfg["terms"]["light"]["boundary_max_rate"] == 3000.0
    assert cfg["terms"]["light"]["raman_boundary_max_rate"] == 800.0
    assert not any("abc" in u for u in report["unsupported"])
    # the absorbers built from them: the same exp profile (equal Labc) scaled by 3000 and 800 /ps, so the
    # rates are in the ratio 3.75 cell by cell (rel 1e-12; the old translator gave 1)
    grid = _finish(cfg)["grid"]
    rate0 = -np.log(np.asarray(grid["light_absorbing_boundaries"]))
    rate1 = -np.log(np.asarray(grid["raman_absorbing_boundaries"]))
    layer = rate1 > 0.0
    assert np.any(layer)
    np.testing.assert_allclose(rate0[layer] / rate1[layer], 3000.0 / 800.0, rtol=1e-12)
    # a static pump's absorber type does not vote; differing evolved types are reported
    cfg, report = _translate_minimal(
        tmp_path, base + "laser.evolution.abc.type = exp;\nraman.evolution.abc.type = pml;\n"
    )
    assert cfg["terms"]["light"]["absorber"] == "pml"
    assert not any("abc.type" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, base + "laser.solver = fd;\nraman.evolution.abc.type = pml;\n")
    assert any("abc.type differ" in u for u in report["unsupported"])


_SEED_DECK = (
    "lw.enable = true;\nlw.spectral.dt = 0.002;\nlw.SRS.enable = true;\nlaser.solver = fd;\nraman.enable = true;\n"
    "raman.solver = fd;\nraman.evolution.Labc = 1;\nraman.nBeams = 1;\nraman.1.intensity = 1e12;\n"
    "raman.1.direction = -1 0 0;\nraman.1.evolution.source = max.x;\n"
)


def test_translator_maps_the_raman_seed_profile_and_frequency_shift(tmp_path):
    """Review 2026-11-16 B19: raman.1.evolution.{width, sgOrder, offset} and raman.1.frequencyShift were dropped.
    LPSE's seed profile is exp(-(|y - offset_y| / width)^sgOrder) (SchrodingerSolver3::superGaussian, sgOrder
    default 4) and its frequency shift a fraction of the Raman carrier W1 (LightSolver.cpp:2425), adept's
    drivers.E1.yw (order 4, exp(-(y / (yw/2))^4)) and delta_omega. The injector's y profile, read from adept's
    own seed source (spectral_light.smooth_seed_source), must equal LPSE's super-Gaussian (rtol 1e-12)."""
    from adept._lpse2d_cpp.core.spectral_light import smooth_seed_source

    cfg, report = _translate_minimal(
        tmp_path,
        _SEED_DECK + "raman.1.evolution.width = 3;\nraman.1.evolution.sgOrder = 4;\nraman.1.frequencyShift = 0.02;\n",
    )
    assert not any("raman" in u for u in report["unsupported"])
    seed = cfg["drivers"]["E1"]
    assert seed["yw"] == "6.0um" and seed["delta_omega"] == 0.02
    cfg = _finish(cfg)
    args = cfg["drivers"]["E1"]["derived"]
    y = np.asarray(cfg["grid"]["y"])
    x = np.asarray(cfg["grid"]["x"])
    source = smooth_seed_source(
        0.05, args, w=1.0, c=1.0, x=x, y=y, x_inject=x[-20], wpe_sq=0.25, profile=np.ones_like(x)
    )
    profile = np.abs(np.asarray(source))[0] / np.abs(np.asarray(source))[0].max()
    lpse = np.exp(-((np.abs(y) / 3.0) ** 4))
    np.testing.assert_allclose(profile, lpse / lpse.max(), rtol=1e-12, atol=1e-300)
    assert profile.min() < 1e-3  # the profile is resolved, not flat


def test_translator_converts_the_raman_wavelength_shift_at_the_raman_wavelength(tmp_path):
    """raman.1.wavelengthShift (Angstrom) means the seed's vacuum wavelength minus the Raman light's,
    lambda_1 = lambda_0 w0 / W1 (W1 = w0 - wp0; wp0 with the combined solver): delta_omega = -dlambda / lambda_1
    to first order, as LPSE's linear form. LPSE itself divides by lambda_0 (its light wavelength is still the
    pump's when the beams are read), w0 / W1 = 2x larger at 0.25 n_c; adept keeps the physical shift and
    notes it. Tolerance rel 1e-12."""
    cfg, report = _translate_minimal(tmp_path, _SEED_DECK + "raman.1.wavelengthShift = 10;\n")
    lambda_1 = 0.351 / (1.0 - np.sqrt(0.25))
    assert cfg["drivers"]["E1"]["delta_omega"] == pytest.approx(-1.0e-3 / lambda_1, rel=1e-12)
    assert any("wavelengthShift" in n for n in report["notes"])


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("raman.nBeams = 2;\nraman.2.intensity = 1e12;\n", "raman.nBeams = 2"),
        ("raman.1.phase = 30;\n", "raman.1.phase"),
        ("raman.1.bandwidth.KAP.frequency = 0.001;\n", "KAP"),
        ("raman.1.direction = -1 0.2 0;\n", "raman.1.direction"),
        ("raman.1.evolution.width = 3;\nraman.1.evolution.sgOrder = 6;\n", "sgOrder = 6"),
        ("raman.1.evolution.width = 3;\nraman.1.evolution.offset = 0 2 0;\n", "offset"),
    ],
)
def test_translator_reports_raman_seed_keys_it_cannot_represent(tmp_path, line, fragment):
    """B19: what adept's single, leftward, order-4 seed cannot carry is reported, not dropped."""
    _, report = _translate_minimal(tmp_path, _SEED_DECK + line)
    assert any(fragment in u for u in report["unsupported"]), report["unsupported"]


def test_translator_seed_face_and_beam_format_follow_lpse(tmp_path):
    """LPSE's beam face defaults to min.x (sourceTag 1, LightSolver.cpp:1705), so a seed without
    raman.1.evolution.source is not adept's x-max seed; the abbreviated beam format (LightSolver.cpp:1460-1478),
    which the translator does not read, raises instead of translating zero intensity."""
    deck = _SEED_DECK.replace("raman.1.evolution.source = max.x;\n", "")
    _, report = _translate_minimal(tmp_path, deck)
    assert any("evolution.source" in u for u in report["unsupported"])
    deck = _SEED_DECK.replace("raman.1.intensity = 1e12;\n", "raman.intensity = [1e12];\n")
    with pytest.raises(ValueError, match="abbreviated"):
        _translate_minimal(tmp_path, deck)


_FD_IAW_DECK = (
    "lw.enable = true;\nlw.spectral.dt = 0.002;\niaw.enable = true;\niaw.solver = fd;\niaw.landauDampingRate = 0.1;\n"
    "iaw.collisionalDampingRate = 0.5;\n"
)


def test_translator_maps_the_fd_iaw_switches(tmp_path):
    """Review 2026-11-16 B6: iaw.fd.include2ndOrderTemporalCorrection (default true, IawSolver.cpp:279-280) maps to
    terms.iaw.temporal_correction; iaw.fd.laplacianOrder = 4 (:249-251), which adept's second-order fd
    Laplacian cannot represent, and an explicit iaw.interpType = linear (:285-294) are reported; LPSE's linear
    default with an fd laser is noted (adept interpolates spectrally, an intentional departure)."""
    cfg, report = _translate_minimal(tmp_path, _FD_IAW_DECK)
    assert cfg["terms"]["iaw"]["temporal_correction"] is True
    assert not any("iaw." in u for u in report["unsupported"])
    assert not any("interpType" in n for n in report["notes"])  # static laser: LPSE's spectral default
    cfg, _ = _translate_minimal(tmp_path, _FD_IAW_DECK + "iaw.fd.include2ndOrderTemporalCorrection = false;\n")
    assert cfg["terms"]["iaw"]["temporal_correction"] is False
    _, report = _translate_minimal(tmp_path, _FD_IAW_DECK + "iaw.fd.laplacianOrder = 4;\n")
    assert any("laplacianOrder = 4" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, _FD_IAW_DECK + "iaw.interpType = linear;\n")
    assert any("interpType = linear" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, _FD_IAW_DECK + "laser.solver = fd;\n")
    assert any("interpType = linear" in n and "default" in n for n in report["notes"])
    assert not any("interpType" in u for u in report["unsupported"])
    _, report = _translate_minimal(tmp_path, _FD_IAW_DECK + "laser.solver = fd;\niaw.fd.superSamples = 1;\n")
    assert not any("interpType" in n for n in report["notes"])  # no interpolation without super-sampling


def test_translator_fd_damping_enable_false_skips_the_kspace_step(tmp_path):
    """B6: iaw.fd.damping.enable = false makes LPSE skip the fd k-space step -- Landau and collisional damping,
    noise and band (IawSolver.cpp:2427); the translator writes a zero simplified rate, which throws adept's
    identical switch (FDIonAcoustic.kspace_step), and turns the noise off. The spectral solver ignores the key."""
    from adept._lpse2d_cpp.core.iaw import IonAcousticWave

    cfg, report = _translate_minimal(
        tmp_path,
        _FD_IAW_DECK
        + "iaw.fd.damping.enable = false;\niaw.fd.damping.isSimplified = false;\niaw.noise.enable = true;\n",
    )
    damping = cfg["terms"]["iaw"]["damping"]
    assert damping["landau"] == 0.0 and damping["landau_form"] == "simplified"
    assert cfg["terms"]["iaw"]["noise"] is False
    assert any("fd.damping.enable = false" in n for n in report["notes"])
    assert IonAcousticWave(_finish(cfg)).fd.kspace_step is False
    cfg, report = _translate_minimal(
        tmp_path,
        _FD_IAW_DECK.replace("iaw.solver = fd;\n", "iaw.spectral.dt = 0.01;\n") + "iaw.fd.damping.enable = false;\n",
    )
    assert cfg["terms"]["iaw"]["damping"]["landau"] == 0.1
    assert any("no effect" in n for n in report["notes"])
