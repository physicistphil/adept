"""Resonance absorption of the pump at the critical surface (terms.light.resonance_absorption;
LPSE laser.evolution.resonanceAbsorption) -- plan 2 L.3."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from adept._lpse2d_cpp.parity import deck_path


def _cfg(angle, ramp, ra, tmax=0.4):
    """A p-polarised super-Gaussian beam (1 um, order 4) at `angle` on a linear ramp n = 0.1 at
    2 um -> 1.9 at 10 um (n_c at 6 um, L_n = 4.4 um, (k0 L_n)^(1/3) = 4.3), FD order 4 at 6 cells per wavelength."""
    with open("tests/test_lpse2d_cpp/configs/srs.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["grid"].update(
        {
            "ymax": "2um",
            "ymin": "-2um",
            "xmax": "10um",
            "dx": "0.0625um",
            "dt": "1fs",
            "tmax": f"{tmax}ps",
            "boundary_width": "1.5um",
        }
    )
    cfg["terms"]["light"] = {"pump_depletion": True, "solver": "fd", "fd_order": 4}
    if ra is not False and ra is not None:
        cfg["terms"]["light"]["resonance_absorption"] = ra
    cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["terms"]["epw"]["source"].update({"noise": False, "srs": False, "tpd": True})
    cfg["units"]["envelope density"] = 0.25
    noise = {"max": 1e-9, "min": 1e-10, "type": "uniform"}
    if ramp:
        cfg["density"] = {
            "basis": "lpse-linear",
            "min": 0.1,
            "max": 1.9,
            "min_location": "2um",
            "max_location": "10um",
            "noise": noise,
        }
    else:
        cfg["density"] = {"basis": "uniform", "val": 0.1, "noise": noise}
    cfg["drivers"]["E0"].update({"angle": angle, "beam_width": "1.0um", "beam_sg_order": 4, "delta_omega_max": 0.0})
    cfg["save"]["fields"]["t"].update({"tmax": f"{tmax}ps", "dt": "0.1ps"})
    cfg["mlflow"]["run"] = f"ra-{angle}-{ramp}-{ra}"
    return cfg


def _run(cfg):
    from adept import ergoExo

    exo = ergoExo()
    modules = exo.setup(cfg)
    sol, _, _ = exo(modules)
    result = sol["solver result"]
    dcfg = exo.adept_module.cfg
    t = np.asarray(result.ts["default"])
    net = np.asarray(result.ys["default"]["incident_flux"])
    x = np.asarray(dcfg["grid"]["x"])
    raw = np.asarray(result.ys["fields"]["E0"])
    e0 = raw.view(np.complex64 if raw.dtype == np.float32 else np.complex128)
    band = (x > 5.5) & (x < 6.5)  # about the critical surface
    energy_nc = np.sum(np.abs(e0[:, band]) ** 2, axis=(1, 2, 3))
    return t, net, np.asarray(result.ts["fields"]), energy_nc


def test_resonance_absorption_at_the_denisov_peak_and_beyond():
    tau = 4.3 * np.sin(np.deg2rad([11.0, 25.0]))
    assert 0.75 < tau[0] < 0.9 and tau[1] > 1.7
    # the launched flux: the same beam in a uniform sub-critical box
    _, net_free, _, _ = _run(_cfg(11.0, ramp=False, ra=False))
    f_inc = np.mean(net_free[-40:])
    assert f_inc > 0.3

    t, net, tf, energy = _run(_cfg(11.0, ramp=True, ra=True))
    steady = t > 0.25
    absorbed = np.mean(net[steady]) / f_inc
    # net flux into the ramp = the absorbed fraction; the Denisov curve peaks (0.5-0.7 across
    # the cold / warm models) near tau 0.8 -- the envelope model with LPSE's warm and Landau
    # terms gives ~0.42 here
    assert 0.3 < absorbed < 0.7
    # a true steady state: the flux and the energy at n_c stop changing
    np.testing.assert_allclose(np.mean(net[t > 0.35]), np.mean(net[(t > 0.25) & (t < 0.3)]), rtol=2e-2)
    assert energy[-1] < 1.05 * energy[np.argmin(np.abs(tf - 0.2))]

    # (without RA the cold-plasma resonance at n_c -- epsilon = 0 with nothing to limit the
    # longitudinal field -- has no steady state: on the 0.05 um grid the energy there grew
    # 6x over 0.1-0.4 ps; the growth is grid-dependent, so it is not asserted here)

    # far past the peak (tau ~ 1.8) the absorption is small; the launched x-flux of the 25 deg
    # beam is the 11 deg one times cos(25)/cos(11) (measured 0.922 vs 0.923)
    t25, net25, _, _ = _run(_cfg(25.0, ramp=True, ra=True))
    absorbed25 = np.mean(net25[t25 > 0.25]) / (f_inc * np.cos(np.deg2rad(25.0)) / np.cos(np.deg2rad(11.0)))
    assert absorbed25 < 0.15


def test_validation_and_translator():
    from adept._lpse2d_cpp.helpers import get_derived_quantities, write_units

    cfg = _cfg(11.0, ramp=True, ra=True)
    cfg["terms"]["light"]["solver"] = "spectral"
    write_units(cfg)
    with pytest.raises(ValueError, match="fd light solver"):
        get_derived_quantities(cfg)

    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms

    if deck_path("test_011") is None:
        pytest.skip("no LPSE decks present")
    cfg, report = translate_parms(parse_parms(deck_path("test_011")), run="test_011")
    ra = cfg["terms"]["light"]["resonance_absorption"]
    assert cfg["terms"]["light"]["solver"] == "fd" and ra["landau_update"] == 10 and ra["filter"] is True
    assert (
        cfg["drivers"]["E0"]["beam_width"] == f"{3.0 / np.sqrt(2.0)}um" and cfg["drivers"]["E0"]["beam_sg_order"] == 4.0
    )
    assert abs(cfg["drivers"]["E0"]["angle"] - np.degrees(np.arctan2(0.1676, 0.98586))) < 1e-6
    assert not any("resonanceAbsorption" in u for u in report["unsupported"])


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


def test_an_empty_mapping_enables_resonance_absorption_like_true():
    """``resonance_absorption: {}`` is on, as ``true`` (PR #378 review): setup builds the same finite
    Landau-rate table for both, and a short run with ``{}`` stays finite (before the fix the
    light solver built the table only for a truthy option and handed the enabled RA no rates)."""
    from adept._lpse2d_cpp.core.light import CoupledLight

    tables = []
    for ra in (True, {}):
        light = CoupledLight(_finish(_cfg(11.0, ramp=True, ra=ra, tmax=0.02)))
        assert light.resonance.enabled
        tables.append(np.asarray(light.resonance.damping_over_k_sq))
    assert np.all(np.isfinite(tables[1])) and np.max(tables[1]) > 0.0
    np.testing.assert_array_equal(tables[1], tables[0])
    short = _cfg(11.0, ramp=True, ra={}, tmax=0.02)
    short["save"]["fields"]["t"]["dt"] = "0.01ps"  # a field frame inside the short run
    _, net, _, energy = _run(short)
    assert np.all(np.isfinite(net)) and np.all(np.isfinite(energy))


def _resonance(landau_update, dt_l, rate=None, filter_on=False):
    """A ResonanceAbsorption on the RA test box with its own update interval, sub-step and rate."""
    from adept._lpse2d_cpp.core.epw import analytic_landau_rate
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.core.resonance import ResonanceAbsorption

    cfg = _cfg(11.0, ramp=False, ra={"landau_update": landau_update, "filter": filter_on})
    cfg = _finish(cfg)
    light = CoupledLight(cfg)
    k_shape = (cfg["grid"]["nx"], cfg["grid"]["ny"])
    rates = np.asarray(analytic_landau_rate(cfg)) if rate is None else np.full(k_shape, float(rate))
    return cfg, ResonanceAbsorption(cfg, light, dt_l, rates)


@pytest.mark.parametrize(("landau_update", "n_sub", "n_epw"), [(10, 1, 20), (5, 3, 5)])
def test_resonance_landau_cadence_runs_over_absolute_light_steps(landau_update, n_sub, n_epw):
    """The k-space term acts on every ``landau_update``-th light sub-step counted across EPW steps
    (LPSE raLdCounter; PR #378 review), also when the interval exceeds and does not divide the
    sub-steps per EPW step. With the warm term zeroed and the k-space step marked by +1, the count
    over n_epw EPW steps of n_sub sub-steps is len(range(0, n_epw n_sub, landau_update)) exactly
    (before the fix the local sub-step index restarted every EPW step: 20 and 5 here, not 2 and 3)."""
    from jax import numpy as jnp

    dt = 1.0e-3  # the RA test box's EPW step (ps)
    dt_l = dt / n_sub
    _, ra = _resonance(landau_update, dt_l)
    ra.warm_term = lambda E: jnp.zeros_like(E)
    ra.landau_step = lambda E: E + 1.0
    E = jnp.zeros((4, 1, 2), dtype=jnp.complex128)
    for n in range(n_epw):
        for i in range(n_sub):
            E = ra(n * dt + i * dt_l, E)
    assert float(jnp.real(E[0, 0, 0])) == len(range(0, n_epw * n_sub, landau_update))


def test_resonance_landau_damping_is_bounded_over_the_update_interval():
    """Over the update interval T = landau_update dt_l the k-space term removes at most the
    longitudinal part (PR #378 review): the rate is capped at 1/T, not 1/dt_l. A longitudinal plane
    wave in the unmasked interior with landau_update = 10 is removed (|E| < 1e-12 A) at a huge rate,
    where the old cap multiplied it by 1 - 10 = -9, and halved at gamma T = 0.5 (rtol 1e-12)."""
    from jax import numpy as jnp

    dt_l = 1.0e-4
    for rate, factor in ((1.0e9, 0.0), (0.5 / (10 * dt_l), 0.5)):
        cfg, ra = _resonance(10, dt_l, rate=rate)
        x = np.asarray(cfg["grid"]["x"])
        k = float(np.asarray(cfg["grid"]["kx"])[5])
        ny = cfg["grid"]["ny"]
        amp = 1.0e-3
        E = np.zeros((x.size, ny, 2), dtype=np.complex128)
        E[..., 0] = amp * np.exp(1j * k * x)[:, None]
        out = np.asarray(ra.landau_step(jnp.asarray(E)))
        interior = np.asarray(ra.landau_mask)[:, 0, 0] == 1.0
        assert interior.sum() > 10
        got = np.abs(out[interior, :, 0])
        if factor == 0.0:
            assert np.max(got) < 1e-12 * amp
        else:
            np.testing.assert_allclose(got, factor * amp, rtol=1e-12)


def test_the_quasi_1d_resonance_filter_keeps_the_retained_band():
    """In quasi-1D (ny == 1) the RA filter's transition has zero width; it is a sharp passband
    |k| <= filter_width K_nyq, not an all-zero mask (PR #378 review: the zero-width s-curve made the
    default filter erase the pump on every damping update). A transverse (y-polarised) mode along x
    passes landau_step with the default filter unchanged (rtol 1e-12, fixed before the run), and a
    narrower filter_width keeps exactly the modes with |k| <= filter_width K_nyq."""
    from jax import numpy as jnp

    from adept._lpse2d_cpp.core.epw import analytic_landau_rate
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.core.resonance import ResonanceAbsorption

    for filter_width in (1.0, 0.5):
        cfg = _cfg(0.0, ramp=False, ra={"filter_width": filter_width})
        cfg["grid"].update({"ymax": "0.02um", "ymin": "-0.02um"})
        cfg["drivers"]["E0"].pop("beam_width")
        cfg = _finish(cfg)
        assert cfg["grid"]["ny"] == 1
        light = CoupledLight(cfg)
        ra = ResonanceAbsorption(cfg, light, light.dt_l, np.asarray(analytic_landau_rate(cfg)))
        x = np.asarray(cfg["grid"]["x"])
        kx = np.asarray(cfg["grid"]["kx"])
        k_nyq = np.abs(kx).max()
        for ik in (3, len(kx) // 3):
            k = float(kx[ik])
            E = np.zeros((x.size, 1, 2), dtype=np.complex128)
            E[:, 0, 1] = 1.0e-3 * np.exp(1j * k * x)
            out = np.asarray(ra.landau_step(jnp.asarray(E)))
            expected = E if abs(k) <= filter_width * k_nyq else 0.0 * E
            np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12 * 1.0e-3)


# ------------------------------------------------- LPSE's RA masks (review 2026-11-16 B10 / B11) --


def _lpse_s_curve(x, start, dx):
    """Beach.h:146-163, transcribed: no beaching (1) for dx <= 0."""
    if dx <= 0.0:
        return np.ones_like(x)
    y = (x - start) / dx
    return np.where(x <= start, 0.0, np.where(x >= start + dx, 1.0, (3.0 - 2.0 * y) * y * y))


def _lpse_left_ramp(x, start, dx):
    if dx <= 0.0:
        return np.ones_like(x)
    y = (x - start) / dx
    return np.where(x <= start, 0.0, np.where(x >= start + dx, 1.0, (2.0 - y) * y))


def _lpse_right_ramp(x, start, dx):
    if dx <= 0.0:
        return np.ones_like(x)
    y = (x - start) / dx
    return np.where(x <= start, 1.0, np.where(x >= start + dx, 0.0, 1.0 - y * y))


def _lpse_ra_beaches(n, h, labc, loff, source_edge_width):
    """LightSolver.cpp:2523-2543 on the nodes x_i = i h of [0, gridSize = (n - 1) h]:
    beachSource.halfBeachX (Lmin = Labc + Loff, Lmax = gridSize - (Labc + Loff), W = clip(sourceEdgeWidth, h))
    and beachRA.fullBeachX (0, gridSize, Labc + Loff on each side), Beach.h:166-216."""
    x = np.arange(n) * h
    size = (n - 1) * h
    w = max(source_edge_width, h)
    lmin, lmax = labc + loff, size - (labc + loff)
    half = _lpse_left_ramp(x, lmin, w) * _lpse_right_ramp(x, lmax - w, w)
    full = _lpse_s_curve(x, 0.0, labc + loff) * (1.0 - _lpse_s_curve(x, size - (labc + loff), labc + loff))
    return half, full


@pytest.mark.parametrize(
    ("edge_width", "offset", "y_boundary"), [(None, None, "periodic"), ("0.15um", "0.2um", "absorbing")]
)
def test_resonance_masks_are_lpse_beaches(edge_width, offset, y_boundary):
    """The warm-term and Landau-increment masks are LPSE's beaches on its node coordinates: they span
    Labc + Loff (``boundary_offset``) from each wall, the warm ramp is ``edge_width`` wide with LPSE's
    default of one cell (``clip(sourceEdgeWidth = 0, h)``), and adept's cell i is LPSE's node i. Before
    the fix the default ramp was 2 cells, Loff was ignored and the masks sat half a cell off. With
    periodic y the masks are uniform in y (LPSE's y beaches there are a departure, see
    test_resonance_absorption_departs_from_lpse_along_a_periodic_axis). rtol 1e-12."""
    from adept._lpse2d_cpp.core.light import CoupledLight

    ra = {}
    if edge_width is not None:
        ra["edge_width"] = edge_width
    if offset is not None:
        ra["boundary_offset"] = offset
    cfg = _cfg(11.0, ramp=True, ra=ra, tmax=0.02)
    cfg["terms"]["epw"]["boundary"]["y"] = y_boundary
    cfg = _finish(cfg)
    light = CoupledLight(cfg)
    h = cfg["grid"]["dx"]
    labc = cfg["grid"]["light_boundary_width_um"]
    loff = 0.0 if offset is None else float(offset.rstrip("um"))
    sew = 0.0 if edge_width is None else float(edge_width.rstrip("um"))
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    half_x, full_x = _lpse_ra_beaches(nx, h, labc, loff, sew)
    if y_boundary == "absorbing":
        half_y, full_y = _lpse_ra_beaches(ny, h, labc, loff, sew)
    else:
        half_y, full_y = np.ones(ny), np.ones(ny)
    warm = np.asarray(light.resonance.warm_mask)[..., 0]
    landau = np.asarray(light.resonance.landau_mask)[..., 0]
    np.testing.assert_allclose(warm, half_x[:, None] * half_y[None, :], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(landau, full_x[:, None] * full_y[None, :], rtol=1e-12, atol=1e-15)
    # one-cell default: on LPSE's nodes the ramp is a sharp step (Labc = 24 h here); a wider one has a skirt
    assert np.any((warm > 0.0) & (warm < 1.0)) == (sew > h)


def test_resonance_absorption_departs_from_lpse_along_a_periodic_axis():
    """LPSE applies its RA beaches along every axis of a 2-D grid. Along a periodic y (Labc.y = Loff.y = 0)
    beachRA.fullBeachX's ``1 - sCurve`` of zero width is 0 (Beach.h:146-168), so LPSE's Landau increment
    is zero everywhere and resonance absorption loses its damping; beachSource zeroes the warm term on
    the first and last y node. adept leaves a periodic axis unmasked (Departures from LPSE); the
    translator says so in its notes."""
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    _, full_y = _lpse_ra_beaches(16, 0.1, 0.0, 0.0, 0.0)
    assert np.all(full_y == 0.0)  # LPSE's mask along the periodic axis
    cfg = _finish(_cfg(11.0, ramp=True, ra=True, tmax=0.02))
    landau = np.asarray(CoupledLight(cfg).resonance.landau_mask)[..., 0]
    assert np.all(landau == landau[:, :1]) and landau.max() == 1.0

    parms = {
        "grid.sizes": "10 4",
        "grid.nodes": "100 40",
        "simulation.time.end": "1",
        "laser.enable": "true",
        "laser.solver": "fd",
        "laser.nBeams": "1",
        "laser.1.intensity": "1e15",
        "laser.evolution.Labc.min.x": "2",
        "laser.evolution.Labc.max.x": "2",
        "laser.evolution.resonanceAbsorption.enable": "true",
    }
    cfg_t, report = translate_parms(parms, run="ra")
    assert "resonance_absorption" in cfg_t["terms"]["light"]
    assert any("periodic y" in n and "resonanceAbsorption" in n for n in report["notes"])
    parms_loff = {**parms, "laser.evolution.Loff.min.x": "0.3", "laser.evolution.Loff.max.x": "0.3"}
    parms_loff["laser.evolution.sourceEdgeWidth"] = "0.25"
    cfg_t, report = translate_parms(parms_loff, run="ra")
    ra_t = cfg_t["terms"]["light"]["resonance_absorption"]
    assert ra_t["boundary_offset"] == "0.3um" and ra_t["edge_width"] == "0.25um"
    assert not any("Loff" in u for u in report["unsupported"])
    cfg_t, report = translate_parms({**parms_loff, "laser.evolution.Loff.max.x": "0.5"}, run="ra")
    assert any("laser.evolution.Loff per side" in u for u in report["unsupported"])


@pytest.mark.parametrize("order", [2, 4, 6])
def test_resonance_warm_term_converges_at_the_stencil_order(order):
    """LPSE refuses resonance absorption at solverOrder != 2 (LightSolver.cpp:925): its gradDivEo is
    the second-order 9-point stencil (:5163-5246). adept evaluates the warm term with the light solver's
    own stencils of ``fd_order``. At order 2 that is LPSE's stencil (transcribed here, rtol 1e-12 on a
    random field); at every order the term on a longitudinal plane wave in a uniform plasma,
    ``-3 C0 grad(div E) = 3 C0 k^2 E``, converges at the stencil's order: halving k (|k| dx ~ 0.6 to
    ~0.3, at ~30 deg) divides the error by 2^order within 25 % (fixed in advance)."""
    from jax import numpy as jnp

    from adept._lpse2d_cpp.core.epw import analytic_landau_rate
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.core.resonance import ResonanceAbsorption

    cfg = _cfg(11.0, ramp=False, ra=True, tmax=0.02)
    cfg["terms"]["light"]["fd_order"] = order
    cfg = _finish(cfg)
    light = CoupledLight(cfg)
    ra = ResonanceAbsorption(cfg, light, light.dt_l, np.asarray(analytic_landau_rate(cfg)))
    ra.warm_mask = jnp.ones_like(ra.warm_mask)
    x = np.asarray(cfg["grid"]["x"])
    y = np.asarray(cfg["grid"]["y"])
    h = cfg["grid"]["dx"]
    nx, ny = x.size, y.size

    errors = []
    # a grid mode at ~30 deg with |k| dx ~ 0.3, and the mode at twice its k
    mx = round(0.3 * np.cos(np.pi / 6) / h / (2 * np.pi / (nx * h)))
    my = round(0.3 * np.sin(np.pi / 6) / h / (2 * np.pi / (ny * h)))
    for scale in (2, 1):
        kx, ky = 2 * np.pi * scale * mx / (nx * h), 2 * np.pi * scale * my / (ny * h)
        k = np.hypot(kx, ky)
        phase = np.exp(1j * (kx * x[:, None] + ky * y[None, :]))
        E = np.stack([kx / k * phase, ky / k * phase], axis=-1)
        got = np.asarray(ra.warm_term(jnp.asarray(E)))
        want = 3.0 * ra.c0 * k**2 * E
        errors.append(np.abs(got - want).max() / np.abs(want).max())
    ratio = errors[0] / errors[1]
    assert 0.75 * 2**order < ratio < 1.25 * 2**order, (errors, ratio)

    if order == 2:
        # LPSE: C0 [divergence(E) gradNoOverNo - 3 gradDivEo], second order (LightSolver.cpp:4640-4643)
        rng = np.random.default_rng(5)
        E = rng.normal(size=(nx, ny, 2)) + 1j * rng.normal(size=(nx, ny, 2))
        ex, ey = E[..., 0], E[..., 1]

        def sh(f, i, j):
            return np.roll(f, (-i, -j), axis=(0, 1))

        div = (sh(ex, 1, 0) - sh(ex, -1, 0)) / (2 * h) + (sh(ey, 0, 1) - sh(ey, 0, -1)) / (2 * h)
        gdx = (
            (sh(ex, 1, 0) - 2 * ex + sh(ex, -1, 0))
            + (sh(ey, 1, 1) - sh(ey, -1, 1) - sh(ey, 1, -1) + sh(ey, -1, -1)) / 4
        ) / h**2
        gdy = (
            (sh(ey, 0, 1) - 2 * ey + sh(ey, 0, -1))
            + (sh(ex, 1, 1) - sh(ex, -1, 1) - sh(ex, 1, -1) + sh(ex, -1, -1)) / 4
        ) / h**2
        gx, gy = (np.asarray(g) for g in ra.grad_n_over_n)
        want = np.stack([ra.c0 * (div * gx - 3 * gdx), ra.c0 * (div * gy - 3 * gdy)], axis=-1)
        got = np.asarray(ra.warm_term(jnp.asarray(E)))
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
