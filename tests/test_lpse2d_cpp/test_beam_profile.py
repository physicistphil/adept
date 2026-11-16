"""The injected beams' transverse super-Gaussian against LPSE (``SchrodingerSolver3::superGaussian``).

LPSE: ``exp(-(|r| / W)^sgOrder)`` from ``laser.N.evolution.{width, sgOrder | sgPower, offset}``,
``sgOrder`` defaulting to 4 (LightSolver.cpp:1704) and a zero width or order giving a flat beam.
Tolerance fixed in advance: 1e-12 relative on the profile.
"""

from copy import deepcopy

import numpy as np
import pytest
import yaml


def _lpse_super_gaussian(y, width, order, offset):
    """SchrodingerSolver3::superGaussian, transcribed for the in-plane distance |y - offset|."""
    if width == 0.0 or order == 0.0:
        return np.ones_like(y)
    return np.exp(-(np.abs((y - offset) / width) ** order))


def _lpse_beam_weight(y, width, order, offset, kx, ky):
    """SchrodingerSolver3.cpp:2375-2392: R = (x, y) - offset with the plane-normal component zeroed,
    rPrime = R - Kdir (Kdir . R), weight = superGaussian(|rPrime|)."""
    r = np.stack([np.zeros_like(y), y - offset], axis=-1)
    kdir = np.array([kx, ky]) / np.hypot(kx, ky)
    r_prime = r - kdir[None, :] * (r @ kdir)[:, None]
    return _lpse_super_gaussian(np.linalg.norm(r_prime, axis=-1), width, order, 0.0)


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


def _cfg(solver, **e0):
    with open("tests/test_lpse2d_cpp/configs/srs.yaml") as fi:
        raw = deepcopy(yaml.safe_load(fi))
    raw["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "6.4um", "ymin": "-6.4um", "dx": "0.1um"})
    raw["terms"]["light"] = {"solver": solver, "pump_depletion": True}
    raw["terms"]["epw"]["source"]["noise"] = False
    raw["terms"]["epw"]["boundary"] = {"x": "absorbing", "y": "periodic"}
    raw["drivers"]["E0"].update(e0)
    return _finish(raw)


def _translate(extra):
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    parms = {
        "grid.sizes": "20 10",
        "grid.nodes": "200 100",
        "simulation.time.end": "1",
        "laser.enable": "true",
        "laser.nBeams": "1",
        "laser.1.intensity": "1e15",
        **extra,
    }
    return translate_parms(parms, run="x")


@pytest.mark.parametrize("solver", ["spectral", "fd"])
@pytest.mark.parametrize("order", [None, 2.0, 6.0, 0.0])
def test_injected_beam_profile_is_lpse_super_gaussian(solver, order):
    """adept's beam_width is W / sqrt(2); with beam_sg_order omitted the order is LPSE's 4."""
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.core.spectral_light import SpectralCoupledLight

    w_lpse, offset = 3.0, 1.0
    e0 = {"beam_width": f"{w_lpse / np.sqrt(2.0)}um", "beam_offset": f"{offset}um"}
    if order is not None:
        e0["beam_sg_order"] = order
    if solver == "fd":
        e0["angle"] = 10.0  # the general FD injector carries the profile
    cfg = _cfg(solver, **e0)
    light = (SpectralCoupledLight if solver == "spectral" else CoupledLight)(cfg)
    y = np.asarray(cfg["grid"]["y"])
    ky = float(light.ky_beams[0])
    k0 = light.w0 / light.c * np.sqrt(1.0 - light.beam_n_src[0])
    want = _lpse_beam_weight(y, w_lpse, 4.0 if order is None else order, offset, np.sqrt(k0**2 - ky**2), ky)
    np.testing.assert_allclose(np.asarray(light.beam_envelope_y)[0], want, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("solver", ["spectral", "fd"])
@pytest.mark.parametrize("snap", [True, False])
def test_oblique_beam_profile_is_across_the_beam_axis(solver, snap):
    """Every beam's profile is LPSE's weight at rPrime, the distance from the beam axis, i.e. at
    |y - y0| cos(theta) on the injection plane: an oblique beam keeps its width across its axis and
    its footprint widens by 1 / cos(theta) (the old profile in y alone made it cos(theta) narrower
    and cos(theta) less powerful: -6 % at 20 deg, -29 % at 45 deg). rtol 1e-12."""
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.core.spectral_light import SpectralCoupledLight

    w_lpse, offset, order = 2.0, 0.5, 4.0
    cfg = _cfg(
        solver,
        beam_width=f"{w_lpse / np.sqrt(2.0)}um",
        beam_offset=f"{offset}um",
        beam_sg_order=order,
        beams=[{"intensity": 0.5, "angle": 20.0}, {"intensity": 0.5, "angle": -45.0}],
    )
    cfg["terms"]["light"]["snap_beam_ky"] = snap
    light = (SpectralCoupledLight if solver == "spectral" else CoupledLight)(cfg)
    y = np.asarray(cfg["grid"]["y"])
    env = np.asarray(light.beam_envelope_y)
    assert env.shape == (2, y.size)
    for b, angle in enumerate((20.0, -45.0)):
        ky = float(light.ky_beams[b])
        k0 = light.w0 / light.c * np.sqrt(1.0 - light.beam_n_src[b])
        if not snap:
            np.testing.assert_allclose(ky, k0 * np.sin(np.deg2rad(angle)), rtol=1e-12)
        want = _lpse_beam_weight(y, w_lpse, order, offset, np.sqrt(k0**2 - ky**2), ky)
        np.testing.assert_allclose(env[b], want, rtol=1e-12, atol=1e-300)
    # the 45 deg footprint is sqrt(2) wider than the 0 deg one: its half-maximum points
    half = env[1] >= 0.5
    width_45 = (y[half].max() - y[half].min()) if half.any() else 0.0
    w_half_0 = 2.0 * w_lpse * np.log(2.0) ** (1.0 / order)
    assert abs(width_45 / w_half_0 - 1.0 / np.cos(np.arcsin(float(light.ky_beams[1]) / k0))) < 0.1


def test_translator_beam_profile_keys_and_defaults():
    cfg, _ = _translate({"laser.1.evolution.width": "3"})
    assert cfg["drivers"]["E0"]["beam_sg_order"] == 4.0  # LightSolver.cpp:1704
    cfg, _ = _translate(
        {"laser.1.evolution.width": "3", "laser.1.evolution.sgOrder": "2", "laser.1.evolution.sgPower": "6"}
    )
    assert cfg["drivers"]["E0"]["beam_sg_order"] == 6.0  # sgPower is read after sgOrder
    for flat in ({"laser.1.evolution.width": "3", "laser.1.evolution.sgOrder": "0"}, {}):
        cfg, _ = _translate(flat)
        assert "beam_width" not in cfg["drivers"]["E0"]
    cfg, report = _translate({"laser.1.width": "3"})
    assert "beam_width" not in cfg["drivers"]["E0"]
    assert any("laser.1.width: not an LPSE key" in u for u in report["unsupported"])
