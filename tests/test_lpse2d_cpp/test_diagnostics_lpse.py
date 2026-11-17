"""LPSE-style diagnostics: Poynting-flux fields (save.fields.poynting) and Thomson probes (save.thomson)."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from jax import numpy as jnp


def _finish(cfg):
    from adept._lpse2d_cpp.helpers import (
        get_density_profile,
        get_derived_quantities,
        get_save_quantities,
        get_solver_quantities,
        write_units,
    )

    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    cfg["grid"]["background_density"] = get_density_profile(cfg)
    cfg = get_save_quantities(cfg)
    return cfg


def _cfg(**save):
    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    cfg["density"] = {"basis": "uniform", "val": 0.2}
    cfg["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": False})
    # the full-grid field save (tpd.yaml's 0.08 um save spacing is finer than this 0.1 um grid, which
    # the interpolated save refuses since PR #388)
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    cfg["save"].update(save)
    return _finish(cfg)


def test_poynting_of_a_plane_wave_is_group_velocity_times_intensity(tmp_path):
    from adept._lpse2d_cpp.helpers import make_field_xarrays

    cfg = _cfg()
    cfg["save"]["fields"]["poynting"] = True
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k0 = kx[6]  # a resolved grid mode
    x = np.asarray(cfg["grid"]["x"])
    amp = 2.0e-3
    e0 = np.zeros((1, nx, ny, 2), dtype=np.complex128)
    e0[0, :, :, 1] = amp * np.exp(1j * k0 * x)[:, None]
    state = {
        "epw": np.zeros((1, nx, ny), dtype=np.complex128).view(np.float64),
        "E0": e0.view(np.float64),
        "E1": np.zeros((1, nx, ny, 2), dtype=np.complex128).view(np.float64),
    }
    (tmp_path / "binary").mkdir()
    _, fields = make_field_xarrays(cfg, np.array([0.0]), state, str(tmp_path))
    v_g = d["c"] ** 2 * k0 / d["w0"]
    s0x = fields["s0_x"].values[0, 4:-4, :]  # away from the one-sided gradient at the edges
    np.testing.assert_allclose(s0x, v_g * amp**2, rtol=2e-2)
    assert np.max(np.abs(fields["s0_y"].values)) < 1e-9 * v_g * amp**2
    assert np.max(np.abs(fields["s1_x"].values)) == 0.0


def test_poynting_counts_the_out_of_plane_component(tmp_path):
    """An s-polarised plane wave (E_z only, three-component state) carries the same flux v_g |E|^2 as a
    p-polarised one (rtol 2e-2, the centred difference's sin(k dx)/(k dx) at this k as above), and a
    wave with E_y and E_z carries the sum."""
    from adept._lpse2d_cpp.helpers import make_field_xarrays

    cfg = _cfg()
    cfg["save"]["fields"]["poynting"] = True
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    k0 = np.asarray(cfg["grid"]["kx"])[6]
    x = np.asarray(cfg["grid"]["x"])
    v_g = d["c"] ** 2 * k0 / d["w0"]
    amp = 2.0e-3
    for comps, expected in (((2,), amp**2), ((1, 2), 2.0 * amp**2)):
        e0 = np.zeros((1, nx, ny, 3), dtype=np.complex128)
        for c in comps:
            e0[0, :, :, c] = amp * np.exp(1j * k0 * x)[:, None]
        state = {
            "epw": np.zeros((1, nx, ny), dtype=np.complex128).view(np.float64),
            "E0": e0.view(np.float64),
            "E1": np.zeros((1, nx, ny, 3), dtype=np.complex128).view(np.float64),
        }
        out = tmp_path / f"c{len(comps)}"
        (out / "binary").mkdir(parents=True)
        _, fields = make_field_xarrays(cfg, np.array([0.0]), state, str(out))
        np.testing.assert_allclose(fields["s0_x"].values[0, 4:-4, :], v_g * expected, rtol=2e-2)


def _thomson_out(cfg, phi_k, iaw_density=None):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {
        "epw": phi_k,
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
        "E1": jnp.zeros((nx, ny, 2), dtype=jnp.complex128),
    }
    if iaw_density is not None:
        y["iaw_density"] = iaw_density
    return cfg["save"]["default"]["func"](0.0, y, {"drivers": {k: v["derived"] for k, v in cfg["drivers"].items()}})


def test_thomson_probe_is_lpse_rho_averaged_over_its_window():
    """LPSE's Thomson EPW value (physics review 2026-11-16, B8): rho_k = K^2 pots_k times
    rhoNormalizationFactor / NoOverNc (ThomsonScattering::collectRho, ParameterManager.cpp:1248), summed
    over the inclusive disc |K - K_probe|^2 <= (bandwidth K0)^2 and divided by its number of grid points
    (ThomsonScattering::reduce). Reference: that chain in ZAK units (ZakUnits, a transcription of
    ParameterManager) applied to the adept potential converted to ZAK; rtol 1e-6 (the ZAK constants
    are rounded the way LPSE's are, about 1e-7). The probe at 0.5 k0 with bandwidth 0.05 k0 holds 4
    grid points here; the old probe summed phi_k itself."""
    from adept._lpse2d_cpp.lpse_deck import ZakUnits

    cfg = _cfg(thomson=[{"k": [0.5, 0.0], "bandwidth": 0.05}, {"k": [-0.5, 0.0], "bandwidth": 0.05}])
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k0 = d["w0"] / d["c"]
    ix = int(np.argmin(np.abs(kx - 0.5 * k0)))
    phi_k = jnp.zeros((nx, ny), dtype=jnp.complex128).at[ix, 0].set(3.0 + 4.0j)
    out = _thomson_out(cfg, phi_k)

    zak = ZakUnits.from_cfg(cfg)
    k_zak = kx[ix] / zak.zak_per_um
    pots_zak = (3.0 + 4.0j) / zak.potential_zak_to_adept
    n_points = int(
        np.sum((kx[:, None] - 0.5 * k0) ** 2 + np.asarray(cfg["grid"]["ky"])[None, :] ** 2 <= (0.05 * k0) ** 2)
    )
    assert n_points == 4
    rho = k_zak**2 * pots_zak * zak.rho_normalization_factor / float(cfg["units"]["envelope density"]) / n_points
    np.testing.assert_allclose(float(out["thomson_0_re"]), rho.real, rtol=1e-6)
    np.testing.assert_allclose(float(out["thomson_0_im"]), rho.imag, rtol=1e-6)
    np.testing.assert_allclose(float(out["thomson_0_power"]), abs(rho) ** 2 * n_points, rtol=1e-6)
    assert float(out["thomson_1_power"]) == 0.0


def test_thomson_window_is_inclusive_and_never_empty():
    """LPSE ThomsonScattering::getKradiusSq: grid points on the circle count (<=), and a window with no
    grid point is widened to 1.01 x the nearest point's squared distance. A probe on a grid node with a
    radius of exactly two x spacings (= one y spacing here) holds 7 points (the strict disc held 3); a
    zero bandwidth off the grid holds the single nearest point. Exact counts; the amplitude is the
    mean of rho over them (rtol 1e-12)."""
    cfg0 = _cfg()
    kx = np.asarray(cfg0["grid"]["kx"])
    ky = np.asarray(cfg0["grid"]["ky"])
    d = cfg0["units"]["derived"]
    k0 = d["w0"] / d["c"]
    dkx, dky = kx[1] - kx[0], ky[1] - ky[0]
    np.testing.assert_allclose(2.0 * dkx, dky, rtol=1e-12)
    cfg = _cfg(
        thomson=[
            {"k": [18 * dkx / k0, 0.0], "bandwidth": 2.0 * dkx / k0},
            {"k": [18.3 * dkx / k0, 0.0], "bandwidth": 0.0},
        ]
    )
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    rng = np.random.default_rng(4)
    phi_k = jnp.asarray(rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny)))
    out = _thomson_out(cfg, phi_k)
    rho = d["e"] / (d["me"] * d["wp0"] ** 2) * (kx[:, None] ** 2 + ky[None, :] ** 2) * np.asarray(phi_k)
    window = [(18 + i, 0) for i in (-2, -1, 0, 1, 2)] + [(18, 1), (18, ny - 1)]
    want = np.mean([rho[i, j] for i, j in window])
    np.testing.assert_allclose(complex(out["thomson_0_re"] + 1j * out["thomson_0_im"]), want, rtol=1e-12)
    np.testing.assert_allclose(complex(out["thomson_1_re"] + 1j * out["thomson_1_im"]), rho[18, 0], rtol=1e-12)


def test_thomson_iaw_probe_is_the_mean_density_spectrum():
    """field: iaw -- LPSE collectNelf sums Nelf_k (delta n / n_b, adept's iaw_density) over the same
    inclusive disc and divides by its point count (rtol 1e-12)."""
    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        raw = deepcopy(yaml.safe_load(fi))
    raw["density"] = {"basis": "uniform", "val": 0.2}
    raw["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    raw["terms"]["epw"]["source"].update({"noise": False, "tpd": False})
    raw["terms"]["iaw"] = {"active": True, "solver": "spectral", "damping": {"landau": 0.0}}
    raw["save"]["fields"].pop("x", None)
    raw["save"]["fields"].pop("y", None)
    raw["save"]["thomson"] = [{"k": [0.5, 0.0], "bandwidth": 0.05, "field": "iaw"}]
    cfg = _finish(raw)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    ky = np.asarray(cfg["grid"]["ky"])
    d = cfg["units"]["derived"]
    k0 = d["w0"] / d["c"]
    rng = np.random.default_rng(8)
    density = 1e-3 * rng.normal(size=(nx, ny))
    out = _thomson_out(cfg, jnp.zeros((nx, ny), dtype=jnp.complex128), jnp.asarray(density))
    window = (kx[:, None] - 0.5 * k0) ** 2 + ky[None, :] ** 2 <= (0.05 * k0) ** 2
    spectrum = np.fft.fft2(density)
    want = spectrum[window].mean()
    np.testing.assert_allclose(complex(out["thomson_0_re"] + 1j * out["thomson_0_im"]), want, rtol=1e-12)
    np.testing.assert_allclose(float(out["thomson_0_power"]), np.mean(np.abs(spectrum[window]) ** 2), rtol=1e-12)


def test_translator_maps_the_thomson_probes():
    """thomsonScattering.N (ThomsonScattering::readParameters): a probe with a file maps each of its
    wavevectors (lw -> field epw, iaw -> field iaw) with the deck bandwidth (default 0); a probe without
    a file, or with a negative bandwidth, is disabled in LPSE and skipped; an IAW wavevector without
    IAWs is noted and skipped. Exact."""
    from adept._lpse2d_cpp.lpse_deck import translate_parms

    base = {
        "grid.sizes": "20 5",
        "grid.nodes": "201 51",
        "laser.enable": "true",
        "laser.wavelength": "0.351",
        "lw.enable": "true",
        "lw.envelopeDensity": "0.25",
        "lw.spectral.dt": "0.002",
        "simulation.time.end": "1",
        "thomsonScattering.nThomsonScattering": "4",
        "thomsonScattering.1.wavevector.lw": "1.2 0.2",
        "thomsonScattering.1.wavevector.iaw": "0.5 0.0",
        "thomsonScattering.1.bandwidth": "0.1",
        "thomsonScattering.1.file": "thomson_1.dat",
        "thomsonScattering.2.wavevector.lw": "-0.5 0.5",
        "thomsonScattering.2.file": "thomson_2.dat",
        "thomsonScattering.3.wavevector.lw": "0.7 0",
        "thomsonScattering.4.wavevector.lw": "0.9 0",
        "thomsonScattering.4.bandwidth": "-1",
        "thomsonScattering.4.file": "thomson_4.dat",
    }
    cfg, report = translate_parms(base, run="thomson")
    assert cfg["save"]["thomson"] == [
        {"k": [1.2, 0.2], "bandwidth": 0.1, "field": "epw"},
        {"k": [-0.5, 0.5], "bandwidth": 0.0, "field": "epw"},
    ]
    assert any("thomsonScattering.1.wavevector.iaw" in note for note in report["notes"])


@pytest.mark.parametrize("deck", ["test_010", "test_013", "test_022", "test_032"])
def test_thomson_window_holds_lpses_sample_count(deck):
    """Ground truth (physics review 2026-11-16, B8): for every Thomson probe of the shipped LPSE runs the
    translated probe's window holds exactly the number of grid points LPSE reports in its
    ``data/thomson_N.dat`` header ("Number of samples: (rho=...)"). Exact. Skipped without the
    original-lpse runs. The values are not asserted: LPSE's potential frames and its Thomson samples are
    one LPSE step apart (0.06-0.2 fs). Applied to the frames of test_013, test_022 and test_032, the
    probe was within 0.1-5.4 % (complex relative difference) of the next sample in 21 of 22 cases and
    18 % in the last (test_032, probe 1, 8 ps), consistent with one step of envelope-frame rotation."""
    from adept._lpse2d_cpp.helpers import get_derived_quantities, get_solver_quantities, write_units
    from adept._lpse2d_cpp.lpse_deck import parse_parms, translate_parms
    from adept._lpse2d_cpp.parity import lpse_root

    root = lpse_root()
    run = None if root is None else root / "runs" / deck
    if run is None or not (run / "data" / "thomson_1.dat").is_file():
        pytest.skip(f"original-lpse run {deck} with Thomson output not available")
    cfg, _ = translate_parms(parse_parms(run / "lpse.parms"), run=deck)
    write_units(cfg)
    cfg = get_derived_quantities(cfg)
    cfg["grid"] = get_solver_quantities(cfg)
    d = cfg["units"]["derived"]
    kx = np.asarray(cfg["grid"]["kx"])[:, None]
    ky = np.asarray(cfg["grid"]["ky"])[None, :]
    k0 = d["w0"] / d["c"]
    probes = cfg["save"]["thomson"]
    assert probes
    for i, probe in enumerate(probes, start=1):
        text = (run / "data" / f"thomson_{i}.dat").read_text()
        n_lpse = int(text.split("rho=")[1].split(")")[0])
        dist_sq = (kx - probe["k"][0] * k0) ** 2 + (ky - probe["k"][1] * k0) ** 2
        radius_sq = (probe["bandwidth"] * k0) ** 2
        if not np.any(dist_sq <= radius_sq * (1.0 + 1e-9)):
            radius_sq = 1.01 * float(dist_sq.min())
        assert int(np.sum(dist_sq <= radius_sq * (1.0 + 1e-9))) == n_lpse, (deck, i)


def _combined_cfg(envelope_density=0.25, density=0.2):
    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["units"]["envelope density"] = envelope_density
    cfg["density"] = {"basis": "uniform", "val": density}
    cfg["grid"].update({"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"})
    cfg["terms"]["epw"]["source"].update({"noise": False, "tpd": True, "srs": True})
    cfg["terms"]["epw"]["solver"] = "combined"
    cfg["terms"]["light"] = {"solver": "spectral", "pump_depletion": False}
    cfg["save"]["fields"]["poynting"] = True
    # the full-grid save path (both save paths take E1 through the same transverse projection,
    # helpers._raman_light_part, before any interpolation)
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    return _finish(cfg)


def _saved_fields(cfg, e1, tmp_path):
    """Run the field save function on one state and build the xarrays from its output."""
    from adept._lpse2d_cpp.helpers import make_field_xarrays

    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.asarray(e1).view(jnp.float64),
    }
    saved = cfg["save"]["fields"]["func"](0.0, y, None)
    state = {k: np.asarray(v)[None] for k, v in saved.items()}
    (tmp_path / "binary").mkdir(exist_ok=True)
    _, fields = make_field_xarrays(cfg, np.array([0.0]), state, str(tmp_path))
    return saved, fields


def test_combined_raman_flux_maps_exclude_the_epw(tmp_path):
    """The combined solver's E1 carries the EPW as its longitudinal part; the saved E1 fields and
    the s1 flux maps are the Raman light (transverse part) only, as the default series. Tolerances
    fixed before the run: a pure longitudinal mode leaves < 1e-9 of v_g |A|^2 in s1, and a mixed
    field's s1 equals its transverse part's to 1e-9."""
    cfg = _combined_cfg()
    d = cfg["units"]["derived"]
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    kx = np.asarray(cfg["grid"]["kx"])
    k = kx[6]
    x = np.asarray(cfg["grid"]["x"])
    amp = 2.0e-3
    wave = amp * np.exp(1j * k * x)[:, None] * np.ones((1, ny))
    longitudinal = np.zeros((nx, ny, 2), dtype=np.complex128)
    longitudinal[..., 0] = wave  # E parallel to k: a travelling plasma wave, no light
    transverse = np.zeros((nx, ny, 2), dtype=np.complex128)
    transverse[..., 1] = 0.5 * wave

    scale = d["c"] ** 2 * k / d["w1"] * amp**2
    saved, fields = _saved_fields(cfg, longitudinal, tmp_path)
    assert np.max(np.abs(np.asarray(saved["E1"]).view(np.complex128))) < 1e-9 * amp
    assert np.max(np.abs(fields["s1_x"].values)) < 1e-9 * scale
    assert np.max(np.abs(fields["s1_y"].values)) < 1e-9 * scale

    _, fields_mixed = _saved_fields(cfg, longitudinal + transverse, tmp_path)
    _, fields_t = _saved_fields(cfg, transverse, tmp_path)
    np.testing.assert_allclose(fields_mixed["s1_x"].values, fields_t["s1_x"].values, rtol=0, atol=1e-9 * scale)
    assert np.max(np.abs(fields_t["s1_x"].values)) > 0.1 * scale


def test_combined_raman_flux_uses_the_combined_carrier_away_from_quarter_critical(tmp_path):
    """The combined solver envelopes E1 at wp0 (CombinedSolver.prop_T), so its Raman flux is
    (c^2/wp0) Im(E* . grad E); the separate carrier w1 = w0 - wp0 agrees with it only at envelope
    density n_c/4 (PR #372 review). At n_env = 0.2, n = 0.15 nc the two differ by 24 %. Tolerances
    fixed before the run: the s1 map of a transverse plane wave equals c^2 k |A|^2 / wp0 to 2e-2
    (the central-difference error at this k, as for s0 above); the default series' Raman fluxes and
    reflectivity of the carrier-wp0 light mode equal c sqrt(eps) |A|^2 / I0 and
    sqrt(eps) |A|^2 / E0_source^2, eps = 1 - n/n_env, to 1e-9."""
    cfg = _combined_cfg(envelope_density=0.2, density=0.15)
    d = cfg["units"]["derived"]
    assert abs(d["w1"] / d["wp0"] - 1.0) > 0.2
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    x = np.asarray(cfg["grid"]["x"])
    amp = 2.0e-3

    k = np.asarray(cfg["grid"]["kx"])[6]
    e1 = np.zeros((nx, ny, 2), dtype=np.complex128)
    e1[..., 1] = amp * np.exp(1j * k * x)[:, None]
    _, fields = _saved_fields(cfg, e1, tmp_path)
    s1x = fields["s1_x"].values[0, 4:-4, :]  # away from the one-sided gradient at the edges
    np.testing.assert_allclose(s1x, d["c"] ** 2 * k / d["wp0"] * amp**2, rtol=2e-2)

    eps = 1.0 - 0.15 / 0.2
    k_mode = d["wp0"] / d["c"] * np.sqrt(eps)
    save_func = cfg["save"]["default"]["func"]
    args = {"drivers": {name: v["derived"] for name, v in cfg["drivers"].items()}}
    for sign, key in ((-1.0, "reflected_flux"), (1.0, "backrefl_flux")):
        e1 = np.zeros((nx, ny, 2), dtype=np.complex128)
        e1[..., 1] = amp * np.exp(1j * sign * k_mode * x)[:, None]
        y = {
            "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
            "E0": jnp.zeros((nx, ny, 2), dtype=jnp.complex128).view(jnp.float64),
            "E1": jnp.asarray(e1).view(jnp.float64),
        }
        out = save_func(0.0, y, args)
        np.testing.assert_allclose(float(out[key]), d["c"] * np.sqrt(eps) * amp**2 / d["I0_code"], rtol=1e-9)
        np.testing.assert_allclose(float(out["reflectivity"]), np.sqrt(eps) * amp**2 / d["E0_source"] ** 2, rtol=1e-9)


def test_poynting_refuses_a_single_point_x_save_grid():
    """An interpolated save of one x point (save.fields.x.dx the box length) cannot take the flux maps'
    d/dx: with save.fields.poynting it is refused at setup, not after the solve (PR #372 review); two or
    more points are accepted."""
    import pytest

    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        base = yaml.safe_load(fi)
    base["density"] = {"basis": "uniform", "val": 0.2}
    base["grid"].update(
        {"dx": "0.1um", "xmax": "12.8um", "ymax": "3.2um", "ymin": "-3.2um", "dt": "1fs", "tmax": "2fs"}
    )
    base["terms"]["epw"]["source"].update({"noise": False, "tpd": False})
    single = deepcopy(base)
    single["save"]["fields"].update({"x": {"dx": "12.8um"}, "y": {"dy": "0.2um"}, "poynting": True})
    with pytest.raises(ValueError, match="at least two saved x points"):
        _finish(single)
    two = deepcopy(base)
    two["save"]["fields"].update({"x": {"dx": "6.4um"}, "y": {"dy": "0.2um"}, "poynting": True})
    assert _finish(two)["save"]["fields"]["x"]["ax"].size == 2


# --- EPW total-to-electric energy factor (physics review 2026-11-16, C3) ---


def _srs_box(density, envelope_density, pump_depletion=False):
    with open("tests/test_lpse2d_cpp/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["units"]["envelope density"] = envelope_density
    cfg["density"] = density
    cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "0.02um", "ymin": "-0.02um", "dx": "0.1um"})
    cfg["terms"]["epw"]["source"]["noise"] = False
    if pump_depletion:
        cfg["terms"]["light"] = {"pump_depletion": True}
        cfg["terms"]["epw"]["boundary"]["x"] = "absorbing"
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    return _finish(cfg)


def test_epw_total_energy_factor_is_w_deps_dw_at_the_carrier():
    """epw_total_energy_factor is the warm-fluid w d(eps)/dw of the EPW at its carrier w = wp0, with
    eps(w) = 1 - wp^2 / (w^2 - 3 k^2 vte^2), wp^2 = wp0^2 n/n_env and the resonant k of the solver's
    detuning, 3 k^2 vte^2 = wp0^2 - wp^2. The reference differentiates eps numerically (central
    difference, step 1e-5 wp0) on a ramp from 0.15 to 0.25 n_c with n_env = 0.25: rtol 1e-7, fixed before
    the run (the difference's O(h^2) error is ~1e-10). The previous factor 2 (2 - n/n_env) was 16 % low
    at the 0.15 end."""
    from adept._lpse2d_cpp.helpers import epw_total_energy_factor

    cfg = _srs_box({"basis": "linear", "gradient scale length": "30um", "min": 0.15, "max": 0.25}, 0.25)
    n = np.asarray(cfg["grid"]["background_density"])
    assert n.min() < 0.16 and n.max() <= 0.25
    got = epw_total_energy_factor(cfg)
    wp0 = cfg["units"]["derived"]["wp0"]
    wp_sq = wp0**2 * n / 0.25
    three_k2_v2 = wp0**2 - wp_sq

    def eps(w):
        return 1.0 - wp_sq / (w**2 - three_k2_v2)

    h = 1e-5 * wp0
    want = wp0 * (eps(wp0 + h) - eps(wp0 - h)) / (2.0 * h)
    np.testing.assert_allclose(got, want, rtol=1e-7)


def test_srs_couplings_conserve_the_factor_weighted_epw_energy():
    """The separate solver's SRS couplings -- the EPW source (SpectralEPWSolver.calc_srs_source, which
    carries n/n_env) and the two light couplings (CoupledLight.srs_depletion_coeff0 on E0,
    RamanLight.srs_coeff on E1) -- conserve |E0|^2 + |E1|^2 + w |E_epw|^2 with w = epw_total_energy_factor / 2
    (the light's total energy is twice its electric energy). For one resonant triplet, d/dt of each part
    follows from the coefficients; w is solved from five random states at n/n_env = 0.75 and must be the
    same for all and equal factor / 2: rtol 1e-10, fixed before the run (round-off)."""
    from adept._lpse2d_cpp.core.epw import SpectralEPWSolver
    from adept._lpse2d_cpp.core.light import CoupledLight
    from adept._lpse2d_cpp.helpers import epw_total_energy_factor

    cfg = _srs_box({"basis": "uniform", "val": 0.15}, 0.2)
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    epw = SpectralEPWSolver(cfg)
    # a pump plane wave on an in-band mode m and a uniform Raman field: the source (a scalar times
    # E0 . E1*, density factor included) lands in mode m -- not k = 0, which the source omits, as LPSE
    a0, a1, m = 0.7 - 0.2j, 0.3 + 0.5j, 3
    x = np.asarray(cfg["grid"]["x"])
    e0 = np.zeros((nx, ny, 3), dtype=np.complex128)
    e1 = np.zeros((nx, ny, 3), dtype=np.complex128)
    e0[..., 1] = (a0 * np.exp(1j * np.asarray(cfg["grid"]["kx"])[m] * (x - x[0])))[:, None]
    e1[..., 1] = a1
    src = np.asarray(epw.calc_srs_source(jnp.asarray(e0), jnp.asarray(e1)))
    p_eff = src[m, 0] / (nx * ny) / (a0 * np.conj(a1))
    light = CoupledLight(_srs_box({"basis": "uniform", "val": 0.15}, 0.2, pump_depletion=True))
    a_dep, b_src = complex(light.srs_depletion_coeff0), complex(light.srs_coeff)
    rng = np.random.default_rng(5)
    k_sq = 3.7  # any |k|^2: it cancels
    weights = []
    for _ in range(5):
        e0_, e1_, phi = rng.normal(size=3) + 1j * rng.normal(size=3)
        lap = -k_sq * phi
        d_light = 2.0 * np.real(np.conj(e0_) * a_dep * lap * e1_) + 2.0 * np.real(
            np.conj(e1_) * b_src * np.conj(lap) * e0_
        )
        d_epw = 2.0 * k_sq * np.real(np.conj(phi) * p_eff * e0_ * np.conj(e1_))
        weights.append(-d_light / d_epw)
    factor = epw_total_energy_factor(cfg)
    assert np.ptp(factor) == 0.0
    np.testing.assert_allclose(weights, float(factor.flat[0]) / 2.0, rtol=1e-10)


# --- the incident laser power and the laser-budget denominators (physics review 2026-11-16, C1, C2, C4) ---


def _pump_box(angle, *, pump_depletion, light_solver="fd", density=None, swelling="constant"):
    with open("tests/test_lpse2d_cpp/configs/srs.yaml") as fi:
        cfg = deepcopy(yaml.safe_load(fi))
    cfg["grid"].update({"xmax": "12.8um", "tmax": "10fs", "ymax": "3.2um", "ymin": "-3.2um", "dx": "0.05um"})
    cfg["terms"]["light"] = {"solver": light_solver, "pump_depletion": pump_depletion}
    cfg["terms"]["epw"]["source"]["noise"] = False
    cfg["terms"]["epw"]["boundary"] = {"x": "absorbing" if pump_depletion else "periodic", "y": "periodic"}
    cfg["drivers"]["E0"]["angle"] = angle
    cfg["drivers"]["E0"]["swelling"] = swelling
    if density is not None:
        cfg["density"] = density
    cfg["save"]["fields"].pop("x", None)
    cfg["save"]["fields"].pop("y", None)
    return _finish(cfg)


def _save(cfg, e0):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    y = {
        "epw": jnp.zeros((nx, ny), dtype=jnp.complex128).view(jnp.float64),
        "E0": jnp.asarray(e0, dtype=jnp.complex128).view(jnp.float64),
        "E1": jnp.zeros(e0.shape, dtype=jnp.complex128).view(jnp.float64),
    }
    return {k: float(v) for k, v in cfg["save"]["default"]["func"](0.0, y, None).items() if np.ndim(v) == 0}


@pytest.mark.parametrize(
    "angle, swelling, density",
    [
        (0.0, "constant", None),
        (30.0, "constant", None),
        (0.0, "local", {"basis": "linear", "gradient scale length": "30um", "min": 0.15, "max": 0.25}),
        (30.0, "constant", {"basis": "linear", "gradient scale length": "30um", "min": 0.15, "max": 0.25}),
        # past the critical density the local swelling zeroes the pump: those cells are left out
        (0.0, "local", {"basis": "linear", "gradient scale length": "30um", "min": 0.5, "max": 1.5}),
    ],
)
def test_prescribed_pump_incident_power_is_its_x_flux(angle, swelling, density):
    """The prescribed pump's incident_pump_flux (units of I0) is the x-projection of the beam it
    represents: cos of the grid-snapped beam direction (laser.Light snaps kx and ky), with either swelling
    model, on a uniform box or a ramp. Reference: the snapped direction from k0(n_env) rounded to the grid,
    computed here. rtol 1e-9, fixed before the run (round-off of the FFT sums). The net two-point probe the
    budget divided by before (incident_flux) reads 0.987 at 0 deg and 0.865 at 30 deg on this grid."""
    from adept._lpse2d_cpp.core.laser import Light

    cfg = _pump_box(angle, pump_depletion=False, density=density, swelling=swelling)
    ny = cfg["grid"]["ny"]
    lw = {"delta_omega": jnp.array([0.0]), "intensities": jnp.ones((1, ny)), "phases": jnp.zeros((1, ny))}
    e0 = np.asarray(Light(cfg).laser_update(0.0, None, lw))
    d = cfg["units"]["derived"]
    k0 = d["w0"] / d["c"] * np.sqrt(1.0 - cfg["units"]["envelope density"])
    dk = 2.0 * np.pi / (cfg["grid"]["nx"] * cfg["grid"]["dx"])
    dky = 2.0 * np.pi / (ny * cfg["grid"]["dy"])
    kx = np.round(k0 * np.cos(np.deg2rad(angle)) / dk) * dk
    ky = np.round(k0 * np.sin(np.deg2rad(angle)) / dky) * dky
    np.testing.assert_allclose(_save(cfg, e0)["incident_pump_flux"], kx / np.hypot(kx, ky), rtol=1e-9)


def _discrete_plane_wave(cfg, iky, direction, amp, light_solver):
    """A pump plane wave on the solver's own dispersion at the box's uniform density, transverse mode
    ``ky[iky]``, travelling toward +x (``direction = 1``) or -x, polarised along z (all rows)."""
    grid, d = cfg["grid"], cfg["units"]["derived"]
    n = float(np.asarray(grid["background_density"])[0, 0])
    k_sq = (d["w0"] / d["c"]) ** 2 * (1.0 - n)
    ky = float(np.asarray(grid["ky"])[iky])
    dx, dy = grid["dx"], grid["dy"]
    if light_solver == "spectral":
        theta = np.sqrt(k_sq - ky**2) * dx
    else:  # second-order stencils in x and y: (2 - 2 cos(ky dy))/dy^2 + (2 - 2 cos theta)/dx^2 = k^2
        ky_eff_sq = (2.0 - 2.0 * np.cos(ky * dy)) / dy**2
        theta = np.arccos(1.0 - (k_sq - ky_eff_sq) * dx**2 / 2.0)
    j = np.arange(grid["nx"])[:, None]
    y = np.asarray(grid["y"])[None, :]
    e = np.zeros((grid["nx"], grid["ny"], 3), dtype=np.complex128)
    e[..., 2] = amp * np.exp(1j * (direction * theta * j + ky * y))
    return e


@pytest.mark.parametrize("light_solver", ["fd", "spectral"])
@pytest.mark.parametrize("angle", [0.0, 180.0])
def test_evolved_pump_incident_power_is_the_inflowing_one_way_flux(light_solver, angle):
    """With the pump evolved, incident_pump_flux is the one-way flux into the box at the entrance probe:
    for a field of two inflowing waves (normal and oblique) plus two outflowing ones (reflected pump
    light), it equals the net probe flux of the inflowing waves alone, at the x-min probe (beams at 0 deg)
    or the x-max probe (180 deg). rtol 1e-9, fixed before the run (round-off): the split is exact for
    waves on the solver's dispersion at the probe density, built here from the second-order stencil
    relation or the exact one. Before the fix the budget divided by the net flux, which the outflowing
    waves reduce."""
    cfg = _pump_box(angle, pump_depletion=True, light_solver=light_solver)
    rightward = angle == 0.0
    sign = 1 if rightward else -1
    inflow = _discrete_plane_wave(cfg, 0, sign, 0.8, light_solver) + _discrete_plane_wave(
        cfg, 3, sign, 0.3 - 0.2j, light_solver
    )
    outflow = _discrete_plane_wave(cfg, 0, -sign, 0.25j, light_solver) + _discrete_plane_wave(
        cfg, -5, -sign, 0.1, light_solver
    )
    e0_source = cfg["units"]["derived"]["E0_source"]
    inflow, outflow = inflow * e0_source, outflow * e0_source
    full, inflow_only = _save(cfg, inflow + outflow), _save(cfg, inflow)
    probe = "incident_flux" if rightward else "transmitted_flux"
    want = sign * inflow_only[probe]
    assert want > 0.5  # the inflowing waves carry most of the nominal flux
    assert abs(full[probe]) < 0.95 * want  # the outflowing waves reduce the net flux by > 5 %
    np.testing.assert_allclose(full["incident_pump_flux"], want, rtol=1e-9)


def _budget_cfg(*, pump_depletion, leftward=False):
    w0 = 5.366e3  # rad/ps at 351 nm
    return {
        "units": {"derived": {"w0": w0, "E0_source": 2.0e-3, "e_norm": 7.0e2}},
        "terms": {"epw": {"source": {"noise": False}}, "light": {"pump_depletion": pump_depletion}},
        "drivers": {"E0": {"derived": {"beam_leftward": np.array([leftward])}}},
    }


def _series(t, **kw):
    import xarray as xr

    data = {"epw_energy": xr.DataArray(1e-6 * np.exp(t), coords=(("t (ps)", t),))}
    for k, v in kw.items():
        data[k] = xr.DataArray(np.broadcast_to(np.asarray(v, dtype=float), t.shape).copy(), coords=(("t (ps)", t),))
    return xr.Dataset(data)


@pytest.mark.parametrize("leftward", [False, True])
def test_evolved_budget_is_a_fraction_of_the_incident_power(leftward):
    """Evolved pump, steady state: the injector delivers I_in = 0.98 I0, the pump light leaving through the
    entrance is R_E0 = 0.10, the Raman light R_SRS = 0.05, the plasma absorbs A = 0.20 (all of it through
    the EPW) and T = 0.65. The budget must report R = R_SRS + R_E0 = 0.15, T = 0.65, absorbed = 0.20,
    laser_absorbed_frac_epw = electron_energy_frac_final = 0.20 and laser_incident_flux_ratio = 0.98, for
    the x-min entrance and its mirror image (every beam leftward). Exact arithmetic: rtol 1e-12, fixed
    before the run. Before the fix: R = R_SRS / (1 - R_E0) = 0.056, A_epw = 0.196, ratio 0.882."""
    from adept._lpse2d_cpp.diagnostics import series_metrics

    cfg = _budget_cfg(pump_depletion=True, leftward=leftward)
    d = cfg["units"]["derived"]
    i0_osiris = 0.5 * (d["E0_source"] * d["e_norm"]) ** 2
    i_in, r_e0, r_srs, absorbed = 0.98, 0.10, 0.05, 0.20
    t = np.linspace(0.0, 4.0, 401)
    net_pump_left, raman_left = i_in * (1.0 - r_e0), i_in * r_srs  # the entrance face
    pump_right = i_in * (1.0 - r_e0 - r_srs - absorbed)
    dissipation = absorbed * i_in * d["w0"] * i0_osiris
    if not leftward:
        flux = dict(incident_flux=net_pump_left, reflected_flux=raman_left, transmitted_flux=pump_right)
    else:  # x -> -x: the entrance is the x-max face, every flux is a signed +x flux
        flux = dict(incident_flux=-pump_right, backrefl_flux=raman_left, transmitted_flux=-net_pump_left)
    flux = {"reflected_flux": 0.0, "backrefl_flux": 0.0, **flux}
    series = _series(t, incident_pump_flux=i_in, epw_dissipation=dissipation, **flux)
    m = series_metrics(series, cfg)
    for key, want in (
        ("laser_reflectivity", r_e0 + r_srs),
        ("laser_transmissivity", 1.0 - r_e0 - r_srs - absorbed),
        ("laser_absorbed_frac", absorbed),
        ("laser_reflectivity_naive", r_srs),
        ("laser_absorbed_frac_epw", absorbed),
        ("laser_absorbed_frac_epw_seg1of4", absorbed),
        ("electron_energy_frac_final", absorbed),
        ("laser_incident_flux_ratio", i_in),
    ):
        np.testing.assert_allclose(m[key], want, rtol=1e-12, err_msg=key)


def test_prescribed_reflectivity_is_a_fraction_of_the_pump_x_flux():
    """Prescribed pump at 30 deg: the pump's x-flux is 0.866 I0 and the Raman light leaving the entrance is
    0.0866 I0, so laser_reflectivity = 0.10 (before the fix: 0.0866, divided by 1). Exact: rtol 1e-12."""
    from adept._lpse2d_cpp.diagnostics import series_metrics

    cfg = _budget_cfg(pump_depletion=False)
    t = np.linspace(0.0, 4.0, 401)
    i_in = np.cos(np.deg2rad(30.0))
    series = _series(
        t,
        incident_pump_flux=i_in,
        incident_flux=i_in,
        reflected_flux=0.1 * i_in,
        transmitted_flux=i_in,
        backrefl_flux=0.0,
    )
    m = series_metrics(series, cfg)
    np.testing.assert_allclose(m["laser_reflectivity"], 0.1, rtol=1e-12)
    np.testing.assert_allclose(m["laser_reflectivity_seg2of4"], 0.1, rtol=1e-12)
    assert "laser_transmissivity" not in m and "laser_absorbed_frac" not in m


def test_fractions_follow_the_delivered_pulse():
    """A pump that turns on at mid-run (incident power 0, then 1) with the EPW dissipating 0.3 of it: the
    delivered fluence is half the nominal I0 t_end, so electron_energy_frac_final = 0.3 (the trapezoids of
    both integrals share the step: exact, rtol 1e-12, fixed before the run; before the fix 0.15, the
    fluence taken as I0 t_end); the quarters with the laser off log no fractions, the last one logs
    laser_absorbed_frac_epw = 0.3."""
    from adept._lpse2d_cpp.diagnostics import series_metrics

    cfg = _budget_cfg(pump_depletion=True)
    d = cfg["units"]["derived"]
    i0_osiris = 0.5 * (d["E0_source"] * d["e_norm"]) ** 2
    t = np.linspace(0.0, 4.0, 401)
    on = (t >= 2.0).astype(float)
    series = _series(
        t,
        incident_pump_flux=on,
        incident_flux=on,
        reflected_flux=0.0,
        transmitted_flux=0.7 * on,
        backrefl_flux=0.0,
        epw_dissipation=0.3 * d["w0"] * i0_osiris * on,
    )
    m = series_metrics(series, cfg)
    np.testing.assert_allclose(m["electron_energy_frac_final"], 0.3, rtol=1e-12)
    np.testing.assert_allclose(m["laser_absorbed_frac_epw"], 0.3, rtol=1e-12)
    np.testing.assert_allclose(m["laser_absorbed_frac"], 0.3, rtol=1e-12)
    assert "laser_reflectivity_seg1of4" not in m and "laser_absorbed_frac_epw_seg2of4" not in m
