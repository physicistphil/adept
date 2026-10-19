"""``terms.hpe.tail_distribution``: the thermal law of the loaded (and wall re-injected) HPE tail.

``gaussian_v`` (default) is the closed-form Maxwellian in v and must be unchanged; ``gaussian_u`` (OSIRIS ``uth``) and
``juttner`` (longitudinal marginal of the 3-D Maxwell-Juttner law) are tabulated in u = gamma v. Tolerances were fixed
before the tests were first run: tabulated tail fractions within 1e-5 relative of closed forms / quadrature; loaded
histograms against the expected law chi^2 < dof + 5 sqrt(2 dof); wall re-injection fractions at independently
integrated deciles within 5 binomial standard deviations.
"""

import numpy as np
import pytest
from jax import numpy as jnp
from scipy import integrate, special

from .test_hpe import _make_cfg, _make_cfg_2d

KINDS = ("gaussian_v", "gaussian_u", "juttner")
MC2_KEV = 510.99895


def _vte_c(cfg):
    derived = cfg["units"]["derived"]
    return float(np.sqrt(derived["vte_sq"])), float(derived["c"])


def test_default_tail_distribution_is_unchanged():
    from adept._lpse2d.core.hpe import load_particles, resonance_arrays, tail_fraction

    implicit = _make_cfg({"n_particles": 20000})
    explicit = _make_cfg({"n_particles": 20000, "tail_distribution": "gaussian_v"})
    a, b = load_particles(implicit), load_particles(explicit)
    for key in ("x_e", "u_e", "epw_hist"):
        np.testing.assert_array_equal(a[key], b[key])
    ra, rb = resonance_arrays(implicit), resonance_arrays(explicit)
    assert ra["f_tail_frac"] == rb["f_tail_frac"] == float(special.erfc(2.5 / np.sqrt(2.0)))
    np.testing.assert_array_equal(ra["f0_expected"], rb["f0_expected"])
    assert tail_fraction({"v_min": 2.5}, 1.0, 1.0) == float(special.erfc(2.5 / np.sqrt(2.0)))


@pytest.mark.parametrize("te_kev", [2.0, 10.0])
def test_tabulated_tail_fractions_match_closed_forms_and_quadrature(te_kev):
    from adept._lpse2d.core import hpe as hpe_mod

    c = 1.0
    vte = np.sqrt(te_kev / MC2_KEV) * c
    v_min = 2.5
    u_lo = float(hpe_mod._u_of_v(v_min * vte, c))

    # the numerical path itself, applied to the Maxwellian in v, against erfc
    u, cdf = hpe_mod._full_law_cdf("gaussian_v", vte, c, u_lo)
    np.testing.assert_allclose(1.0 - np.interp(u_lo, u, cdf), special.erfc(v_min / np.sqrt(2.0)), rtol=1e-5)

    # Gaussian in u: P(|u| > u(v_min)) = erfc(u_lo / sqrt(2) vte)
    frac_u = hpe_mod.tail_fraction({"v_min": v_min, "tail_distribution": "gaussian_u"}, vte, c)
    np.testing.assert_allclose(frac_u, special.erfc(u_lo / (np.sqrt(2.0) * vte)), rtol=1e-5)

    # Maxwell-Juttner marginal against adaptive quadrature
    def dens(x):
        return hpe_mod.tail_density_u("juttner", x, vte, c)

    total = integrate.quad(dens, 0.0, np.inf, epsabs=0.0, epsrel=1e-12, limit=500)[0]
    tail = integrate.quad(dens, u_lo, np.inf, epsabs=0.0, epsrel=1e-12, limit=500)[0]
    frac_j = hpe_mod.tail_fraction({"v_min": v_min, "tail_distribution": "juttner"}, vte, c)
    np.testing.assert_allclose(frac_j, tail / total, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_loaded_tail_follows_the_chosen_law(kind):
    from adept._lpse2d.core.hpe import load_particles, resonance_arrays

    n_p = 200000
    cfg = _make_cfg({"n_particles": n_p, "tail_distribution": kind})
    state, arrays = load_particles(cfg), resonance_arrays(cfg)
    counts = np.asarray(state["epw_hist"]) * n_p * arrays["dv"]
    expected = np.asarray(arrays["f0_expected"]) * n_p * arrays["dv"]
    np.testing.assert_allclose(counts.sum(), n_p, rtol=1e-12)
    np.testing.assert_allclose(expected.sum(), n_p, rtol=1e-12)
    use = expected >= 20.0
    dof = int(use.sum()) - 1
    chi2 = float(np.sum((counts[use] - expected[use]) ** 2 / expected[use]))
    assert chi2 < dof + 5.0 * np.sqrt(2.0 * dof), (kind, chi2, dof)

    vte, c = _vte_c(cfg)
    v = np.asarray(state["u_e"]) / np.sqrt(1.0 + (np.asarray(state["u_e"]) / c) ** 2)
    assert np.all(np.abs(v) > 2.5 * vte * (1.0 - 1e-12))
    assert np.all(np.abs(v) <= 0.99 * c * (1.0 + 1e-12))


def test_the_laws_order_the_tail_as_expected():
    """Above 50 keV longitudinal energy at 4 keV: Gaussian in u < Maxwell-Juttner < Maxwellian in v (closed-form
    ratios 0.45 and 4.6 computed independently on a fine u grid)."""
    from adept._lpse2d.core import hpe as hpe_mod

    c = 1.0
    vte = np.sqrt(4.0 / MC2_KEV)
    u50 = np.sqrt((1.0 + 50.0 / MC2_KEV) ** 2 - 1.0)
    frac = {}
    for kind in KINDS:
        u, cdf = hpe_mod._full_law_cdf(kind, vte, c, u50)
        frac[kind] = 1.0 - np.interp(u50, u, cdf)
    np.testing.assert_allclose(frac["gaussian_u"] / frac["juttner"], 0.449, rtol=0.01)
    np.testing.assert_allclose(frac["gaussian_v"] / frac["juttner"], 4.565, rtol=0.01)


@pytest.mark.parametrize("kind", ("gaussian_u", "juttner"))
def test_wall_reinjection_follows_the_flux_weighted_law(kind):
    from adept._lpse2d.core import hpe as hpe_mod

    n_p = 100000
    cfg = _make_cfg({"n_particles": n_p, "tail_distribution": kind}, {"terms.epw.boundary.x": "absorbing"})
    hpe = hpe_mod.HybridParticleEvolution(cfg)
    assert hpe.p_therm_x == 1.0
    half = n_p // 2
    x = np.full(n_p, hpe.xmax + 0.1 * hpe.dx)
    x[:half] = hpe.xmin - 0.1 * hpe.dx
    x_new, u_new, *_ = hpe._apply_boundaries(jnp.asarray(x), jnp.zeros(n_p), 0.0)
    x_new, u_new = np.asarray(x_new), np.asarray(u_new)
    assert np.all(u_new[:half] > 0.0) and np.all(u_new[half:] < 0.0)
    np.testing.assert_array_equal(x_new[:half], hpe.xmin)
    np.testing.assert_array_equal(x_new[half:], hpe.xmax)

    # independent flux-weighted law by quadrature: v f(u) du on [u(v_min), u(0.99 c)]
    vte, c = hpe.vte, hpe.c
    u_lo, u_hi = float(hpe_mod._u_of_v(hpe.v_min, c)), float(hpe_mod._u_of_v(0.99 * c, c))

    def weight(u):
        return hpe_mod.tail_density_u(kind, u, vte, c) * u / np.sqrt(1.0 + (u / c) ** 2)

    total = integrate.quad(weight, u_lo, u_hi, epsabs=0.0, epsrel=1e-11, limit=500)[0]
    speed = np.abs(u_new)
    for q in np.linspace(0.1, 0.9, 9):
        # the q-quantile of the law, by bisection on the quadrature CDF
        lo, hi = u_lo, u_lo + 20.0 * vte
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if integrate.quad(weight, u_lo, mid, epsabs=0.0, epsrel=1e-11, limit=500)[0] / total < q:
                lo = mid
            else:
                hi = mid
        frac = float(np.mean(speed <= 0.5 * (lo + hi)))
        assert abs(frac - q) < 5.0 * np.sqrt(q * (1.0 - q) / n_p), (kind, q, frac)


@pytest.mark.parametrize("kind", ("gaussian_u", "juttner"))
def test_non_default_laws_are_refused_in_2d(kind):
    with pytest.raises(ValueError, match="1-D tracker"):
        _make_cfg_2d({"tail_distribution": kind})


def test_unknown_law_is_refused():
    with pytest.raises(ValueError, match="tail_distribution"):
        _make_cfg({"tail_distribution": "maxwellian"})
