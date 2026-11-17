"""The EPW collision rate of ``terms.epw.damping.collisions: true`` follows LPSE's formula.

``write_units`` computes the electron-ion collision rate at the envelope density with LPSE's NRL
Coulomb logarithm (``ParameterManager.cpp:1461-1487``). Physics review 2026-11-16, finding B22: the
branch test compared T_e in eV with LPSE's keV condition and LPSE's floor ``max(2, lnLambda)`` was
missing.
"""

import numpy as np
import pytest


def _lpse_nu_eo_per_ps(te_kev: float, z: float, n_env: float, lambda_um: float) -> float:
    """Transcription of ParameterManager.cpp:1148-1150 (Nc) and 1461-1487 (nu_eo), in float64."""
    c_cgs, qe_cgs, me_cgs = 2.9979245e10, 4.8032068e-10, 9.1093829e-28
    mec2_kev = 510.99895
    laser_frequency = 2.0 * np.pi * c_cgs / (lambda_um * 1.0e-4)
    nc = laser_frequency**2 * me_cgs / (4.0 * np.pi * qe_cgs**2)
    no = n_env * nc
    me = mec2_kev / c_cgs**2
    r0 = 2.8179e-13
    e_sq = me * c_cgs**2 * r0
    ni = no / z
    if te_kev < 0.01 * z**2:
        log_lambda = 22.8487 - np.log(np.sqrt(no) * z / (te_kev * 1000.0) ** 1.5)
    else:
        log_lambda = 24.0 - np.log(np.sqrt(no) / (te_kev * 1000.0))
    log_lambda = max(2.0, log_lambda)
    nu_eo = 4.0 * np.sqrt(2.0 * np.pi) / 3.0 * e_sq**2 / np.sqrt(me) * z**2 * ni * log_lambda / te_kev**1.5
    return nu_eo * 0.5 * 1.0e-12


@pytest.mark.parametrize(
    "te_ev, z",
    [
        (2000.0, 1.0),  # the high-temperature branch, lnLambda ~ 7.6 (unchanged by the fix)
        (50.0, 3.0),  # T_e < 10 Z^2 eV: LPSE's low-temperature branch (the old code took the other one)
        (10.0, 1.0),  # lnLambda = 1.8 from the formula: LPSE's floor of 2 applies
    ],
)
def test_collision_rate_matches_lpse(te_ev, z):
    """nu_coll equals the transcribed C++ rate to rtol 1e-6 (fixed before the run: the constants of the
    two codes agree to ~1e-8 relative; the physics differences the fix removes are 9 % in lnLambda for the
    second and third cases)."""
    from adept._lpse2d_cpp.helpers import write_units

    n_env, lambda_um = 0.25, 0.351
    cfg = {
        "units": {
            "reference electron temperature": f"{te_ev}eV",
            "reference ion temperature": f"{te_ev / 2}eV",
            "ionization state": z,
            "atomic number": 2.0 * z,
            "laser_wavelength": f"{lambda_um}um",
            "laser intensity": "1e14W/cm^2",
            "envelope density": n_env,
        },
        "terms": {"epw": {"damping": {"collisions": True}}},
    }
    write_units(cfg)
    got = cfg["units"]["derived"]["nu_coll"]
    want = _lpse_nu_eo_per_ps(te_ev / 1000.0, z, n_env, lambda_um)
    np.testing.assert_allclose(got, want, rtol=1e-6)
