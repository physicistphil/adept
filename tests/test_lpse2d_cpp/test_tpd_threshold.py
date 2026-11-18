"""The TPD absolute threshold in a linear density profile brackets Simon et al. (1983).

Simon, Short, Williams & Dewandre, Phys. Fluids 26, 3107 (1983), Eq. (39) in practical units:
``I_th = 81.86 Te[keV] / (L[um] lambda[um]) x 1e14 W/cm^2``. Written out here, independently of
``calc_threshold_intensity``.

Two runs of ``configs/tpd.yaml`` (linear ramp 0.23-0.32 n_c, L = 200 um, Te = 2 keV, 351 nm, 15 ps, thermal
noise with a fixed seed): at half the threshold the EPW energy only reaches a convective, noise-driven
level; at twice the threshold it grows exponentially (absolute instability).

Criteria, fixed before the first run, on ``r = log10(e_sq(t_end) / e_sq(t_end - 5 ps))``:
  * below (0.5 I_th): ``r < 0.5`` -- no sustained growth;
  * above (2.0 I_th): ``r > 2`` -- more than two decades of growth in the last 5 ps.
The bracket is a factor of 2 on either side because Simon's threshold leaves out Landau damping and the
finite box; the test checks that the solver's TPD threshold lies inside it, not where.
GPU only (two 7500-step 2-D runs).
"""

import numpy as np
import pytest
from jax import devices

LAMBDA_UM = 0.351
L_UM = 200.0
TE_KEV = 2.0


def simon_threshold_w_cm2(te_kev: float, l_um: float, lambda_um: float) -> float:
    return 81.86 * te_kev / (l_um * lambda_um) * 1e14


def log10_growth_last_5ps(intensity_w_cm2: float, run_name: str) -> float:
    import yaml

    from adept import ergoExo

    with open("tests/test_lpse2d_cpp/configs/tpd.yaml") as fi:
        cfg = yaml.safe_load(fi)
    cfg["units"]["laser intensity"] = f"{intensity_w_cm2:.6g}W/cm^2"
    cfg["units"]["reference electron temperature"] = f"{TE_KEV}keV"
    cfg["density"]["gradient scale length"] = f"{L_UM}um"
    cfg["drivers"]["E0"]["params"] = {"phases": {"seed": 1}}
    cfg["terms"]["epw"]["source"]["noise_seed"] = 7
    cfg["mlflow"]["experiment"] = "test-lpse2d-tpd-threshold"
    cfg["mlflow"]["run"] = run_name

    exo = ergoExo()
    _, ppo, _ = exo(exo.setup(cfg))
    t = np.asarray(ppo["series"]["t (ps)"].values)
    e_sq = np.asarray(ppo["series"]["e_sq"].values)
    i_late = int(np.argmin(np.abs(t - (t[-1] - 5.0))))
    # the EPW starts from zero (e_sq(0) = 0); the criterion needs a finite, positive energy in its window
    assert np.all(np.isfinite(e_sq)) and np.all(e_sq[i_late:] > 0.0)
    return float(np.log10(e_sq[-1] / e_sq[i_late]))


def test_tpd_threshold_brackets_simon():
    if not any(device.platform == "gpu" for device in devices()):
        pytest.skip("two 7500-step 2-D runs: GPU only")
    i_th = simon_threshold_w_cm2(TE_KEV, L_UM, LAMBDA_UM)
    below = log10_growth_last_5ps(0.5 * i_th, "below-0.5x-simon")
    above = log10_growth_last_5ps(2.0 * i_th, "above-2x-simon")
    assert below < 0.5, f"0.5 x Simon threshold: EPW energy grew {below:.2f} decades in the last 5 ps"
    assert above > 2.0, f"2 x Simon threshold: EPW energy grew only {above:.2f} decades in the last 5 ps"
