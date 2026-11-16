"""Absolute-threshold bisection driver (LPSE AbsoluteThreshold), with a fake runner."""

import numpy as np
import pytest


def test_bisection_converges_on_a_known_threshold():
    from adept._lpse2d_cpp.threshold import find_threshold

    true_threshold = 3.0e14

    def fake_runner(cfg, intensity, run_name):
        rate = 2.0 * (intensity / true_threshold - 1.0)
        return {"epw_growth_measurable": 1.0, "epw_growth_rate_per_ps": rate}

    cfg = {"units": {"laser intensity": "1e15W/cm^2"}, "mlflow": {"run": "t"}}
    result = find_threshold(cfg, "1e14W/cm^2", "1e15W/cm^2", n_iter=10, runner=fake_runner, log_mlflow=False)
    lo, hi = result["bracket"]
    assert lo < true_threshold < hi
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.01
    assert len(result["history"]) == 12  # two bracket runs + 10 bisections
    assert all(h[1] == (h[0] > true_threshold) for h in result["history"])


def test_bad_brackets_are_rejected():
    from adept._lpse2d_cpp.threshold import find_threshold

    def always(cfg, intensity, name):
        return {"epw_growth_measurable": 1.0, "epw_growth_rate_per_ps": 5.0}

    def never(cfg, intensity, name):
        return {"epw_growth_measurable": 0.0, "epw_growth_rate_per_ps": 0.0}

    cfg = {"units": {"laser intensity": "1e15W/cm^2"}}
    with pytest.raises(ValueError, match="already unstable"):
        find_threshold(cfg, 1e14, 1e15, n_iter=1, runner=always, log_mlflow=False)
    with pytest.raises(ValueError, match="still stable"):
        find_threshold(cfg, 1e14, 1e15, n_iter=1, runner=never, log_mlflow=False)
    assert np.isfinite(1.0)


def test_lpse_search_converges_with_the_gain_criterion():
    """LPSE's stage-1 march and stage-2 halving with the ln(A_end / A_noise) > gain criterion,
    on a fake runner whose peak amplitude grows as exp(g(I) t)."""
    from adept._lpse2d_cpp.threshold import find_threshold_lpse, gain_criterion

    i_0, gain, t_end = 3.0e14, 23.026, 2.0

    def fake_runner(cfg, intensity, run_name):
        t = np.linspace(0.0, t_end, 401)
        # exp(g t) from 0.1 ps with g = (gain / t_end) I / I_0: the gain is reached at t_end
        # for I = I_0 t_end / (t_end - 0.1)
        rate = (gain / t_end) * (intensity / i_0)
        return {"t (ps)": t, "max_rho": 1e-10 * np.exp(rate * np.where(t > 0.1, t - 0.1, 0.0))}

    true_threshold = i_0 * t_end / (t_end - 0.1)
    cfg = {"units": {"laser intensity": "1e15W/cm^2"}, "mlflow": {"run": "t"}}
    result = find_threshold_lpse(cfg, n_iter=8, runner=fake_runner)
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.02
    assert all(h[1] == gain_criterion(h[2], gain, (0.01, 0.1)) for h in result["history"])
    # a start below the threshold marches upward first
    cfg["units"]["laser intensity"] = "1e14W/cm^2"
    result = find_threshold_lpse(cfg, n_iter=8, runner=fake_runner)
    assert result["history"][1][0] > result["history"][0][0]
    assert abs(result["threshold"] - true_threshold) / true_threshold < 0.02
    # the translator's block overrides the defaults
    cfg["threshold"] = {"gain": gain, "dI_fract": 0.5, "n_iter": 8, "noise_time_range": [0.01, 0.1]}
    result = find_threshold_lpse(cfg, runner=fake_runner)
    assert abs(result["history"][1][0] - 1.5e14) < 1.0


def _lpse_update_laser_intensity_sequence(i0, d_i_fract, n_iter, is_above, max_runs=200):
    """A line-by-line transcription of LPSE's AbsoluteThreshold::updateLaserIntensity
    (AbsoluteThreshold.cpp:143-250): the intensities it runs, and the final threshold."""
    step = initial_step = d_i_fract * i0
    intensity, stage, n_two, first_above = i0, 0, 0, None
    ran = []
    while len(ran) < max_runs:
        ran.append(intensity)
        above = is_above(intensity)
        if stage == 0:
            first_above = above
            intensity += -step if above else step
            stage = 1
        elif stage == 1:
            if first_above:
                if above:
                    intensity -= step
                else:
                    stage, step = 2, step / 2
                    intensity += step
            else:
                if above:
                    stage, step = 2, step / 2
                    intensity -= step
                else:
                    intensity += step
            if stage == 1 and intensity <= initial_step / 1000:
                stage, step = 2, step / 2
                intensity = step
        else:
            n_two += 1
            step /= 2
            intensity += -step if above else step
            if intensity <= 0:
                n_two += 1
                step /= 2
                intensity = step
            if n_two >= n_iter:
                return ran, intensity
    raise AssertionError("reference search did not finish")


@pytest.mark.parametrize("true_threshold", [5.0e13, 1.3e14, 1.6e14, 2.0e14, 3.0e14, 7.0e14, 1.2e15, 2.5e15])
def test_lpse_search_runs_the_same_intensities_as_lpse(true_threshold):
    """Same deck defaults (dI_fract 0.3333, 5 iterations) and a step-function criterion: the search must
    run LPSE's intensities and return LPSE's threshold (rtol 1e-12), including thresholds below I0 / 6,
    which the old "intensity too small" exit could not reach (it re-tested I0 / 3)."""
    from adept._lpse2d_cpp.threshold import find_threshold_lpse

    i0, gain = 1.0e15, 23.026
    t = np.linspace(0.0, 1.0, 201)

    def fake_runner(cfg, intensity, run_name):
        rate = 30.0 if intensity > true_threshold else 0.0  # ln(A_end / A_noise) = 27 > gain, or 0
        return {"t (ps)": t, "max_rho": 1e-10 * np.exp(rate * np.where(t > 0.1, t - 0.1, 0.0))}

    cfg = {"units": {"laser intensity": f"{i0}W/cm^2"}, "mlflow": {"run": "t"}}
    result = find_threshold_lpse(cfg, gain=gain, runner=fake_runner)
    ran, expected = _lpse_update_laser_intensity_sequence(i0, 0.3333, 5, lambda i: i > true_threshold)
    np.testing.assert_allclose([h[0] for h in result["history"]], ran, rtol=1e-12)
    np.testing.assert_allclose(result["threshold"], expected, rtol=1e-12)


def test_gain_criterion_is_lpse_running_check():
    """Inventory A18 / A29 (AbsoluteThreshold::checkIsAboveThreshold): samples before noiseTimeRange[0]
    are skipped, the noise level is the running mean over [t0, t1), and the first sample with
    ln(A / A_noise) > gain ends the run above threshold -- a later drop does not undo it."""
    from adept._lpse2d_cpp.threshold import gain_criterion

    t = np.linspace(0.0, 1.0, 101)
    base = np.full_like(t, 1.0e-10)
    # crosses e^5 x the noise at 0.5 ps, back to the noise level by the end
    bump = base * np.where((t > 0.45) & (t < 0.6), np.exp(6.0), 1.0)
    assert gain_criterion({"t (ps)": t, "max_rho": bump}, gain=5.0, noise_time_range=(0.01, 0.1))
    assert not gain_criterion({"t (ps)": t, "max_rho": base}, gain=5.0, noise_time_range=(0.01, 0.1))
    # a spike before t0 is not part of the noise level (it would have lifted it by 1e3 / 10)
    spiked = base.copy()
    spiked[0] = 1.0e-7
    growth = spiked * np.where(t > 0.5, np.exp(6.0), 1.0)
    assert gain_criterion({"t (ps)": t, "max_rho": growth}, gain=5.0, noise_time_range=(0.01, 0.1))
    with pytest.raises(ValueError, match="max_rho"):
        gain_criterion({"t (ps)": t, "max_phi": base}, gain=5.0)


def test_a_zero_lower_bound_is_rejected_before_any_run():
    """The geometric midpoint sqrt(lo hi) of a zero lower bound never moves, so the bracket would
    never narrow and the reported threshold would be 0 (PR #372 review): the bracket needs
    0 < intensity_lo < intensity_hi, checked before a single run; a positive bracket still bisects."""
    from adept._lpse2d_cpp.threshold import find_threshold

    calls = []

    def runner(cfg, intensity, name):
        calls.append(intensity)
        return {"epw_growth_measurable": 1.0, "epw_growth_rate_per_ps": 2.0 * (intensity / 3.0e14 - 1.0)}

    cfg = {"units": {"laser intensity": "1e15W/cm^2"}, "mlflow": {"run": "t"}}
    for lo in (0.0, "0W/cm^2", -1.0e14):
        with pytest.raises(ValueError, match="0 < intensity_lo"):
            find_threshold(cfg, lo, 1e15, n_iter=3, runner=runner, log_mlflow=False)
    assert calls == []
    result = find_threshold(cfg, 1e13, 1e15, n_iter=12, runner=runner, log_mlflow=False)
    assert abs(result["threshold"] - 3.0e14) / 3.0e14 < 0.01


def test_lpse_search_refuses_a_zero_start_before_any_run():
    """find_threshold_lpse steps by dI_fract * intensity, so a zero (or negative) start never moves and
    stage 1 would re-run the same point until its step budget ran out: it is refused before a run."""
    from adept._lpse2d_cpp.threshold import find_threshold_lpse

    calls = []

    def runner(cfg, intensity, run_name):
        calls.append(intensity)
        return {"t (ps)": np.linspace(0.0, 1.0, 11), "max_rho": np.full(11, 1e-10)}

    for start in ("0W/cm^2", "-1e14W/cm^2"):
        cfg = {"units": {"laser intensity": start}, "mlflow": {"run": "t"}}
        with pytest.raises(ValueError, match="positive starting intensity"):
            find_threshold_lpse(cfg, n_iter=2, runner=runner)
    assert calls == []
