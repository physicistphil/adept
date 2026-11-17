"""Quasilinear evolution of the box-averaged electron distribution (LPSE ``qle.*``,
``QuasilinearEvolution.cpp``) -- plan 2 K.1.

The electron velocity distribution ``f(v)`` (box-averaged, on a ``nv``-point grid per
dimension over ``[-v_max, v_max]``, initially the Maxwellian of ``Te``) diffuses in the EPW
spectrum,

    df/dt = d/dv_i ( C_ij(v) df/dv_j ),

    C_ij(v) = (pi/2) (e/m)^2 sum_k k_i k_j |phi_k / N|^2 delta(w_k - k . v)

summed over the retained modes of the periodic box (``w = wpe`` cold, or the Bohm-Gross
``sqrt(wpe^2 + 3 vte^2 k^2)`` with ``thermal_correction``). The constant is derived in this
code's scaled-Gaussian units from the envelope field ``E = Re(E_env e^{-i wpe t})`` -- the +-k
pair of an envelope mode shares one phase velocity, hence pi/2 with ``|E_env,k|^2`` -- and
verified by the wave-energy / electron-heating balance (tests/test_lpse2d_cpp/test_qle.py).

Each mode's resonance is deposited onto the velocity grid as a cell average: in 1-D the point
``v = w_k / k`` with linear (cloud-in-cell) weights, in 2-D the line ``k^ . v = w_k / |k|``
(``delta(w_k - k . v) = delta(k^ . v - v_phase) / |k|``) sampled every ``dv / 2`` with bilinear
weights. Every mode therefore heats the electrons it damps, however fine the k grid. (LPSE --
and this port before -- samples the continuum form ``C(v) ~ |phi(k_res(v))|^2 / |v - v_g|`` at
the velocity nodes only; on a box whose k spacing is finer than a velocity cell's k-space
footprint ``k^2 dv / w`` most modes fell between the samples and were damped without heating
anything: 0 of 12 adjacent modes at Lx = 128 um with nv = 100.)

The explicit update (``updateVelocityDistribution_explicit``) is sub-cycled to the diffusion
limit ``dv^2 / (2 nDim max C)`` -- in conservative flux form here rather than LPSE's
``C d^2 f + (d_i C_ij) d_j f`` (see ``_flux_increment``; ``derivative_in_tensor`` is
accepted from the deck and implied) -- relaxes ``f`` towards the Maxwellian at
``p_x |v_x| / Lx + p_y |v_y| / Ly`` (particles leaving the box return thermalised with
``thermalization_probability``), clips ``f >= 0`` and renormalises the density.

With ``landau_evolution`` the Landau rate of every EPW mode follows the evolved ``f``
(``updateLandauDamping``). The slope functional

    S_k[f] = -(pi/2) wpe^3 / k^3  sum_perp dv_perp  k . grad_v f |_{v = v_phase k^ + v_perp}

(the reduced distribution along ``k``) is the non-relativistic rate of ``f``; LPSE applies it as
is. Here it is calibrated per mode to the configured fluid rate (``terms.epw.damping``: its form,
``landau_lower_threshold`` and ``landau_multiplier``),

    gamma_L(k, t) = multiplier * threshold( gamma_form(k) * S_k[f(t)] / S_k[f_M] ),

so the rate starts at the fluid one and changes by the relative change of the resonant slope,
the discretisation of ``S`` cancelling in the ratio. (The raw ``S_k`` of the non-relativistic
grid Maxwellian is 1.4-13 times the default relativistic rate at 4 keV for k lambda_D 0.40-0.20,
so the first update made the damping jump.) Modes whose resonance is off the velocity grid keep
the fluid rate (LPSE: undamped), and modes whose Maxwellian resonance lies in the unresolved
tail (``f_M(v_phase) < 1e-25 f_M(0)``) use ``S_k[f]`` uncalibrated. Every ``update_every`` EPW
steps from ``t_start``.
"""

import equinox.internal as eqxi
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jax.scipy.ndimage import map_coordinates

QLE_KEYS = ("vdf", "gamma_L")


def _qle_cfg(cfg: dict) -> dict:
    from adept._lpse2d_cpp.datamodel import QLEModel

    qle = cfg["terms"].get("qle", {}) or {}
    return {**QLEModel(**qle).model_dump(), **qle}


def initial_vdf(cfg: dict) -> np.ndarray:
    """The Maxwellian on the velocity grid, ``(nv,)`` for a quasi-1-D box, ``(nv, nv)`` in 2-D."""
    qle = _qle_cfg(cfg)
    derived = cfg["units"]["derived"]
    vte = np.sqrt(derived["vte_sq"])
    v_max = float(qle["v_max"]) * derived["c"]
    v = np.linspace(-v_max, v_max, int(qle["nv"]))
    if int(cfg["grid"]["ny"]) > 1:
        v_sq = v[:, None] ** 2 + v[None, :] ** 2
        return np.exp(-v_sq / (2.0 * vte**2)) / (2.0 * np.pi * vte**2)
    return np.exp(-(v**2) / (2.0 * vte**2)) / np.sqrt(2.0 * np.pi * vte**2)


def load_qle_state(cfg: dict) -> dict:
    from adept._lpse2d_cpp.core.epw import analytic_landau_rate

    return {"vdf": initial_vdf(cfg), "gamma_L": np.asarray(analytic_landau_rate(cfg), dtype=np.float64)}


class QuasilinearEvolution:
    def __init__(self, cfg: dict):
        qle = _qle_cfg(cfg)
        derived = cfg["units"]["derived"]
        grid = cfg["grid"]
        self.wpe = float(derived["wp0"])
        self.vte_sq = float(derived["vte_sq"])
        self.q_over_m = float(derived["e"] / derived["me"])
        self.c = float(derived["c"])
        self.nx, self.ny = int(grid["nx"]), int(grid["ny"])
        self.is_2d = self.ny > 1
        self.ndim = 2 if self.is_2d else 1
        self.dx, self.dy = float(grid["dx"]), float(grid["dy"])
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.dt = float(grid["dt"])
        tmin = grid["tmin"]
        if isinstance(tmin, str):
            from astropy.units import Quantity as _Q

            tmin = _Q(tmin).to("ps").value
        self.tmin = float(tmin)
        self.update_every = int(qle["update_every"])
        self.t_start = float(qle["t_start"])
        self.thermal_correction = bool(qle["thermal_correction"])
        self.derivative_in_tensor = bool(qle["derivative_in_tensor"])
        self.landau_evolution = bool(qle["landau_evolution"])
        self.subcycling = int(qle["subcycling"])
        self.max_subcycles = int(qle["max_subcycles"])
        self.multiplier = float(qle["multiplier"])
        self.solver = str(qle["solver"])
        self.cg_iterations = int(qle["cg_iterations"])
        p = list(qle["thermalization_probability"]) + [0.0, 0.0]
        self.p_therm = (float(p[0]), float(p[1]))

        # velocity grid
        self.nv = int(qle["nv"])
        self.v_max = float(qle["v_max"]) * self.c
        v = np.linspace(-self.v_max, self.v_max, self.nv)
        self.v = jnp.asarray(v)
        self.dv = float(v[1] - v[0])
        if self.is_2d:
            vx, vy = np.meshgrid(v, v, indexing="ij")
            self.v_nodes = jnp.asarray(np.stack([vx.ravel(), vy.ravel()], axis=-1))  # (nv^2, 2)
        else:
            self.v_nodes = jnp.asarray(v[:, None])  # (nv, 1)
        f0 = initial_vdf(cfg)
        self.f0 = jnp.asarray(f0)
        self.volume0 = float(np.sum(f0) * self.dv**self.ndim)
        # thermalisation rate on the grid
        rate = self.p_therm[0] * np.abs(v) / self.Lx
        if self.is_2d:
            rate = rate[:, None] + self.p_therm[1] * np.abs(v)[None, :] / self.Ly
        self.therm_rate = jnp.asarray(rate)

        # k grid (LPSE: kMax per axis from the anti-aliasing range)
        kx = np.asarray(grid["kx"], dtype=np.float64)
        ky = np.asarray(grid["ky"], dtype=np.float64)
        self.kx, self.ky = jnp.asarray(kx), jnp.asarray(ky)
        self.n_cells = self.nx * self.ny
        k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
        k_mag = np.sqrt(k_sq)
        omega = np.sqrt(self.wpe**2 + 3.0 * self.vte_sq * k_sq) if self.thermal_correction else self.wpe
        with np.errstate(divide="ignore", invalid="ignore"):
            v_phase = np.where(k_mag > 0, omega / np.where(k_mag > 0, k_mag, 1.0), np.inf)
        # the EPW solver's own retained band (LPSE's rounded anti-aliasing mask, lw.maxWavenumber
        # included): every mode the solver keeps both damps and diffuses the VDF
        inside = np.asarray(grid["low_pass_filter_grid"]) > 0.0

        # ---- the resonance deposit of C_ij: every retained mode, weight k_i k_j / |k|
        mode_ix, mode_iy = np.nonzero(inside & (k_mag > 0))
        mkx, mky = kx[mode_ix], ky[mode_iy]
        mk = np.sqrt(mkx**2 + mky**2)
        m_vph = v_phase[mode_ix, mode_iy]
        self.mode_index = (jnp.asarray(mode_ix), jnp.asarray(mode_iy))
        comps = [mkx * mkx, mky * mky, mkx * mky] if self.is_2d else [mkx * mkx]
        self.mode_geometry = jnp.asarray(np.stack(comps, axis=-1) / mk[:, None])
        self.deposit_constant = self.multiplier * 0.5 * np.pi * self.q_over_m**2
        if self.is_2d:
            self.mode_v_phase = jnp.asarray(m_vph)
            self.mode_khat = jnp.asarray(np.stack([mkx / mk, mky / mk], axis=-1))
            ds = self.dv / 2.0  # two samples per cell crossing
            n_s = int(np.ceil(2.0 * np.sqrt(2.0) * self.v_max / ds))
            self.line_s = jnp.asarray((np.arange(n_s) + 0.5 - 0.5 * n_s) * ds)
            self.line_ds = ds
        else:
            # the resonance v = w_k / k, linear weights on the two nodes about it (static)
            x = (m_vph * np.sign(mkx) + self.v_max) / self.dv
            i0 = np.floor(x).astype(np.int64)
            frac = x - i0
            idx = np.stack([i0, i0 + 1], axis=-1)
            w = np.stack([1.0 - frac, frac], axis=-1)
            ok = (idx >= 0) & (idx < self.nv)
            self.deposit_index = jnp.asarray(np.where(ok, idx, 0))
            self.deposit_weight = jnp.asarray(np.where(ok, w, 0.0))

        # ---- the Landau slope functional S_k[f] and its calibration to the fluid rate
        if self.is_2d:
            self.v_perp = self.v
            self.dv_plane = self.dv
        else:
            self.v_perp = jnp.zeros(1)
            self.dv_plane = 1.0
        # k validity for S: non-zero, inside the band, phase velocity on the grid
        self.k_valid = jnp.asarray((k_mag > 0) & inside & (v_phase <= self.v_max))
        self.v_phase = jnp.asarray(np.where(np.isfinite(v_phase), v_phase, 0.0))
        self.k_mag = jnp.asarray(k_mag)
        self.landau_constant = -0.5 * np.pi * self.wpe**3 * self.dv_plane
        self._init_landau_calibration(cfg, v_phase)

    def _init_landau_calibration(self, cfg: dict, v_phase: np.ndarray) -> None:
        """The per-mode calibration ``gamma_form / S_k[f_M]`` and the fluid rate's threshold and
        multiplier (``terms.epw.damping``)."""
        from adept._lpse2d_cpp.core.epw import analytic_landau_rate

        damping = cfg["terms"]["epw"]["damping"]
        if not damping.get("landau", True):
            raise ValueError(
                "terms.qle needs terms.epw.damping.landau: the quasilinear diffusion is the electron side of "
                "the Landau damping, and without it the electrons would gain energy no wave loses"
            )
        form_cfg = {
            **cfg,
            "terms": {
                **cfg["terms"],
                "epw": {
                    **cfg["terms"]["epw"],
                    # the form alone: the threshold and multiplier act on the evolved rate below
                    "damping": {**damping, "landau": True, "landau_lower_threshold": 0.0, "landau_multiplier": 1.0},
                },
            },
        }
        gamma_form = np.asarray(analytic_landau_rate(form_cfg), dtype=np.float64)
        self.landau_threshold = float(damping.get("landau_lower_threshold", 0.0) or 0.0)
        mult = damping.get("landau_multiplier")
        self.landau_multiplier = float(mult) if mult is not None else 1.0
        s_ref = np.asarray(self._slope_rate(self.f0), dtype=np.float64)
        valid = np.asarray(self.k_valid)
        with np.errstate(over="ignore", invalid="ignore"):
            resolved = np.exp(-0.5 * np.where(np.isfinite(v_phase), v_phase, 0.0) ** 2 / self.vte_sq) >= 1.0e-25
        calibrated = valid & resolved & (s_ref > 0.0)
        self.landau_calibration = jnp.asarray(np.where(calibrated, gamma_form / np.where(calibrated, s_ref, 1.0), 1.0))
        # off the velocity grid the fluid rate stays (the VDF does not represent that resonance)
        self.off_grid_rate = jnp.asarray(np.where(valid, 0.0, gamma_form))
        # without landau_evolution the applied rate is the static fluid rate: each mode's diffusion
        # carries rate / S_k[f_M], so the exchange balances on the Maxwellian
        static = np.asarray(analytic_landau_rate(cfg), dtype=np.float64)
        self.static_coupling = jnp.asarray(
            np.where(calibrated, static / np.where(calibrated, s_ref, 1.0), np.where(static > 0.0, 1.0, 0.0))
        )

    def coupling(self, f: Array) -> Array:
        """Per-mode factor ``kappa_k`` on the diffusion of mode k, ``(nx, ny)``: the applied Landau rate
        over the slope functional, ``gamma_k = kappa_k S_k[f]``. The wave then loses what the electrons
        gain (to the grid's O(dv^2) error) whatever the form, threshold and multiplier -- calibration x
        multiplier, 0 where the threshold zeroes the rate. Without ``landau_evolution``, the static
        rate's factor on the Maxwellian.

        Known limitation: with a relativistic form ``kappa_k`` is well below 1 for fast modes (Te = 2 keV:
        0.58 at k lambda_D 0.25, 0.024 at 0.15), so the non-relativistic VDF's response to them -- the
        tail flattening -- is slowed by ``1 / kappa_k``. Energy balances; the flattening rate does not
        match the relativistic plasma's until the VDF is a momentum-space Maxwell-Juttner one."""
        if not self.landau_evolution:
            return self.static_coupling
        coupling = self.landau_multiplier * self.landau_calibration
        if self.landau_threshold > 0.0:
            coupling = jnp.where(self.landau_calibration * self._slope_rate(f) > self.landau_threshold, coupling, 0.0)
        return coupling

    # ----------------------------------------------------------------- tensor --

    def _phi_sq_coef(self, phi_k: Array) -> Array:
        """``|phi_k / N|^2`` on the (nx, ny) k grid: the squared Fourier coefficient."""
        return jnp.abs(phi_k) ** 2 / self.n_cells**2

    def diffusion_tensor(self, phi_k: Array, coupling: Array | None = None) -> Array:
        """``C_ij(v)`` on the velocity grid, the cell average of every mode's resonance:
        ``(nv, nv, 3)`` [C_xx, C_yy, C_xy] in 2-D, ``(nv, 1)`` [C_xx] in 1-D. ``coupling`` (nx, ny) scales
        each mode's share (``self.coupling(f)`` in the update); None is the bare quasilinear tensor."""
        phi_sq = self._phi_sq_coef(phi_k)
        if coupling is not None:
            phi_sq = phi_sq * coupling
        weights = self.deposit_constant * phi_sq[self.mode_index][:, None] * self.mode_geometry  # (n_modes, n_comp)
        if not self.is_2d:
            c = jnp.zeros(self.nv).at[self.deposit_index.ravel()].add((self.deposit_weight * weights[:, :1]).ravel())
            return (c / self.dv)[:, None]
        nv, v_max, dv = self.nv, self.v_max, self.dv
        vph, khat = self.mode_v_phase, self.mode_khat
        perp = jnp.stack([-khat[:, 1], khat[:, 0]], axis=-1)

        def deposit(m, acc):
            s = self.line_s[m]
            x = (vph * khat[:, 0] + s * perp[:, 0] + v_max) / dv
            y = (vph * khat[:, 1] + s * perp[:, 1] + v_max) / dv
            i0, j0 = jnp.floor(x), jnp.floor(y)
            fx, fy = x - i0, y - j0
            i0, j0 = i0.astype(jnp.int32), j0.astype(jnp.int32)
            for di, dj, w in (
                (0, 0, (1 - fx) * (1 - fy)),
                (1, 0, fx * (1 - fy)),
                (0, 1, (1 - fx) * fy),
                (1, 1, fx * fy),
            ):
                ii, jj = i0 + di, j0 + dj
                ok = (ii >= 0) & (ii < nv) & (jj >= 0) & (jj < nv)
                acc = acc.at[jnp.where(ok, ii * nv + jj, 0)].add(jnp.where(ok, w, 0.0)[:, None] * weights)
            return acc

        acc = lax.fori_loop(0, int(self.line_s.shape[0]), deposit, jnp.zeros((nv * nv, 3)))
        return acc.reshape(nv, nv, 3) * (self.line_ds / dv**2)

    # ------------------------------------------------------------- FP update --

    def _flux_increment(self, f: Array, tensor: Array) -> Array:
        """``d/dv_i (C_ij d f/dv_j)`` in conservative flux form: ``C`` averaged to the cell faces,
        the cross derivative on a face as the mean of the two adjacent central differences,
        zero flux through the velocity walls. Density is conserved exactly and the update is
        stable at the diffusion limit ``dv^2 / (2 nDim max C)``. (LPSE differentiates the
        tensor into a drift ``D_j = d_i C_ij`` and steps ``C d^2 f + D . d f``; on a resonance
        narrower than a velocity cell that drift's own CFL ``dv Delta v / C`` is far below the
        diffusion limit its sub-cycling honours, and the update runs away.)"""
        dv = self.dv
        if not self.is_2d:
            c = tensor[:, 0]
            flux = 0.5 * (c[1:] + c[:-1]) * (f[1:] - f[:-1]) / dv
            flux = jnp.concatenate([jnp.zeros(1), flux, jnp.zeros(1)])
            return (flux[1:] - flux[:-1]) / dv
        cxx, cyy, cxy = tensor[..., 0], tensor[..., 1], tensor[..., 2]
        # x faces (i + 1/2, j)
        dfdy = (
            jnp.pad(f, ((0, 0), (1, 1)), mode="edge")[:, 2:] - jnp.pad(f, ((0, 0), (1, 1)), mode="edge")[:, :-2]
        ) / (2.0 * dv)
        fx = 0.5 * (cxx[1:, :] + cxx[:-1, :]) * (f[1:, :] - f[:-1, :]) / dv + 0.5 * (cxy[1:, :] + cxy[:-1, :]) * 0.5 * (
            dfdy[1:, :] + dfdy[:-1, :]
        )
        fx = jnp.pad(fx, ((1, 1), (0, 0)))
        # y faces (i, j + 1/2)
        dfdx = (
            jnp.pad(f, ((1, 1), (0, 0)), mode="edge")[2:, :] - jnp.pad(f, ((1, 1), (0, 0)), mode="edge")[:-2, :]
        ) / (2.0 * dv)
        fy = 0.5 * (cyy[:, 1:] + cyy[:, :-1]) * (f[:, 1:] - f[:, :-1]) / dv + 0.5 * (cxy[:, 1:] + cxy[:, :-1]) * 0.5 * (
            dfdx[:, 1:] + dfdx[:, :-1]
        )
        fy = jnp.pad(fy, ((0, 0), (1, 1)))
        return (fx[1:, :] - fx[:-1, :]) / dv + (fy[:, 1:] - fy[:, :-1]) / dv

    def evolve_vdf(self, f: Array, phi_k: Array, dt: float) -> Array:
        """One quasilinear update of ``f`` over ``dt`` in the spectrum ``phi_k``, each mode diffusing the
        electrons with the coupling of the Landau rate applied to it (``coupling``)."""
        tensor = self.diffusion_tensor(phi_k, self.coupling(f))
        c_max = jnp.max(jnp.abs(tensor))
        dt_max = self.dv**2 / (2.0 * self.ndim * jnp.where(c_max > 0.0, c_max, 1.0))
        n_needed = jnp.where(c_max > 0.0, jnp.ceil(dt / dt_max), 1.0) * self.subcycling
        # LPSE sub-cycles to the diffusion limit however many steps that takes ([qle:N]); the
        # trip count is data-dependent. Beyond max_subcycles -- the saturated TPD / SRS spectrum
        # on a coarse velocity grid (test_029 at 1.85 ps needed more than 20000) -- the step is
        # taken implicitly instead (backward Euler, CG)
        n_sub = jnp.clip(n_needed, 1, self.max_subcycles).astype(jnp.int32)
        dt_sub = dt / n_sub

        def relax(f_new, step):
            f_new = jnp.maximum(0.0, f_new - step * self.therm_rate * (f_new - self.f0))
            return f_new * (self.volume0 / (jnp.sum(f_new) * self.dv**self.ndim))

        def substep(carry):
            i, f = carry
            return i + 1, relax(f + dt_sub * self._flux_increment(f, tensor), dt_sub)

        def explicit(f):
            # a data-dependent trip count is a while loop, which JAX cannot transpose; the
            # checkpointed bounded loop (diffrax's RecursiveCheckpointAdjoint machinery) runs only
            # the n_sub steps forward and reverse-differentiates by recursive checkpointing, with
            # max_subcycles (beyond which the step is implicit) as its static bound
            _, f = eqxi.while_loop(
                lambda carry: carry[0] < n_sub,
                substep,
                (jnp.zeros((), jnp.int32), f),
                max_steps=self.max_subcycles,
                kind="checkpointed",
            )
            return f

        def implicit(f):
            return relax(self._implicit_step(f, tensor, dt), dt)

        if self.solver == "implicit":
            return implicit(f)
        return lax.cond(n_needed > self.max_subcycles, implicit, explicit, f)

    def _implicit_step(self, f: Array, tensor: Array, dt: float) -> Array:
        """Backward Euler ``(1 - dt L) f_new = f`` with ``L`` the conservative flux operator
        (symmetric, negative semi-definite: ``1 - dt L`` is SPD), solved matrix-free by
        conjugate gradients. Unconditionally stable, density-exact (``L`` conserves the sum);
        first order in ``dt`` (theta = 1; LPSE's implicit ADI scheme defaults to theta = 0.5 in 2-D, 0.67 in 3-D)."""
        from jax.scipy.sparse.linalg import cg

        def operator(x):
            return x - dt * self._flux_increment(x, tensor)

        f_new, _ = cg(operator, f, x0=f, tol=1e-10, atol=0.0, maxiter=self.cg_iterations)
        return f_new

    # ------------------------------------------------------------ Landau rate --

    def landau_rate(self, f: Array) -> Array:
        """``gamma_L(k)`` on the EPW k grid from the evolved ``f``: the fluid rate scaled by the
        relative change of the resonant slope, ``gamma_form S_k[f] / S_k[f_M]``, then the fluid
        rate's threshold and multiplier; the fluid rate where the resonance is off the grid."""
        rate = self.landau_calibration * self._slope_rate(f) + self.off_grid_rate
        if self.landau_threshold > 0.0:
            rate = jnp.where(rate > self.landau_threshold, rate, 0.0)
        return self.landau_multiplier * rate

    def _slope_rate(self, f: Array) -> Array:
        """``S_k[f]``, the non-relativistic Landau rate of ``f`` on the EPW k grid (LPSE's evolved
        rate; 0 where the mode's phase velocity is off the grid)."""
        # grad f = f grad(ln f): exact for a Maxwellian on any grid (a plain central difference
        # of the exponential over-estimates |f'| by 20 % at dv = 0.13 vte, LPSE's default).
        # Where a neighbour sits at the floor (the clipped edge of a plateau) ln f jumps by
        # hundreds across one cell and f grad(ln f) is meaningless: the plain difference there
        # (test_029/qle/2: a -10 wpe Landau rate at the plateau edge blew the EPW up in one step)
        floor = 1e-30 * jnp.max(f)
        log_f = jnp.log(jnp.maximum(f, floor))
        axes = (0, 1) if self.is_2d else (0,)
        parts = []
        for axis in axes:
            g_log = f * jnp.gradient(log_f, self.dv, axis=axis)
            g_lin = jnp.gradient(f, self.dv, axis=axis)
            pad = [(0, 0)] * f.ndim
            pad[axis] = (1, 1)
            fp = jnp.pad(f, pad, mode="edge")
            lo = jnp.minimum(jnp.roll(fp, 1, axis=axis), jnp.roll(fp, -1, axis=axis))
            lo = lo[tuple(slice(1, -1) if a == axis else slice(None) for a in range(f.ndim))]
            parts.append(jnp.where(jnp.minimum(lo, f) > floor, g_log, g_lin))
        grad = jnp.stack(parts, axis=-1)
        v_perp = self.v_perp

        def at_k(args):
            kx_, ky_, k_mag, v_ph = args
            k_hat = jnp.array([kx_, ky_]) / jnp.where(k_mag > 0.0, k_mag, 1.0)
            if self.is_2d:
                perp_hat = jnp.array([-k_hat[1], k_hat[0]])
                vxs = v_ph * k_hat[0] + v_perp * perp_hat[0]
                vys = v_ph * k_hat[1] + v_perp * perp_hat[1]
                inside = (jnp.abs(vxs) <= self.v_max) & (jnp.abs(vys) <= self.v_max)
                coords = [(vxs + self.v_max) / self.dv, (vys + self.v_max) / self.dv]
                gx = map_coordinates(grad[..., 0], coords, order=1, mode="constant", cval=0.0)
                gy = map_coordinates(grad[..., 1], coords, order=1, mode="constant", cval=0.0)
                k_dot = jnp.sum(jnp.where(inside, kx_ * gx + ky_ * gy, 0.0))
            else:
                vxs = v_ph * k_hat[0] * jnp.ones(1)
                inside = jnp.abs(vxs) <= self.v_max
                gx = map_coordinates(grad[:, 0], [(vxs + self.v_max) / self.dv], order=1, mode="constant", cval=0.0)
                k_dot = jnp.sum(jnp.where(inside, kx_ * gx, 0.0))
            return self.landau_constant * k_dot / jnp.where(k_mag > 0.0, k_mag, 1.0) ** 3

        kx_grid = jnp.broadcast_to(self.kx[:, None], (self.nx, self.ny)).ravel()
        ky_grid = jnp.broadcast_to(self.ky[None, :], (self.nx, self.ny)).ravel()
        gamma = lax.map(at_k, (kx_grid, ky_grid, self.k_mag.ravel(), self.v_phase.ravel())).reshape(self.nx, self.ny)
        return jnp.where(self.k_valid, gamma, 0.0)

    # -------------------------------------------------------------------- step --

    def __call__(self, t: float, y: dict) -> dict:
        step = jnp.round((t - self.tmin) / self.dt).astype(jnp.int32)
        due = jnp.logical_and(t + 0.5 * self.dt >= self.t_start, (step % self.update_every) == 0)
        f = y["vdf"]

        def update(f):
            f_new = self.evolve_vdf(f, y["epw"], self.update_every * self.dt)
            gamma = self.landau_rate(f_new) if self.landau_evolution else y["gamma_L"]
            return f_new, gamma

        f_new, gamma_new = lax.cond(due, update, lambda f: (f, y["gamma_L"]), f)
        return {**y, "vdf": f_new, "gamma_L": gamma_new}
