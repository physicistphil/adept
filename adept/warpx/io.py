r"""Loaders and converters for WarpX output.

WarpX writes SI everywhere: full diagnostics as openPMD (here always the
HDF5 backend, file-based encoding — ``diags/<diag>/openpmd_%06T.h5``) and
reduced diagnostics as whitespace-delimited text tables under
``diags/reducedfiles/``. This module reads both with plain ``h5py`` /
line parsing (no openpmd-api dependency) and converts them into the same
per-diagnostic NetCDF contract the OSIRIS wrapper emits under ``binary/``:

- one ``.nc`` per diagnostic, keyed like the OSIRIS ``MS/`` tree
  (``FLD/e1``, ``DENSITY/electrons/charge``, ``RAW/electrons``,
  ``HIST/energy``), holding the stacked ``(t, x1)`` time history;
- data in **code units** (time in ``1/ω_p``, length in ``c/ω_p``, fields
  in ``m_e c ω_p / e``, …) fixed by the manifest reference density, with
  the same dim names / attrs (``axis_units``, ``sim.XMIN`` …) OSIRIS
  series carry.

Because the contract matches, :func:`adept.osiris.io.list_diagnostics` /
``load_series`` read these files unchanged, and the OSIRIS canned-plot set
(:mod:`adept.osiris.plots`) renders WarpX runs as-is — that is the whole
point (see ``dev_docs/warpx-wrapper-plan.md``, M2).

Axis convention (1D): the WarpX axis is ``z``; the OSIRIS-comparison
mapping is the cyclic relabeling ``(z, x, y) → (1, 2, 3)``:

    ``E_z → e1``  ``E_x → e2``  ``E_y → e3``
    ``B_z → b1``  ``B_x → b2``  ``B_y → b3``   (likewise ``j``)

which preserves handedness, so OSIRIS sign conventions — including the
left/right-going Riemann pairs ``(e2, b3)`` / ``(e3, b2)`` — carry over
verbatim. Multi-D output keeps WarpX-native names and axis labels (no
OSIRIS mapping is defined for it here).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

# SI constants (CODATA, matching scipy.constants to the digits WarpX uses).
_C = 299792458.0
_QE = 1.602176634e-19
_ME = 9.1093837015e-31
_EPS0 = 8.8541878128e-12

# Diagnostic *data* is stored float32 in the saved NetCDFs (same convention
# as the OSIRIS wrapper: halves artifact size; coordinates stay float64).
_DIAG_DTYPE = "float32"

_OPENPMD_H5_RE = re.compile(r"^openpmd_(\d+)\.h5$")


# --- code units -------------------------------------------------------------


@dataclass(frozen=True)
class CodeUnits:
    """The OSIRIS-style normalization, fully determined by a reference density.

    ``n0`` is the reference electron density in 1/m^3; everything else
    derives from it: ``wp0 = sqrt(n0 e^2 / (eps0 m_e))``, lengths in the
    skin depth ``x0 = c/wp0``, fields in ``E0 = m_e c wp0 / e`` (and
    ``B0 = E0/c``), current in ``j0 = e n0 c``, charge density in
    ``rho0 = e n0``. The 1D energy-per-transverse-area unit is
    ``u_area0 = m_e c^2 n0 x0`` (equivalently ``eps0 E0^2 x0``), chosen so
    ``U_code = 0.5 * ∫ E_code^2 dx_code`` matches the OSIRIS field-energy
    convention.
    """

    n0: float  # 1/m^3
    wp0: float = field(init=False)  # rad/s
    x0: float = field(init=False)  # m
    t0: float = field(init=False)  # s
    E0: float = field(init=False)  # V/m
    B0: float = field(init=False)  # T
    j0: float = field(init=False)  # A/m^2
    rho0: float = field(init=False)  # C/m^3
    u_area0: float = field(init=False)  # J/m^2

    def __post_init__(self) -> None:
        wp0 = math.sqrt(self.n0 * _QE**2 / (_EPS0 * _ME))
        object.__setattr__(self, "wp0", wp0)
        object.__setattr__(self, "x0", _C / wp0)
        object.__setattr__(self, "t0", 1.0 / wp0)
        object.__setattr__(self, "E0", _ME * _C * wp0 / _QE)
        object.__setattr__(self, "B0", _ME * wp0 / _QE)
        object.__setattr__(self, "j0", _QE * self.n0 * _C)
        object.__setattr__(self, "rho0", _QE * self.n0)
        object.__setattr__(self, "u_area0", _ME * _C**2 * self.n0 * (_C / wp0))

    @classmethod
    def from_density_cc(cls, n0_cc: float) -> CodeUnits:
        return cls(n0=float(n0_cc) * 1.0e6)

    @classmethod
    def from_cfg(cls, cfg: dict) -> CodeUnits | None:
        """Rebuild the normalization from a run config's derived units.

        Reads ``cfg["units"]["derived"]["n0"]`` — a pint Quantity in the live
        post-processing path, or its string form when the config came back off
        a ``units.yaml`` artifact. Returns ``None`` when no units were derived
        (no reference density anywhere), in which case conversion falls back
        to SI passthrough.
        """
        n0 = ((cfg.get("units") or {}).get("derived") or {}).get("n0")
        if n0 is None:
            return None
        if hasattr(n0, "to"):  # pint Quantity
            return cls(n0=float(n0.to("1/m^3").magnitude))
        if isinstance(n0, str):
            from adept.normalization import UREG

            return cls(n0=float(UREG.Quantity(n0).to("1/m^3").magnitude))
        return cls(n0=float(n0) * 1.0e6)  # bare number: cm^-3 convention


def code_units_for_run(src: str | Path) -> CodeUnits | None:
    """Best-effort normalization for an already-finished run's artifacts.

    For offline regeneration: looks for ``units.yaml`` (the adept artifact
    dumping ``write_units``'s output) at ``src`` or its parent, then for
    ``config.yaml`` with ``warpx.reference_density``, then for the run's
    rendered ``inputs`` deck with ``my_constants.n0`` (SI m^-3). Returns
    ``None`` when nothing usable is found.
    """
    import yaml

    from adept.normalization import UREG

    src = Path(src)
    dirs = (src, src.parent)
    for d in dirs:
        p = d / "units.yaml"
        if p.is_file():
            try:
                units = yaml.safe_load(p.read_text()) or {}
                n0 = units.get("derived", units).get("n0")
                if n0 is not None:
                    return CodeUnits(n0=float(UREG.Quantity(str(n0)).to("1/m^3").magnitude))
            except Exception:
                pass
    for d in dirs:
        p = d / "config.yaml"
        if p.is_file():
            try:
                cfg = yaml.safe_load(p.read_text()) or {}
                n0 = (cfg.get("warpx") or {}).get("reference_density")
                if n0 is not None:
                    q = UREG.Quantity(n0) if isinstance(n0, str) else UREG.Quantity(float(n0), "1/cc")
                    if q.dimensionless:
                        q = q.magnitude * UREG.cc**-1
                    return CodeUnits(n0=float(q.to("1/m^3").magnitude))
            except Exception:
                pass
    for d in dirs:
        p = d / "inputs"
        if p.is_file():
            try:
                from adept.warpx import deck as _deck

                n0 = _deck.parse_deck_file(p).get("my_constants.n0")
                if n0 is not None:
                    return CodeUnits(n0=float(n0))
            except Exception:
                pass
    return None


# --- OSIRIS-comparison naming ----------------------------------------------

# 1D cyclic axis relabeling (z, x, y) -> (1, 2, 3); see module docstring.
_COMPONENT_1D = {"z": "1", "x": "2", "y": "3"}

# Per-record conversion + labels. ``scale`` names the CodeUnits attribute
# that divides the SI values; ``units`` is the OSIRIS-style TeX unit label.
_MESH_KINDS = {
    "E": {"osiris": "e", "scale": "E0", "units": r"m_e c \omega_p e^{-1}", "long": "E"},
    "B": {"osiris": "b", "scale": "B0", "units": r"m_e \omega_p e^{-1}", "long": "B"},
    "j": {"osiris": "j", "scale": "j0", "units": r"e n_0 c", "long": "j"},
}


def _mesh_diag_key(mesh: str, comp: str | None, one_d: bool) -> tuple[str, dict[str, Any]] | None:
    """Map an openPMD mesh (+ component) to an OSIRIS-contract diagnostic key.

    Returns ``(relpath, info)`` where ``info`` carries the conversion scale
    attribute name and label strings, or ``None`` for records this layer does
    not convert (e.g. multi-D fields keep native names via the fallback in
    the caller).
    """
    if mesh in _MESH_KINDS and comp is not None:
        kind = _MESH_KINDS[mesh]
        if one_d and comp in _COMPONENT_1D:
            name = f"{kind['osiris']}{_COMPONENT_1D[comp]}"
            long_name = f"{kind['long']}_{_COMPONENT_1D[comp]}"
        else:
            name = f"{mesh}{comp}"
            long_name = f"{mesh}_{comp}"
        return f"FLD/{name}", {"scale": kind["scale"], "units": kind["units"], "long_name": long_name, "name": name}
    if mesh == "rho" and comp is None:
        return "DENSITY/total/charge", {
            "scale": "rho0",
            "units": r"e n_0",
            "long_name": r"\rho",
            "name": "charge",
        }
    if mesh.startswith("rho_") and comp is None:
        species = mesh[len("rho_") :]
        return f"DENSITY/{species}/charge", {
            "scale": "rho0",
            "units": r"e n_0",
            "long_name": r"\rho_{" + species + "}",
            "name": "charge",
        }
    return None


# --- openPMD (h5) reading ---------------------------------------------------


def _openpmd_files(diag_dir: Path) -> list[Path]:
    return sorted(
        (p for p in diag_dir.iterdir() if p.is_file() and _OPENPMD_H5_RE.match(p.name)),
        key=lambda p: int(_OPENPMD_H5_RE.match(p.name).group(1)),
    )


def _decode(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.flat[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _iteration_groups(f: h5py.File) -> list[tuple[int, h5py.Group]]:
    """``(iteration, group)`` pairs in a file, sorted (handles group-based too)."""
    data = f.get("data")
    if data is None:
        return []
    out = []
    for k in data.keys():
        try:
            out.append((int(k), data[k]))
        except ValueError:
            continue
    return sorted(out, key=lambda kv: kv[0])


def _record_components(rec: h5py.Group | h5py.Dataset) -> dict[str | None, h5py.Dataset | h5py.Group]:
    """Map component name -> dataset/constant-group for an openPMD record.

    A scalar record (``rho``) is the dataset itself (key ``None``); a vector
    record (``E``) is a group of components. Constant components are groups
    carrying ``value``/``shape`` attrs instead of data.
    """
    if isinstance(rec, h5py.Dataset):
        return {None: rec}
    out: dict[str | None, h5py.Dataset | h5py.Group] = {}
    for k in rec.keys():
        out[k] = rec[k]
    return out if out else {None: rec}


def _read_component(node: h5py.Dataset | h5py.Group) -> tuple[np.ndarray, float]:
    """Values (native shape) and unitSI for a (possibly constant) component."""
    unit_si = float(node.attrs.get("unitSI", 1.0))
    if isinstance(node, h5py.Dataset):
        return node[...], unit_si
    # Constant record component: attrs value + shape.
    shape = tuple(int(s) for s in np.atleast_1d(node.attrs.get("shape", [1])))
    value = node.attrs.get("value", 0.0)
    return np.full(shape, value), unit_si


def _mesh_axes(mesh: h5py.Group | h5py.Dataset, n_cells: tuple[int, ...]) -> list[dict]:
    """Per-axis dicts (label, SI coordinates) for a mesh record.

    Coordinates are node positions ``(offset + (i + stagger) * spacing) *
    gridUnitSI`` in the order of ``axisLabels`` (matching the C-order data
    layout WarpX writes).
    """
    attrs = mesh.attrs
    labels = [_decode(x) for x in np.atleast_1d(attrs.get("axisLabels", [b"z"]))]
    spacing = np.atleast_1d(attrs.get("gridSpacing", [1.0])).astype(float)
    offset = np.atleast_1d(attrs.get("gridGlobalOffset", [0.0])).astype(float)
    grid_unit = float(attrs.get("gridUnitSI", 1.0))
    axes = []
    for d, label in enumerate(labels):
        n = n_cells[d] if d < len(n_cells) else 1
        axes.append(
            {
                "label": label,
                "coords": (offset[d] + (np.arange(n) + 0.0) * spacing[d]) * grid_unit,
                "spacing": spacing[d] * grid_unit,
            }
        )
    return axes


def _component_stagger(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    return np.atleast_1d(node.attrs.get("position", [0.0])).astype(float)


def list_field_records(diag_dir: str | Path) -> list[tuple[str, str | None]]:
    """``(mesh, component)`` pairs present in a full diagnostic's first dump."""
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        return []
    out: list[tuple[str, str | None]] = []
    with h5py.File(files[0], "r") as f:
        meshes_path = _decode(f.attrs.get("meshesPath", "fields/")).strip("/")
        for _it, grp in _iteration_groups(f)[:1]:
            meshes = grp.get(meshes_path)
            if meshes is None:
                continue
            for mesh_name in meshes.keys():
                for comp in _record_components(meshes[mesh_name]):
                    out.append((mesh_name, comp))
    return out


def load_field_series(
    diag_dir: str | Path,
    mesh: str,
    comp: str | None = None,
    *,
    units: CodeUnits | None = None,
) -> xr.DataArray:
    """Stack one mesh component's time history into a ``(t, x1)`` DataArray.

    Values and axes are converted to code units when ``units`` is given
    (fields via the record's scale, coordinates via ``x0``, time via
    ``wp0``); otherwise everything stays SI and the attrs say so. The
    returned array follows the OSIRIS series conventions (dims ``(t, x1)``
    in 1D, coords ``t``/``iter``, attrs ``axis_units`` / ``sim.XMIN`` / …)
    so downstream plotting treats it like any OSIRIS diagnostic.
    """
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        raise FileNotFoundError(f"No openpmd_*.h5 files in {diag_dir}")

    slabs: list[np.ndarray] = []
    times: list[float] = []
    iters: list[int] = []
    axes: list[dict] | None = None
    unit_si = 1.0
    one_d = True
    for path in files:
        with h5py.File(path, "r") as f:
            meshes_path = _decode(f.attrs.get("meshesPath", "fields/")).strip("/")
            for it, grp in _iteration_groups(f):
                meshes = grp.get(meshes_path)
                if meshes is None or mesh not in meshes:
                    continue
                rec = meshes[mesh]
                comps = _record_components(rec)
                if comp not in comps:
                    raise KeyError(f"component {comp!r} not in mesh {mesh!r} ({sorted(comps)})")
                arr, unit_si = _read_component(comps[comp])
                if axes is None:
                    axes = _mesh_axes(rec, arr.shape)
                    stag = _component_stagger(comps[comp])
                    for d, ax in enumerate(axes):
                        s = stag[d] if d < stag.size else 0.0
                        ax["coords"] = ax["coords"] + s * ax["spacing"]
                    one_d = len(axes) == 1
                slabs.append(np.asarray(arr))
                t_si = float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))
                times.append(t_si)
                iters.append(int(it))
    if not slabs or axes is None:
        raise FileNotFoundError(f"mesh {mesh!r} (comp {comp!r}) not found in {diag_dir}")

    order = np.argsort(iters, kind="stable")
    data = np.stack([slabs[i] for i in order]).astype(_DIAG_DTYPE)
    t = np.asarray(times, dtype="float64")[order]
    its = np.asarray(iters, dtype="int64")[order]

    key_info = _mesh_diag_key(mesh, comp, one_d)
    if key_info is not None:
        _rel, info = key_info
        scale = getattr(units, info["scale"]) if units is not None else 1.0
        name, long_name, val_units = info["name"], info["long_name"], info["units"]
    else:
        scale = 1.0
        name = mesh if comp is None else f"{mesh}{comp}"
        long_name = name
        val_units = "SI"
    if units is not None:
        data = data / np.asarray(scale, dtype=_DIAG_DTYPE)
        t = t * units.wp0
    else:
        val_units = "SI"

    coords: dict[str, Any] = {"t": t, "iter": ("t", its)}
    dims = ["t"]
    axis_units: dict[str, str] = {}
    axis_long: dict[str, str] = {}
    xmin: list[float] = []
    xmax: list[float] = []
    nx: list[int] = []
    for d, ax in enumerate(axes):
        if one_d and ax["label"] == "z":
            dim = "x1"
            axis_long[dim] = "x_1"
        else:
            dim = ax["label"]
            axis_long[dim] = ax["label"]
        cv = np.asarray(ax["coords"], dtype="float64")
        if units is not None:
            cv = cv / units.x0
            axis_units[dim] = r"c / \omega_p"
        else:
            axis_units[dim] = "m"
        coords[dim] = cv
        dims.append(dim)
        xmin.append(float(cv[0]))
        xmax.append(float(cv[-1]))
        nx.append(int(cv.size))

    attrs = {
        "long_name": long_name,
        "units": val_units,
        "time_units": r"1/\omega_p" if units is not None else "s",
        "axis_units": axis_units,
        "axis_long_names": axis_long,
        "source_dir": str(diag_dir),
        "warpx_record": mesh if comp is None else f"{mesh}/{comp}",
        "unit_si": float(unit_si) * (float(np.asarray(scale)) if units is not None else 1.0),
        "sim.NDIMS": len(axes),
        "sim.XMIN": xmin,
        "sim.XMAX": xmax,
        "sim.NX": nx,
    }
    return xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=attrs)


def load_particle_species(
    diag_dir: str | Path,
    species: str,
    *,
    units: CodeUnits | None = None,
) -> xr.Dataset:
    """Concatenate one species' particle dumps long-form (the RAW analog).

    Every dump's particles are concatenated along ``pidx`` with per-row
    ``t`` / ``iter`` coordinates (particle counts vary dump-to-dump, so a
    rectangular stack is impossible — same layout as
    :func:`adept.osiris.io.load_raw_series`). Variables, in code units when
    ``units`` is given:

    - ``x1`` (1D): position + positionOffset along z, in ``c/ω_p``;
    - ``p1``/``p2``/``p3``: momentum (z, x, y — the cyclic OSIRIS mapping)
      in units of ``m_s c`` for the species mass ``m_s``;
    - ``ene``: ``γ - 1`` (kinetic energy in ``m_s c^2``);
    - ``q``: signed macroparticle charge ``(q_s/e) · w / (n0 · x0)``,
      dimensionless — depositing ``q`` over cells of size ``Δx1`` recovers
      the charge density in ``e n_0`` units;
    - ``w``: the raw openPMD weighting (physical particles per macroparticle,
      per m^2 of transverse area in 1D).

    Without ``units``, positions stay in m, momenta in kg·m/s, and ``q`` is
    omitted.
    """
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        raise FileNotFoundError(f"No openpmd_*.h5 files in {diag_dir}")

    rows: dict[str, list[np.ndarray]] = {}
    times: list[np.ndarray] = []
    iters: list[np.ndarray] = []
    n_dumps = 0
    for path in files:
        with h5py.File(path, "r") as f:
            ppath = _decode(f.attrs.get("particlesPath", "particles/")).strip("/")
            for it, grp in _iteration_groups(f):
                parts = grp.get(ppath)
                if parts is None or species not in parts:
                    continue
                sp = parts[species]
                rec = _species_records(sp, units)
                if rec is None:
                    continue
                n = next(len(v) for v in rec.values())
                n_dumps += 1
                for k, v in rec.items():
                    rows.setdefault(k, []).append(v)
                t_si = float(grp.attrs.get("time", 0.0)) * float(grp.attrs.get("timeUnitSI", 1.0))
                times.append(np.full(n, t_si * (units.wp0 if units is not None else 1.0), dtype="float64"))
                iters.append(np.full(n, int(it), dtype="int64"))
    if n_dumps == 0:
        raise FileNotFoundError(f"species {species!r} not found in {diag_dir}")

    data_vars = {k: ("pidx", np.concatenate(v).astype(_DIAG_DTYPE)) for k, v in rows.items()}
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(t=("pidx", np.concatenate(times)), iter=("pidx", np.concatenate(iters)))
    ds.attrs.update(
        long_name=f"{species} particles",
        time_units=r"1/\omega_p" if units is not None else "s",
        source_dir=str(diag_dir),
        n_dumps=n_dumps,
    )
    return ds


def _species_records(sp: h5py.Group, units: CodeUnits | None) -> dict[str, np.ndarray] | None:
    """Extract one dump's per-particle arrays for a species (SI or code units)."""
    if "position" not in sp:
        return None
    pos = _record_components(sp["position"])
    off = _record_components(sp["positionOffset"]) if "positionOffset" in sp else {}
    out: dict[str, np.ndarray] = {}

    # Number of particles: from any non-constant dataset, else constant shape.
    n = None
    for node in list(pos.values()) + list(off.values()):
        arr, _u = _read_component(node)
        n = max(n or 0, arr.size)
    if not n:
        return None

    def full(node) -> np.ndarray:
        arr, u = _read_component(node)
        arr = np.asarray(arr, dtype="float64") * u
        return np.broadcast_to(arr, (n,)) if arr.size != n else arr.reshape(n)

    # 1D: z is the only stored axis; the cyclic mapping puts it on x1.
    for axis, name in (("z", "x1"), ("x", "x2"), ("y", "x3")):
        if axis in pos and isinstance(pos[axis], h5py.Dataset):
            x = full(pos[axis])
            if axis in off:
                x = x + full(off[axis])
            out[name] = x / units.x0 if units is not None else x

    mass = None
    if "mass" in sp:
        m_arr, m_u = _read_component(_record_components(sp["mass"])[None])
        mass = float(np.asarray(m_arr).flat[0]) * m_u
    if "momentum" in sp:
        mom = _record_components(sp["momentum"])
        p_scale = (mass or _ME) * _C
        usq = None
        for axis, name in (("z", "p1"), ("x", "p2"), ("y", "p3")):
            if axis in mom:
                p = full(mom[axis])
                pc = p / p_scale
                out[name] = pc if units is not None else p
                usq = pc**2 if usq is None else usq + pc**2
        if usq is not None:
            out["ene"] = np.sqrt(1.0 + usq) - 1.0

    if "weighting" in sp:
        w = full(_record_components(sp["weighting"])[None])
        out["w"] = w
        if units is not None and "charge" in sp:
            q_arr, q_u = _read_component(_record_components(sp["charge"])[None])
            q_over_e = float(np.asarray(q_arr).flat[0]) * q_u / _QE
            out["q"] = q_over_e * w / (units.n0 * units.x0)
    return out


def list_species(diag_dir: str | Path) -> list[str]:
    """Species names present in a full diagnostic's first dump."""
    diag_dir = Path(diag_dir)
    files = _openpmd_files(diag_dir)
    if not files:
        return []
    with h5py.File(files[0], "r") as f:
        ppath = _decode(f.attrs.get("particlesPath", "particles/")).strip("/")
        for _it, grp in _iteration_groups(f)[:1]:
            parts = grp.get(ppath)
            if parts is not None:
                return sorted(parts.keys())
    return []


# --- reduced diagnostics ----------------------------------------------------

_COLUMN_RE = re.compile(r"\[(\d+)\]([^([\]]+)\(([^)]*)\)")


def parse_reduced_diag(path: str | Path) -> xr.Dataset:
    """Parse one reduced-diagnostic text table into an SI ``xr.Dataset``.

    WarpX reduced diags are whitespace-separated with a one-line header of
    ``[i]name(unit)`` tokens — ``#``-prefixed for every type except
    FieldProbe. Column 0 is the step and column 1 the time in seconds; they
    become the ``step`` coordinate and ``time`` variable, with every further
    column a data variable named from the header (its unit in
    ``attrs["units"]``). Values are left in SI — converters downstream pick
    the columns they understand.
    """
    path = Path(path)
    header: list[tuple[int, str, str]] = []
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            cols = _COLUMN_RE.findall(s)
            if cols and not header:
                header = [(int(i), name.strip(), unit) for i, name, unit in cols]
                continue
            try:
                rows.append([float(tok) for tok in s.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype="float64")
    names = {i: (name, unit) for i, name, unit in header if i < width}

    step = arr[:, 0].astype("int64")
    ds = xr.Dataset(coords={"step": step})
    for i in range(1, width):
        name, unit = names.get(i, (f"col{i}", ""))
        var = "time" if i == 1 else name
        ds[var] = ("step", arr[:, i])
        ds[var].attrs["units"] = unit
    ds.attrs["source"] = str(path)
    ds.attrs["diag_name"] = path.stem
    return ds


def list_reduced_diags(run_dir: str | Path) -> dict[str, Path]:
    """Map reduced-diagnostic names to their ``.txt`` tables for a run."""
    reduced = Path(run_dir) / "diags" / "reducedfiles"
    if not reduced.is_dir():
        return {}
    return {p.stem: p for p in sorted(reduced.glob("*.txt"))}


def hist_energy_from_reduced(run_dir: str | Path, units: CodeUnits | None = None) -> xr.Dataset | None:
    """Build the OSIRIS ``HIST/energy`` dataset from WarpX reduced diags.

    Combines the ``FieldEnergy`` total with the per-species ``ParticleEnergy``
    columns into the exact schema :func:`adept.osiris.io.load_hist_energy`
    returns (``field_energy``, ``kinetic_<species>``, ``kinetic_total``,
    ``total`` + ``attrs["total_drift_frac"]``, on a shared code-units ``t``
    axis), so the OSIRIS energy-conservation plots and the drift metric work
    on WarpX runs unchanged. Values are converted to the code energy unit
    (``m_e c^2 n_0 x_0`` per transverse area in 1D) when ``units`` is given,
    else left in the native J/m^2. Returns ``None`` when neither diagnostic
    was enabled.
    """
    tables = {name: parse_reduced_diag(p) for name, p in list_reduced_diags(run_dir).items()}
    fld = next((ds for ds in tables.values() if any(v.startswith("total_lev") for v in ds.data_vars)), None)
    par = next((ds for ds in tables.values() if "total" in ds.data_vars and "time" in ds.data_vars), None)
    if fld is None and par is None:
        return None

    e_scale = units.u_area0 if units is not None else 1.0
    t_scale = units.wp0 if units is not None else 1.0

    ref = fld if fld is not None else par
    ref_t = ref["time"].values * t_scale

    def onto(ds: xr.Dataset, col: str) -> np.ndarray:
        src_t = ds["time"].values * t_scale
        v = ds[col].values / e_scale
        if src_t.shape == ref_t.shape and np.allclose(src_t, ref_t):
            return v
        return np.interp(ref_t, src_t, v)

    data_vars: dict[str, tuple] = {}
    if fld is not None:
        data_vars["field_energy"] = ("t", onto(fld, "total_lev0"))

    kin_total = None
    if par is not None:
        species_cols = [v for v in par.data_vars if v not in ("time", "total") and not v.endswith("_mean")]
        for sp in species_cols:
            data_vars[f"kinetic_{sp}"] = ("t", onto(par, sp))
        kin_total = onto(par, "total")
        data_vars["kinetic_total"] = ("t", kin_total)

    attrs: dict[str, float] = {}
    if fld is not None and kin_total is not None:
        total = data_vars["field_energy"][1] + kin_total
        data_vars["total"] = ("t", total)
        denom = float(np.max(np.abs(total))) or 1.0
        attrs["total_drift_frac"] = float((total.max() - total.min()) / denom)

    ds = xr.Dataset(data_vars, coords={"t": ref_t}, attrs=attrs)
    ds["t"].attrs.update(long_name="time", units=r"1/\omega_p" if units is not None else "s")
    return ds


# --- batch conversion (the binary/ contract) --------------------------------


def _full_diag_dirs(run_dir: Path) -> list[Path]:
    """Full-diagnostic openPMD directories under ``diags/`` (skips reducedfiles)."""
    diags = run_dir / "diags"
    if not diags.is_dir():
        return []
    out = []
    for d in sorted(diags.iterdir()):
        if d.is_dir() and d.name != "reducedfiles" and _openpmd_files(d):
            out.append(d)
    return out


def save_run_datasets(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    units: CodeUnits | None = None,
    diagnostics: list[str] | set[str] | None = None,
) -> list[Path]:
    """Convert a WarpX run's outputs into the OSIRIS ``binary/`` NetCDF tree.

    Walks every full openPMD diagnostic under ``run_dir/diags/`` and every
    reduced-diagnostic text table, writing one NetCDF per diagnostic under
    ``out_dir``:

    - ``FLD/e1.nc`` … (fields, 1D OSIRIS naming), ``DENSITY/<sp>/charge.nc``;
    - ``RAW/<species>.nc`` (long-form particle dumps);
    - ``REDUCED/<name>.nc`` (native SI reduced tables);
    - ``HIST/energy.nc`` (the OSIRIS energy-history schema, from
      FieldEnergy + ParticleEnergy when present).

    When several full diagnostics dump the same record, later ones get a
    ``-<diagname>`` suffix on the key. ``diagnostics``, when given,
    whitelists keys (matched on the relative path or its leaf name). Each
    diagnostic is best-effort: a failure logs and skips rather than
    aborting the rest. Returns the written paths.
    """
    from adept.osiris.io import series_to_dataset

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []
    taken: set[str] = set()

    def want(rel: str) -> bool:
        return diagnostics is None or rel in diagnostics or Path(rel).name in diagnostics

    def write(ds: xr.Dataset, rel: str) -> None:
        dest = out_dir / f"{rel}.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        enc = {name: {"zlib": True, "complevel": 4, "shuffle": True} for name in ds.data_vars}
        ds.to_netcdf(dest, engine="h5netcdf", encoding=enc)
        written.append(dest)

    for diag_dir in _full_diag_dirs(run_dir):
        for mesh, comp in list_field_records(diag_dir):
            try:
                da = load_field_series(diag_dir, mesh, comp, units=units)
            except Exception as e:  # one bad record must not abort the rest
                print(f"[post] skipping {mesh}/{comp} from {diag_dir.name}: {e}")
                continue
            key_info = _mesh_diag_key(mesh, comp, one_d=int(da.attrs["sim.NDIMS"]) == 1)
            rel = key_info[0] if key_info is not None else f"FLD/{da.name}"
            if rel in taken:
                rel = f"{rel}-{diag_dir.name}"
            if not want(rel):
                continue
            try:
                write(series_to_dataset(da), rel)
                taken.add(rel)
            except Exception as e:
                print(f"[post] skipping {rel} from {diag_dir.name}: {e}")
        for species in list_species(diag_dir):
            rel = f"RAW/{species}"
            if rel in taken:
                rel = f"{rel}-{diag_dir.name}"
            if not want(rel):
                continue
            try:
                ds = load_particle_species(diag_dir, species, units=units)
                write(ds, rel)
                taken.add(rel)
            except Exception as e:
                print(f"[post] skipping {rel} from {diag_dir.name}: {e}")

    for name, path in list_reduced_diags(run_dir).items():
        rel = f"REDUCED/{name}"
        if not want(rel):
            continue
        try:
            write(parse_reduced_diag(path), rel)
        except Exception as e:
            print(f"[post] skipping {rel}: {e}")

    try:
        energy = hist_energy_from_reduced(run_dir, units=units)
        if energy is not None and want("HIST/energy"):
            write(energy, "HIST/energy")
    except Exception as e:
        print(f"[post] skipping HIST energy: {e}")

    return written
