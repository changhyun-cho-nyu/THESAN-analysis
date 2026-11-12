"""
Intrinsic 1500 A luminosity of each star particle, the way the run computes it.

AREPO with ``MRT_FOF_1500`` stores ``Subhalo_Luminosity1500`` /
``Group_Luminosity1500``: the sum over member stars of

    L_lambda(1500) = M_init * <L_lambda>_{1490-1510 A}(age, Z)        [erg/s/A]

(``src/MRT/RT_stellar_diagnostics.cc``).  This module rebuilds it per particle
from the same spectrum tables, so a map of it adds up to the catalogue value:

* tables: the run's own ``SpectrumTablePath`` (Pop II, ``Lnu[N_metallicity,
  N_age, Number_of_freq]`` in erg/s/Hz per Msun, ``Energy`` in eV, ``age`` in
  Gyr) and ``PopIIISpectrumTablePath`` (``Lnu[N_age, Number_of_freq]``), read
  from the snapshot's ``/Parameters``;
* band: integral of L_nu/h over E between hc/1510 A and hc/1490 A, divided by
  20 A (``integrate_uv_band``; L_nu is piecewise linear in E between table
  points, so the trapezoid is exact);
* interpolation: bilinear in (log age, log Z) on log L when all four corners are
  positive, clamped to the table edges (``interpolate_uv``); Pop III in log
  age only;
* star: age = t(a_now) - t(GFM_StellarFormationTime), mass = GFM_InitialMass,
  Z = GFM_Metallicity, Pop III from ``IsPOPIII``; wind particles (formation
  time <= 0) emit nothing.

No dust, no nebular emission.  ``l_nu`` converts to L_nu = L_lambda lambda^2 / c
at 1500 A, the quantity behind M_UV.
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from . import arepo_io as io

PLANCK = 6.62607015e-27            # erg s
CLIGHT = 2.99792458e10             # cm/s
EV = 1.602176634e-12               # erg
EV_ANGSTROM = PLANCK * CLIGHT / (1.0e-8 * EV)       # E[eV] x lambda[A]
BAND_A = (1490.0, 1510.0)
LAMBDA_CM = 1500.0e-8


def band_lambda(energy_ev, lnu):
    """<L_lambda> over 1490-1510 A [erg/s/A] of spectra ``lnu[..., nfreq]``
    (erg/s/Hz), as ``integrate_uv_band`` does it."""
    E = np.asarray(energy_ev, dtype=np.float64)
    e_min, e_max = EV_ANGSTROM / BAND_A[1], EV_ANGSTROM / BAND_A[0]
    if E[0] > e_min or E[-1] < e_max:
        raise ValueError("spectrum table does not cover 1490-1510 A")
    y = np.asarray(lnu, dtype=np.float64) / PLANCK
    out = np.zeros(y.shape[:-1])
    for i in range(1, len(E)):
        lo, hi = max(E[i - 1], e_min), min(E[i], e_max)
        if hi <= lo:
            continue
        slope = (y[..., i] - y[..., i - 1]) / (E[i] - E[i - 1])
        y_lo = y[..., i - 1] + slope * (lo - E[i - 1])
        y_hi = y[..., i - 1] + slope * (hi - E[i - 1])
        out += 0.5 * (y_lo + y_hi) * (hi - lo)
    return out * EV / (BAND_A[1] - BAND_A[0])


class SpectrumTable:
    """One spectrum table reduced to <L_lambda>(1500) on its (Z, age) grid."""

    def __init__(self, path, pop3=False):
        self.path, self.pop3 = path, pop3
        with h5py.File(path, "r") as f:
            na = int(np.asarray(f["N_age"][()]).ravel()[0])
            self.log_age = np.log10(np.maximum(f["age"][:].astype(np.float64), 1e-6 if pop3 else 0.0))
            lnu = f["Lnu"][:].astype(np.float64)
            E = f["Energy"][:]
            if pop3:
                self.log_z = None
                self.tab = band_lambda(E, lnu.reshape(na, -1))[None, :]        # [1, N_age]
            else:
                nz = int(np.asarray(f["N_metallicity"][()]).ravel()[0])
                self.log_z = np.log10(f["metallicity"][:].astype(np.float64))
                self.tab = band_lambda(E, lnu.reshape(nz, na, -1))           # [N_Z, N_age]

    @staticmethod
    def _weights(value, log_axis):
        """Lower index and fraction, clamped at both ends (``uv_axis_weights``)."""
        v = np.asarray(value, dtype=np.float64)
        lv = np.log10(np.where(v > 0, v, 1e-300))
        n = len(log_axis)
        i = np.clip(np.searchsorted(log_axis, lv, side="left") - 1, 0, n - 2)
        f = (lv - log_axis[i]) / (log_axis[i + 1] - log_axis[i])
        f = np.where(lv <= log_axis[0], 0.0, np.where(lv >= log_axis[-1], 1.0, f))
        return i, np.clip(f, 0.0, 1.0)

    def __call__(self, age_gyr, metallicity=None):
        """<L_lambda>(1500) per initial Msun [erg/s/A] for each star."""
        ia, da = self._weights(age_gyr, self.log_age)
        if self.log_z is None:
            iz, dz = np.zeros_like(ia), np.zeros_like(da)
        else:
            iz, dz = self._weights(metallicity, self.log_z)
        izp = np.minimum(iz + 1, self.tab.shape[0] - 1)
        v = np.stack([self.tab[iz, ia], self.tab[iz, ia + 1], self.tab[izp, ia], self.tab[izp, ia + 1]])
        w = np.stack([(1 - dz) * (1 - da), (1 - dz) * da, dz * (1 - da), dz * da])
        pos = np.all((v > 0) | (w <= 0), axis=0)
        with np.errstate(divide="ignore"):
            logv = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), 0.0)
        return np.where(pos, 10.0 ** np.sum(w * logv, axis=0), np.maximum(np.sum(w * v, axis=0), 0.0))


def table_paths(snap, spectra_dir=None):
    """(Pop II path, Pop III path or None, note).  Paths come from the run's
    parameters; with ``spectra_dir`` the same file names are looked up there
    (for analysing a copy of the run elsewhere)."""
    p = snap.parameters or {}
    out = []
    for key in ("SpectrumTablePath", "PopIIISpectrumTablePath"):
        path = p.get(key)
        if isinstance(path, str) and path.strip():
            path = path.strip()
            if spectra_dir:
                path = os.path.join(spectra_dir, os.path.basename(path))
            out.append(path)
        else:
            out.append(None)
    if out[0] is None:
        return None, None, "no SpectrumTablePath in the run's parameters (built without MRT stars)"
    if not os.path.isfile(out[0]):
        return None, None, "spectrum table not found: %s (use --spectra-dir)" % out[0]
    if out[1] is not None and not os.path.isfile(out[1]):
        return out[0], None, "Pop III table not found: %s -- Pop III stars get no UV" % out[1]
    return out[0], out[1], "tables: %s%s" % (out[0], (", " + out[1]) if out[1] else "")


_TABLES = {}


def _table(path, pop3):
    key = (path, pop3)
    if key not in _TABLES:
        _TABLES[key] = SpectrumTable(path, pop3)
    return _TABLES[key]


def star_l_lambda(snap, stars, spectra_dir=None):
    """Per-star intrinsic <L_lambda>(1500) [erg/s/A] for the PartType4 dict
    ``stars`` (needs GFM_StellarFormationTime, GFM_InitialMass, GFM_Metallicity;
    IsPOPIII if present).  Returns (array or None, note)."""
    p2, p3, note = table_paths(snap, spectra_dir)
    if p2 is None:
        return None, note
    need = ("GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metallicity")
    if any(k not in stars for k in need):
        return None, "stars lack %s" % ", ".join(k for k in need if k not in stars)
    u = snap.units
    tform = np.asarray(stars["GFM_StellarFormationTime"], dtype=np.float64)
    star = tform > 0
    age = np.zeros(len(tform))
    age[star] = snap.age_gyr - u.age_gyr(tform[star])
    mass = np.asarray(stars["GFM_InitialMass"], dtype=np.float64) * u.mass_to_msun
    zmet = np.asarray(stars["GFM_Metallicity"], dtype=np.float64)
    is3 = np.asarray(stars["IsPOPIII"], dtype=bool) if "IsPOPIII" in stars else np.zeros(len(tform), dtype=bool)
    out = np.zeros(len(tform))
    m2 = star & ~is3
    if m2.any():
        out[m2] = mass[m2] * _table(p2, False)(age[m2], zmet[m2])
    m3 = star & is3
    if m3.any() and p3:
        out[m3] = mass[m3] * _table(p3, True)(age[m3])
    return out, note


def catalogue_check(snap, spectra_dir=None):
    """Compare the sum of :func:`star_l_lambda` over the target galaxy's bound
    stars with the catalogue's ``Subhalo_Luminosity1500`` (same recipe in
    AREPO, so the ratio should be 1).  Uses the snapshot's group ordering: the
    target must be subhalo 0 (central of group 0), whose stars are then the
    first ``SubhaloLenType[0][4]`` star particles.  Returns a one-line string."""
    from . import catalog

    cat = catalog.load(snap)
    if cat is None:
        return "UV check: no catalogue"
    j = cat.target_index()
    cl = cat.s("Subhalo_Luminosity1500")
    lt = cat.s("lentype")
    if cl is None or lt is None:
        return "UV check: catalogue has no Subhalo_Luminosity1500 (built without MRT_FOF_1500)"
    if j != 0:
        return "UV check: target is subhalo %s, not 0 -- needs group offsets, skipped" % j
    n = int(np.asarray(lt)[0, io.PT_STAR])
    st = snap.part(io.PT_STAR, stars_with_uv_fields())
    sub = {k: np.asarray(v)[:n] for k, v in st.items() if np.ndim(v) > 0 and len(v) == st.get("count", -1)}
    lam, note = star_l_lambda(snap, sub, spectra_dir)
    if lam is None:
        return "UV check: " + note
    u = snap.units
    cat_cgs = float(np.asarray(cl)[0]) * u.UnitEnergy_in_cgs / u.UnitTime_in_s
    ours = float(lam.sum())
    return ("UV check: target galaxy (%d stars) L_lambda(1500) = %.4g erg/s/A here vs %.4g in the catalogue, ratio %.4f "
            "(1 expected; assumes a group-ordered snapshot), M_UV = %.2f"
            % (n, ours, cat_cgs, ours / cat_cgs if cat_cgs > 0 else np.nan, float(muv(l_nu(ours)))))


def l_nu(l_lambda):
    """L_lambda [erg/s/A] at 1500 A -> L_nu [erg/s/Hz]."""
    return np.asarray(l_lambda) * 1.0e8 * LAMBDA_CM ** 2 / CLIGHT


def muv(l_nu_cgs):
    """Absolute AB magnitude of L_nu [erg/s/Hz]."""
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(np.asarray(l_nu_cgs) / (4.0 * np.pi * (10.0 * 3.0856775814913673e18) ** 2)) - 48.6


def stars_with_uv_fields():
    return ["Coordinates", "Masses", "GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metallicity", "IsPOPIII"]


__all__ = ["band_lambda", "SpectrumTable", "table_paths", "star_l_lambda", "l_nu", "muv", "io"]
