"""
Which black holes sit in low-resolution gas -- and should be hidden.

In a zoom, a small group of high-res dark matter at the edge of the high-res
region can pass the seeding threshold, and AREPO then seeds the BH in the
group's densest gas cell even when that cell is a heavy low-res one
(``fof.cc``; see the vault note "12. BH Seeding Guard -- Seed Only in High-Res
Gas").  Such a BH is born with a huge dynamical mass, its kernel covers coarse
gas, and it can grow to 1e9-1e10 Msun far from the target galaxy.  It is a
numerical artefact, and it dominates any figure that lists the most massive
BHs.

The test here is the one the code itself uses for star formation
(``starformation.cc:312``) and that the seeding guard uses: a gas cell is
high-res when ``HighResGasMass >= HIGHRESMASSFAC * Masses`` with
``HIGHRESMASSFAC = 0.5`` (``constants.h:51``).  For a BH we take the gas it
actually accretes from -- every cell within its ``BH_Hsml`` -- and compute

    f_hr = sum(HighResGasMass) / sum(Masses)        (mass-weighted)

The BH is **low-res** when ``f_hr < 0.5``.  If no cell lies within
``BH_Hsml`` (or the snapshot has no ``BH_Hsml``), the nearest gas cell is used.

A snapshot without ``HighResGasMass`` (not a zoom, or built without
``REFINEMENT_HIGH_RES_GAS``) cannot be tested: every BH is kept.

The gas is read one snapshot file at a time, so the memory cost is one
file's gas, not the whole snapshot's.  With scipy a k-d tree finds the
neighbours; without it a brute-force distance loop does (fine for tens of BHs).
"""

from __future__ import annotations

import h5py
import numpy as np

from . import arepo_io as io

#: constants.h:51 -- a cell is high-res when HighResGasMass >= this x its mass
HIGHRESMASSFAC = 0.5

_CACHE_KEY = "_lowres_classification"


def highres_fraction(bh_pos, bh_hsml, chunks, boxsize=None):
    """Mass-weighted high-res fraction of the gas within each BH's ``BH_Hsml``.

    ``chunks`` yields ``(Coordinates, Masses, HighResGasMass)`` arrays (one
    snapshot file each).  Returns ``(f_hr, n_cells)``; for a BH with no cell
    inside ``BH_Hsml`` the nearest cell's own fraction is used and
    ``n_cells`` is 0.
    """
    bh_pos = np.atleast_2d(np.asarray(bh_pos, dtype=np.float64))
    nbh = bh_pos.shape[0]
    hsml = np.asarray(bh_hsml, dtype=np.float64) if bh_hsml is not None else np.zeros(nbh)
    m_sum, hr_sum = np.zeros(nbh), np.zeros(nbh)
    n_in = np.zeros(nbh, dtype=np.int64)
    near_d, near_f = np.full(nbh, np.inf), np.full(nbh, np.nan)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    for pos, mass, hr in chunks:
        if pos is None or len(pos) == 0:
            continue
        pos = np.asarray(pos, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        hr = np.asarray(hr, dtype=np.float64)
        if cKDTree is not None:
            if boxsize:
                tree = cKDTree(np.mod(pos, boxsize), boxsize=boxsize)
                q = np.mod(bh_pos, boxsize)
            else:
                tree, q = cKDTree(pos), bh_pos
            dist, j = tree.query(q, k=1)
            for i in range(nbh):
                if hsml[i] > 0:
                    idx = tree.query_ball_point(q[i], hsml[i])
                    if idx:
                        m_sum[i] += mass[idx].sum()
                        hr_sum[i] += hr[idx].sum()
                        n_in[i] += len(idx)
        else:
            dist, j = np.empty(nbh), np.empty(nbh, dtype=np.int64)
            for i in range(nbh):
                r = io.radius_from(pos, bh_pos[i], boxsize)
                j[i] = int(np.argmin(r))
                dist[i] = r[j[i]]
                inside = r <= hsml[i]
                if inside.any():
                    m_sum[i] += mass[inside].sum()
                    hr_sum[i] += hr[inside].sum()
                    n_in[i] += int(inside.sum())
        closer = dist < near_d
        near_d[closer] = dist[closer]
        near_f[closer] = hr[j[closer]] / np.maximum(mass[j[closer]], 1e-300)

    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n_in > 0, hr_sum / np.maximum(m_sum, 1e-300), near_f)
    return f, n_in


def _gas_chunks(snap):
    """(Coordinates, Masses, HighResGasMass) from each snapshot file in turn."""
    for fn in snap.files:
        with h5py.File(fn, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            if "Coordinates" not in g or "HighResGasMass" not in g:
                continue
            yield g["Coordinates"][:], g["Masses"][:], g["HighResGasMass"][:]


def _has_highres_field(snap):
    for fn in snap.files:
        with h5py.File(fn, "r") as f:
            if "PartType0" in f and "Coordinates" in f["PartType0"]:
                return "HighResGasMass" in f["PartType0"]
    return False


def classify(snap):
    """Classify every BH in ``snap``.  Returns a dict:

    ``id``, ``highres_frac``, ``n_cells`` (arrays, PartType5 order),
    ``lowres`` (bool array), ``testable`` (bool) and ``note`` (str).
    Cached on the snapshot object, so asking again costs nothing.
    """
    cache = getattr(snap, "_cache", None)
    if cache is not None and _CACHE_KEY in cache:
        return cache[_CACHE_KEY]

    bh = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs", "BH_Hsml"])
    n = int(bh.get("count", 0))
    out = {"id": np.zeros(0, dtype=np.int64), "highres_frac": np.zeros(0), "n_cells": np.zeros(0, dtype=np.int64),
           "lowres": np.zeros(0, dtype=bool), "testable": False, "note": "no black holes"}
    if n and "Coordinates" in bh:
        ids = np.asarray(bh["ParticleIDs"], dtype=np.int64)
        out.update(id=ids, highres_frac=np.full(n, np.nan), n_cells=np.zeros(n, dtype=np.int64),
                   lowres=np.zeros(n, dtype=bool))
        if not _has_highres_field(snap):
            out["note"] = "no HighResGasMass in the snapshot: low-res BHs cannot be identified, all kept"
        else:
            f, nc = highres_fraction(bh["Coordinates"], bh.get("BH_Hsml"), _gas_chunks(snap), snap.boxsize)
            low = np.isfinite(f) & (f < HIGHRESMASSFAC)
            out.update(highres_frac=f, n_cells=nc, lowres=low, testable=True,
                       note="%d of %d BHs in low-res gas (high-res mass fraction within BH_Hsml < %.1f)"
                            % (int(low.sum()), n, HIGHRESMASSFAC))
    if cache is not None:
        cache[_CACHE_KEY] = out
    return out


def lowres_ids(snap):
    """IDs of the low-res BHs in ``snap`` (empty when none or untestable)."""
    c = classify(snap)
    return set(int(i) for i in c["id"][c["lowres"]])


def report_lines(snap, units=None, hidden=True):
    """Text for report.txt: what was found and whether it was hidden."""
    c = classify(snap)
    lines = ["LOW-RES BLACK HOLES (in %s)" % snap.files[0].split("/")[-1], "  " + c["note"]]
    if c["testable"] and c["lowres"].any():
        mass = snap.part(io.PT_BH, ["ParticleIDs", "BH_Mass", "Masses"])
        m = np.asarray(mass.get("BH_Mass", mass.get("Masses")), dtype=np.float64)
        msun = (units or snap.units).mass_to_msun
        for k in np.where(c["lowres"])[0]:
            lines.append("  ID %d  M_BH = %.3g Msun  high-res fraction %.3f (%d cells within BH_Hsml)"
                         % (c["id"][k], m[k] * msun, c["highres_frac"][k], c["n_cells"][k]))
        lines.append("  %s" % ("hidden from figs 02(d), 08, 16, 17, 18, 19 and bh_table.txt; box totals "
                               "(figs 02(a-c), 04, 09) still include them. --show-lowres-bhs shows them"
                               if hidden else "shown everywhere (--show-lowres-bhs)"))
    return "\n".join(lines)
