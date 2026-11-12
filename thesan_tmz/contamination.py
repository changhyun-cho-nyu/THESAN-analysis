"""
Low-resolution contamination around the zoom target (report.txt block
CONTAMINATION), after Rahul's ``contamination.py``.

Low-res particles are types 2 and 3 (``LOWRES_TYPES``), the coarse dark
matter of the zoom.  Everything is measured from ``GroupPos`` of the target
galaxy's FoF group and expressed in physical kpc, Msun and units of that
group's R200c:

1. **members** -- low-res particles that are FoF members of the group and of
   its main subhalo (``GroupFirstSub``): count, mass, mass fraction and their
   r/R200c (min / 5 % / median / max).  Membership comes from the catalogue
   offsets (``GroupOffsetType``, ``SubhaloOffsetType``) of the group-ordered
   snapshot; without them this part is skipped.
2. **all low-res particles**, membership ignored: the nearest one (the radius of
   the clean sphere, and whether it is a member), and count / mass within
   0.5 ... 10 R200c.
3. **cumulative low-res mass fraction** M_low(<r) / M_all(<r) (all particle
   types) at those radii, and the radius where it first exceeds 1e-4, 1e-3,
   1e-2, 0.1 (bins of 0.025 R200c out to 10 R200c).

The snapshot is read one file and one type at a time, so memory stays at one
file's particles of one type.
"""

from __future__ import annotations

import h5py
import numpy as np

from . import arepo_io as io

LOWRES_TYPES = (2, 3)
CHECK_RADII = (0.5, 1, 2, 3, 4, 5, 6, 8, 10)
THRESHOLDS = (1e-4, 1e-3, 1e-2, 0.1)
RMAX_R200 = 10.0
NBINS = 400


def iter_type(files, ptype, fields=("Coordinates", "Masses")):
    """Yield (global offset, {field: array}) per snapshot file for one type;
    masses from ``MassTable`` when the type has no ``Masses`` dataset."""
    offset = 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            n = int(np.asarray(f["Header"].attrs["NumPart_ThisFile"])[ptype])
            if n == 0:
                continue
            g = f["PartType%d" % ptype]
            mt = float(np.asarray(f["Header"].attrs["MassTable"])[ptype])
            d = {}
            for k in fields:
                d[k] = g[k][:] if (k != "Masses" or "Masses" in g) else np.full(n, mt)
            yield offset, d
            offset += n


def radius(pos, centre, box):
    d = np.asarray(pos, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    if box:
        d -= box * np.round(d / box)
    return np.sqrt(np.sum(d * d, axis=1))


def summarise(r, m):
    """(N, M, r percentiles min/5/50/max) of a set of radii and masses."""
    if r.size == 0:
        return 0, 0.0, None
    return int(r.size), float(m.sum()), np.percentile(r, [0, 5, 50, 100])


def profile(chunks, lowres_types, centre, r200, box, rmax=RMAX_R200, nbins=NBINS):
    """Low-res and total mass in bins of r/R200c, the nearest low-res particle,
    and the low-res particles within ``rmax``.

    ``chunks`` yields (type, global offset, positions, masses).  Returns a dict:
    edges, m_low, m_all (per bin), rmin (code length), rmin_type, rmin_index,
    low_r, low_m (low-res within rmax)."""
    edges = np.linspace(0.0, rmax, nbins + 1)
    m_low, m_all = np.zeros(nbins), np.zeros(nbins)
    rmin, rmin_type, rmin_index = np.inf, None, None
    low_r, low_m = [], []
    for t, off, pos, mass in chunks:
        r = radius(pos, centre, box)
        x = r / r200
        sel = x < rmax
        h, _ = np.histogram(x[sel], bins=edges, weights=np.asarray(mass, dtype=np.float64)[sel])
        m_all += h
        if t in lowres_types:
            m_low += h
            i = int(np.argmin(r))
            if r[i] < rmin:
                rmin, rmin_type, rmin_index = float(r[i]), t, off + i
            low_r.append(r[sel])
            low_m.append(np.asarray(mass, dtype=np.float64)[sel])
    cat = lambda x: np.concatenate(x) if x else np.zeros(0)
    return dict(edges=edges, m_low=m_low, m_all=m_all, rmin=rmin, rmin_type=rmin_type, rmin_index=rmin_index,
                low_r=cat(low_r), low_m=cat(low_m))


def first_exceeding(edges, m_low, m_all, thresholds=THRESHOLDS):
    """Radius (in the units of ``edges``) where the cumulative fraction first
    exceeds each threshold; None when it never does."""
    cl, ca = np.cumsum(m_low), np.cumsum(m_all)
    frac = np.divide(cl, ca, out=np.zeros_like(cl), where=ca > 0)
    out = {}
    for thr in thresholds:
        i = np.nonzero(frac > thr)[0]
        out[thr] = float(edges[i[0] + 1]) if i.size else None
    return out


def analyse(snap, lowres_types=LOWRES_TYPES):
    """The CONTAMINATION block for the target galaxy's FoF group in ``snap``."""
    from . import catalog

    lines = ["CONTAMINATION (low-res particle types %s; snapshot %d, z = %.3f)"
             % (", ".join(str(t) for t in lowres_types), snap.num, snap.z)]
    ntot = np.asarray(snap.header["NumPart_Total"], dtype=np.int64)
    present = [t for t in lowres_types if t < len(ntot) and ntot[t] > 0]
    cat = catalog.load(snap)
    if cat is None:
        return "\n".join(lines + ["  no group catalogue -- skipped"])
    j = cat.target_index()
    gnr = cat.s("groupnr")
    if j is None or gnr is None:
        return "\n".join(lines + ["  no target galaxy / SubhaloGroupNr in the catalogue -- skipped"])
    g = int(np.asarray(gnr)[j])
    u, L = snap.units, snap.units.length_to_kpc(snap.a)
    msun = u.mass_to_msun
    centre = np.asarray(cat.g("pos"))[g].astype(np.float64)
    r200 = float(np.asarray(cat.g("r200c"))[g])
    m200 = float(np.asarray(cat.g("m200c"))[g])
    mgrp = float(np.asarray(cat.g("mass"))[g])
    fsub = int(np.asarray(cat.g("firstsub"))[g]) if cat.g("firstsub") is not None else -1
    lines.append("  target galaxy = subhalo %d in FoF group %d: R200c = %.3g pkpc, M200c = %.3g Msun, "
                 "GroupMass = %.3g Msun" % (j, g, r200 * L, m200 * msun, mgrp * msun))
    if not r200 > 0:
        return "\n".join(lines + ["  R200c = 0 -- cannot normalise, skipped"])
    if not present:
        return "\n".join(lines + ["  the snapshot has no particles of types %s: no low-res contamination possible"
                                  % (", ".join(str(t) for t in lowres_types))])

    # 1. members
    goff, glen = cat.g("offsettype"), cat.g("lentype")
    soff, slen = cat.s("offsettype"), cat.s("lentype")
    mem_g, mem_s = ([], []), ([], [])
    if goff is None or glen is None:
        lines.append("  members: catalogue has no GroupOffsetType -- membership part skipped")
    else:
        for t in present:
            a = int(np.asarray(goff)[g, t])
            b = a + int(np.asarray(glen)[g, t])
            s0 = s1 = None
            if fsub >= 0 and soff is not None and slen is not None:
                s0 = int(np.asarray(soff)[fsub, t])
                s1 = s0 + int(np.asarray(slen)[fsub, t])
            for off, d in iter_type(snap.files, t):
                n = len(d["Masses"])
                for (lo, hi), out in (((a, b), mem_g), ((s0, s1), mem_s)):
                    if lo is None:
                        continue
                    x0, x1 = max(lo, off), min(hi, off + n)
                    if x1 > x0:
                        out[0].append(radius(d["Coordinates"][x0 - off:x1 - off], centre, snap.boxsize))
                        out[1].append(np.asarray(d["Masses"][x0 - off:x1 - off], dtype=np.float64))
        cc = lambda x: np.concatenate(x) if x else np.zeros(0)
        msub = float(np.asarray(cat.s("mass"))[fsub]) if fsub >= 0 else np.nan
        for lab, (rr, mm), mtot in (("FoF group %d" % g, mem_g, mgrp), ("main subhalo %d" % fsub, mem_s, msub)):
            if lab.startswith("main") and fsub < 0:
                continue
            n, m, q = summarise(cc(rr) / r200, cc(mm))
            if n == 0:
                lines.append("  members of %-18s: no low-res particles -> clean" % lab)
            else:
                lines.append("  members of %-18s: N = %d, M = %.3g Msun, M/M_total = %.3g; r/R200c min %.2f, 5%% %.2f, "
                             "median %.2f, max %.2f" % (lab, n, m * msun, m / mtot, q[0], q[1], q[2], q[3]))

    # 2. + 3. all low-res particles and the cumulative fraction (all types)
    def chunks():
        for t in range(len(ntot)):
            if ntot[t] == 0:
                continue
            for off, d in iter_type(snap.files, t):
                yield t, off, d["Coordinates"], d["Masses"]

    pr = profile(chunks(), present, centre, r200, snap.boxsize)
    if np.isfinite(pr["rmin"]):
        member = ""
        if goff is not None and glen is not None and pr["rmin_type"] is not None:
            a = int(np.asarray(goff)[g, pr["rmin_type"]])
            member = ", %s of the group" % ("a member" if a <= pr["rmin_index"] < a + int(np.asarray(glen)[g, pr["rmin_type"]])
                                           else "NOT a member")
        lines.append("  nearest low-res particle: r = %.3g pkpc = %.3f R200c (type %d%s) => clean sphere radius %.3f R200c"
                     % (pr["rmin"] * L, pr["rmin"] / r200, pr["rmin_type"], member, pr["rmin"] / r200))
    cl, ca = np.cumsum(pr["m_low"]), np.cumsum(pr["m_all"])
    lines.append("  %8s %10s %12s %12s" % ("r/R200c", "N_low(<r)", "M_low(<r)", "f_low(<r)"))
    x = pr["low_r"] / r200
    for k in CHECK_RADII:
        s = x < k
        i = int(np.searchsorted(pr["edges"], k, side="right")) - 1
        f = cl[i - 1] / ca[i - 1] if i >= 1 and ca[i - 1] > 0 else 0.0
        lines.append("  %8g %10d %12.3g %12.3g" % (k, int(s.sum()), pr["low_m"][s].sum() * msun, f))
    for thr, rr in first_exceeding(pr["edges"], pr["m_low"], pr["m_all"]).items():
        lines.append("  f_low(<r) first exceeds %.0e at r = %s" % (thr, ("%.3f R200c" % rr) if rr is not None
                                                                   else "> %g R200c" % RMAX_R200))
    return "\n".join(lines)
