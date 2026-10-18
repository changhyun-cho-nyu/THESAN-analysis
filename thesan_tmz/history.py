"""
Build (and cache) a time series of galaxy properties by walking the snapshots.

Reading every snapshot of a zoom run takes minutes to hours, and every figure
downstream wants the same handful of scalars, so the result is cached to a
single ``.npz``.  Delete the cache (or pass ``rebuild=True``) after changing
anything in :mod:`galaxy`.

The cache stores the aperture radii and the centring method it used, and
:func:`load_history` refuses to reuse a cache built with different settings --
silently mixing 5 kpc and 30 kpc apertures in one plot is exactly the sort of
error that survives review.
"""

from __future__ import annotations

import json
import os
import traceback

import numpy as np

from . import arepo_io as io
from . import galaxy
from .units import Units

DEFAULT_RADII = (1.0, 2.0, 5.0, 10.0, 30.0)


def build_history(
    output_dir,
    nums=None,
    radii_kpc=DEFAULT_RADII,
    centre_prefer=("bh", "subhalo", "group", "potmin", "densest"),
    bh_id=None,
    group_index=0,
    verbose=True,
):
    """Measure every snapshot and return a dict of arrays keyed by quantity.

    Snapshots that fail (missing file, no centre, corrupt chunk) are skipped
    with a message rather than aborting the whole walk -- a long zoom run
    almost always has one bad snapshot.
    """
    nums = io.available_snapshots(output_dir) if nums is None else list(nums)
    if not nums:
        raise FileNotFoundError("no snapshots under %s" % output_dir)

    rows, used = [], []
    units = None
    for n in nums:
        try:
            snap = io.Snapshot(output_dir, n, units=units)
            units = units or snap.units
            centre, cinfo = galaxy.auto_centre(snap, prefer=centre_prefer, bh_id=bh_id, group_index=group_index)
            if centre is None:
                if verbose:
                    print("  [history] snap %3d: no centre (%s) -- skipped" % (n, cinfo.get("reason")))
                continue
            # Reuse the catalogue the centring already read rather than opening
            # every chunk of it a second time -- for a zoom with a few hundred
            # catalogue files that is the difference between minutes and hours.
            rec = galaxy.measure(
                snap, centre, radii_kpc=radii_kpc, group_index=group_index,
                cat=cinfo.get("_catalog"), subhalo_index=cinfo.get("subhalo_index"),
            )
            rec["centre_method"] = cinfo.get("method", "?")
            for k in ("bh_id", "M200c_msun", "R200c_kpc", "subhalo_index"):
                if k in cinfo:
                    rec.setdefault(k, cinfo[k])
            rows.append(rec)
            used.append(n)
            if verbose:
                print(
                    "  [history] snap %3d  z=%6.3f  M*(sub)=%.3e  M*(r%g)=%.3e  M_BH=%.3e  SFR=%.3e  [%s]"
                    % (
                        n,
                        rec["z"],
                        rec.get("sub_mstar_msun", np.nan),
                        max(radii_kpc),
                        rec.get("mstar_r%g" % max(radii_kpc), 0.0),
                        rec.get("bh_mass_msun", 0.0),
                        rec.get("sub_sfr_msun_per_yr", rec.get("sfr_r%g" % max(radii_kpc), 0.0)),
                        rec.get("centre_method", "?"),
                    )
                )
        except Exception as exc:
            # Deliberately broad.  A single unreadable snapshot must not cost
            # the whole history: an exception type nobody anticipated used to
            # propagate out of the loop and take every other snapshot with it,
            # and the only symptom was two figures quietly missing.  The type is
            # printed so an unexpected one is still visible.
            if verbose:
                print("  [history] snap %3d failed: %s: %s" % (n, type(exc).__name__, exc))
            if os.environ.get("THESAN_TMZ_TRACEBACK"):
                traceback.print_exc()

    if not rows:
        raise RuntimeError(
            "no snapshot could be measured under %s -- every one of the %d snapshots failed "
            "above; rerun with THESAN_TMZ_TRACEBACK=1 to see why" % (output_dir, len(nums)))

    # Union of keys, so a snapshot missing e.g. a BH still lines up.
    #
    # The shape of a key must be decided from a value that is actually present,
    # not from the first row: a quantity that only exists once stars form (j,
    # sigma, half-mass radius) is absent from the early snapshots, and the
    # placeholder for those is a scalar NaN.  Deciding on row 0 would classify
    # such a key as scalar and then fail on the first real 3-vector.  Missing
    # entries are padded with NaN of the right shape instead.
    keys = sorted({k for r in rows for k in r})
    hist = {}
    for k in keys:
        present = [r[k] for r in rows if k in r and r[k] is not None]
        if not present:
            continue
        proto = present[0]

        if isinstance(proto, str):
            hist[k] = np.asarray([str(r.get(k, "")) for r in rows])
            continue

        try:
            shape = np.shape(proto)
        except (TypeError, ValueError):
            continue

        try:
            if shape == ():
                hist[k] = np.asarray([float(r[k]) if k in r and r[k] is not None else np.nan for r in rows], dtype=np.float64)
            else:
                blank = np.full(shape, np.nan, dtype=np.float64)
                stacked = [np.asarray(r[k], dtype=np.float64) if (k in r and np.shape(r[k]) == shape) else blank for r in rows]
                hist[k] = np.asarray(stacked, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            if verbose:
                print("  [history] key %r could not be stacked (%s) -- dropped" % (k, exc))
            continue

    hist["snapnum"] = np.asarray(used, dtype=np.float64)
    hist["_radii_kpc"] = np.asarray(radii_kpc, dtype=np.float64)
    hist["_units"] = units
    return hist


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _meta(radii_kpc, centre_prefer, bh_id, group_index):
    return json.dumps(
        {"radii": list(map(float, radii_kpc)), "centre": list(centre_prefer), "bh_id": bh_id, "group": group_index},
        sort_keys=True,
    )


def load_history(
    output_dir,
    cache_path=None,
    rebuild=False,
    radii_kpc=DEFAULT_RADII,
    centre_prefer=("bh", "subhalo", "group", "potmin", "densest"),
    bh_id=None,
    group_index=0,
    nums=None,
    verbose=True,
):
    """Return a cached history, rebuilding it when stale or missing."""
    cache_path = cache_path or os.path.join(output_dir, "analysis_history.npz")
    meta = _meta(radii_kpc, centre_prefer, bh_id, group_index)

    if os.path.exists(cache_path) and not rebuild:
        try:
            z = np.load(cache_path, allow_pickle=True)
            if str(z["_meta"]) == meta:
                hist = {k: z[k] for k in z.files if not k.startswith("_")}
                hist["_radii_kpc"] = z["_radii_kpc"]
                hist["_units"] = _units_from_cache(z)
                if verbose:
                    print("  [history] loaded cache %s (%d snapshots)" % (cache_path, hist["snapnum"].size))
                return hist
            if verbose:
                print("  [history] cache settings changed -- rebuilding")
        except (OSError, KeyError, ValueError) as exc:
            if verbose:
                print("  [history] unreadable cache (%s) -- rebuilding" % exc)

    hist = build_history(
        output_dir, nums=nums, radii_kpc=radii_kpc, centre_prefer=centre_prefer, bh_id=bh_id, group_index=group_index, verbose=verbose
    )
    save_history(hist, cache_path, meta)
    return hist


def save_history(hist, cache_path, meta):
    u = hist.get("_units")
    payload = {k: v for k, v in hist.items() if not k.startswith("_")}
    payload["_meta"] = np.asarray(meta)
    payload["_radii_kpc"] = hist["_radii_kpc"]
    if u is not None:
        payload["_unit_pack"] = np.asarray(
            [u.UnitLength_in_cm, u.UnitMass_in_g, u.UnitVelocity_in_cm_per_s, u.h, u.Omega0, u.OmegaLambda, u.OmegaBaryon, float(u.comoving)]
        )
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print("  [history] wrote cache %s" % cache_path)


def _units_from_cache(z):
    if "_unit_pack" not in z.files:
        return None
    p = np.asarray(z["_unit_pack"], dtype=np.float64)
    return Units(
        UnitLength_in_cm=p[0],
        UnitMass_in_g=p[1],
        UnitVelocity_in_cm_per_s=p[2],
        HubbleParam=p[3],
        Omega0=p[4],
        OmegaLambda=p[5],
        OmegaBaryon=p[6],
        ComovingIntegrationOn=bool(p[7]),
    )


# ---------------------------------------------------------------------------
# Small conveniences on top of a history dict
# ---------------------------------------------------------------------------


#: Catalogue equivalents of the aperture quantities, so that the same plotting
#: code works on a history built by :func:`build_catalog_history` (no particle
#: data) as on one built by :func:`build_history`.  ``vcirc`` maps onto
#: ``SubhaloVmax``, which is the circular velocity the BTFR is normally quoted
#: against anyway.
CATALOG_EQUIVALENT = {
    "mstar": "mstar_msun",
    "mgas": "mgas_msun",
    "mdm": "mdm_msun",
    "sfr": "sfr_msun_per_yr",
    "Zgas": "Zgas",
    "Zstar": "Zstar",
    "vcirc": "vmax_kms",
}


def key(hist, base, radius=None):
    """Look up ``base`` at a given aperture, defaulting to the largest one.

    >>> key(hist, 'mstar')          # mstar at the outermost radius
    >>> key(hist, 'mgas_cold', 5.0) # mgas_cold_r5

    A catalogue-only history has no apertures -- Subfind's own boundary is the
    aperture -- so the lookup falls back to the catalogue quantity of the same
    meaning.  Missing entirely gives NaN rather than an error, which is what
    lets a partial run still produce figures.
    """
    if radius is None:
        radii = np.asarray(hist.get("_radii_kpc", []), dtype=np.float64)
        radius = float(np.max(radii)) if radii.size and np.isfinite(radii).any() else np.nan
    if np.isfinite(radius):
        name = "%s_r%g" % (base, radius)
        if name in hist:
            return hist[name]
    alt = CATALOG_EQUIVALENT.get(base)
    if alt and alt in hist:
        return hist[alt]
    n = np.shape(hist.get("snapnum", np.zeros(0)))
    return np.full(n, np.nan)


def growth_rate(t_gyr, m):
    """d m / d t in units of [m]/yr, using centred differences.

    Returned on the same grid as the input, with the endpoints one-sided.
    Negative values are real (mergers can remove mass from a fixed aperture)
    and are not clipped -- clipping them hides bugs.
    """
    t = np.asarray(t_gyr, dtype=np.float64) * 1.0e9
    m = np.asarray(m, dtype=np.float64)
    return np.gradient(m, t)


def cumulative_from_rate(t_gyr, rate_per_yr):
    """Trapezoidal integral of a rate given in [X]/yr over an age axis in Gyr."""
    t = np.asarray(t_gyr, dtype=np.float64) * 1.0e9
    r = np.nan_to_num(np.asarray(rate_per_yr, dtype=np.float64))
    out = np.zeros_like(r)
    if r.size > 1:
        out[1:] = np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))
    return out


# ---------------------------------------------------------------------------
# Catalogue-only history: the fast path
# ---------------------------------------------------------------------------


def build_catalog_history(output_dir, nums=None, use_inrad=True, prefer="central", bh_id=None,
                          units=None, verbose=True):
    """Time series of the target galaxy from the group catalogues alone.

    No particle data is read, so this costs seconds where :func:`build_history`
    costs minutes-to-hours: a catalogue is one small file per snapshot, a
    snapshot is every particle in the zoom.  Everything Subfind already
    measured -- stellar mass, SFR, half-mass radius, velocity dispersion, BH
    mass, host M200c -- is here, and for the evolution figures that is all of
    it.  Use :func:`build_history` when a figure needs something Subfind does
    not compute: gas phases, apertures other than Subfind's, kinematics of a
    chosen component, anything spatially resolved.

    Returns the same dict-of-arrays shape as :func:`build_history`, with keys
    named as :meth:`Catalog.properties` names them.
    """
    from . import catalog as cat_mod

    nums = io.available_group_catalogs(output_dir) if nums is None else list(nums)
    if not nums:
        raise FileNotFoundError("no group catalogues under %s" % output_dir)

    rows, used = [], []
    for n in nums:
        try:
            cat = cat_mod.Catalog(output_dir, n, units=units)
            units = units or cat.units
            if cat.n_sub == 0:
                if verbose:
                    print("  [catalog history] snap %3d: empty catalogue -- skipped" % n)
                continue
            i = cat.target_index(prefer=prefer)
            rec = cat.properties(i, use_inrad=use_inrad)
            if not rec:
                continue
            rec["num"] = n
            rec["age_gyr"] = float(units.age_gyr(cat.a))
            rec["n_subhalos"] = cat.n_sub
            rec["n_groups"] = cat.n_grp
            rows.append(rec)
            used.append(n)
            if verbose:
                print("  [catalog history] snap %3d  z=%6.3f  M*=%.3e  SFR=%.3e  M_BH=%.3e  sub %d/%d"
                      % (n, rec["z"], rec.get("mstar_msun", np.nan), rec.get("sfr_msun_per_yr", np.nan),
                         rec.get("bh_mass_msun", np.nan), i, cat.n_sub))
        except Exception as exc:
            if verbose:
                print("  [catalog history] snap %3d failed: %s: %s" % (n, type(exc).__name__, exc))
            if os.environ.get("THESAN_TMZ_TRACEBACK"):
                traceback.print_exc()

    if not rows:
        raise RuntimeError("no catalogue could be read under %s" % output_dir)

    hist = {}
    for k in sorted({k for r in rows for k in r}):
        present = [r[k] for r in rows if k in r and r[k] is not None]
        if not present or isinstance(present[0], str):
            continue
        shape = np.shape(present[0])
        if shape == ():
            hist[k] = np.asarray([float(r[k]) if k in r and r[k] is not None else np.nan for r in rows])
        else:
            blank = np.full(shape, np.nan)
            hist[k] = np.asarray([np.asarray(r[k], float) if (k in r and np.shape(r[k]) == shape) else blank
                                  for r in rows])
    hist["snapnum"] = np.asarray(used, dtype=np.float64)
    # No apertures: Subfind's own boundary is the aperture.  ``key`` sees the
    # NaN and falls through to CATALOG_EQUIVALENT.
    hist["_radii_kpc"] = np.asarray([np.nan])
    if "sigma_kms" in hist:
        hist["sigma_star_kms"] = hist["sigma_kms"]  # the name the plots use
    hist["_units"] = units
    hist["_source"] = "catalog"
    return hist
