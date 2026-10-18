"""
Diagnostics for every black hole in the run, not just the target galaxy's.

The rest of the package follows *one* black hole -- the galaxy's own, see
:func:`galaxy.measure`.  This module looks at all of them:

  * :func:`bh_table` -- one row per BH particle in a snapshot: mass, accretion
    rate, Eddington ratio, the gas at the BH, the host galaxy, the offset from
    its centre, the cumulative feedback energy;
  * :func:`plot_population` -- the population at one snapshot (fig 16);
  * :func:`plot_tracks` -- every BH's history from ``blackhole_details``, with
    the mergers marked (fig 17).

The Eddington ratio uses the code's own denominator, ``BH_MdotEddington``,
when the snapshot has it, so it is exactly the simulation's lambda_Edd; the
package's :meth:`Units.eddington_ratio` otherwise.

Host galaxy: the subhalo whose centre is nearest in units of its own
:func:`galaxy.target_bh_radius_kpc` (five stellar half-mass radii, else
max(5 kpc, 0.1 R200c)), provided the BH is within that radius; -1 otherwise
("unhosted": wandering, ejected, or in a galaxy too small for Subfind).
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from . import bh_logs
from . import catalog as cat_mod
from .plotstyle import CYCLE, PALETTE, label_panel, plt, savefig
from .units import GAMMA, GAMMA_MINUS1

#: Columns of :func:`bh_table`, in the order :func:`write_bh_table` writes them.
TABLE_COLS = [
    "id", "mass_msun", "mdot_msun_per_yr", "eddington_ratio", "nH_cgs", "csnd_kms", "hsml_kpc",
    "host_subhalo", "host_is_central", "host_mstar_msun", "offset_kpc", "cum_egy_erg",
]


def _subhalo_radii_kpc(cat):
    """target_bh_radius_kpc() for every subhalo at once (kpc, physical)."""
    n = cat.n_sub
    L = cat.units.length_to_kpc(cat.a)
    hr = cat.s("halfmassradtype")
    rhalf = np.asarray(hr, dtype=np.float64)[:, io.PT_STAR] * L if hr is not None else np.zeros(n)
    r200 = np.zeros(n)
    gn, g200 = cat.group_of(), cat.g("r200c")
    if gn is not None and g200 is not None and cat.n_grp:
        r200 = np.asarray(g200, dtype=np.float64)[np.clip(gn, 0, cat.n_grp - 1)] * L
    # the same rule as target_bh_radius_kpc(), vectorised
    return np.where(rhalf > 0, 5.0 * rhalf, np.maximum(5.0, 0.1 * r200))


def bh_table(snap, cat=None):
    """One row per black hole in ``snap``.  Returns a dict of arrays (TABLE_COLS)
    plus ``'count'``; physical units throughout."""
    u, a = snap.units, snap.a
    L, M = u.length_to_kpc(a), u.mass_to_msun
    d = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs", "BH_Mass", "Masses", "BH_Mdot", "BH_MdotEddington",
                             "BH_Density", "BH_U", "BH_Hsml", "BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"])
    n = int(d.get("count", 0))
    if n == 0 or "Coordinates" not in d:
        return {"count": 0}

    mass = np.asarray(d.get("BH_Mass", d.get("Masses")), dtype=np.float64)
    t = {"count": n, "id": np.asarray(d["ParticleIDs"], dtype=np.int64), "mass_msun": mass * M}
    mdot = np.asarray(d["BH_Mdot"], dtype=np.float64) if "BH_Mdot" in d else np.full(n, np.nan)
    t["mdot_msun_per_yr"] = mdot * u.mdot_to_msun_per_yr
    if "BH_MdotEddington" in d:
        medd = np.asarray(d["BH_MdotEddington"], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            t["eddington_ratio"] = np.where(medd > 0, mdot / medd, np.nan)   # the code's own lambda
        t["_eddington_source"] = "BH_MdotEddington"
    else:
        t["eddington_ratio"] = np.asarray(u.eddington_ratio(t["mdot_msun_per_yr"], t["mass_msun"]))
        t["_eddington_source"] = "Units.eddington_ratio"
    # BH_Density is comoving in the snapshot (the details log prints it physical)
    t["nH_cgs"] = np.asarray(d.get("BH_Density", np.full(n, np.nan)), dtype=np.float64) * u.density_to_nH_cgs(a)
    bhu = np.asarray(d.get("BH_U", np.full(n, np.nan)), dtype=np.float64)
    t["csnd_kms"] = np.sqrt(np.maximum(GAMMA * GAMMA_MINUS1 * bhu, 0.0)) * u.velocity_to_kms   # as blackhole_bondi_rate()
    t["hsml_kpc"] = np.asarray(d.get("BH_Hsml", np.full(n, np.nan)), dtype=np.float64) * L
    cum = np.zeros(n)
    for k in ("BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"):
        if k in d:
            cum = cum + np.asarray(d[k], dtype=np.float64)
    t["cum_egy_erg"] = cum * u.energy_to_erg if any(k in d for k in ("BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM")) \
        else np.full(n, np.nan)

    # -- host galaxy ----------------------------------------------------------
    t["host_subhalo"] = np.full(n, -1, dtype=np.int64)
    t["host_is_central"] = np.zeros(n, dtype=bool)
    t["host_mstar_msun"] = np.full(n, np.nan)
    t["offset_kpc"] = np.full(n, np.nan)
    if cat is None:
        cat = cat_mod.load(snap)
    spos = cat.s("pos") if cat is not None and cat.n_sub else None
    if spos is not None:
        rlim = _subhalo_radii_kpc(cat)
        spos = np.asarray(spos, dtype=np.float64)
        cen = cat.is_central()
        mt = cat.s("massinradtype")
        mt = cat.s("masstype") if mt is None else mt
        for i in range(n):
            r = io.radius_from(spos, d["Coordinates"][i], snap.boxsize) * L
            j = int(np.argmin(r / np.maximum(rlim, 1e-30)))
            if r[j] <= rlim[j]:
                t["host_subhalo"][i] = j
                t["host_is_central"][i] = bool(cen[j])
                t["offset_kpc"][i] = float(r[j])
                if mt is not None:
                    t["host_mstar_msun"][i] = float(np.asarray(mt)[j, io.PT_STAR] * M)
    return t


def write_bh_table(path, table, snap=None):
    """Write :func:`bh_table` as an aligned text table, most massive first."""
    with open(path, "w") as fh:
        if snap is not None:
            fh.write("# black holes in %s (z = %.4g); lambda_Edd from %s\n"
                     % (snap, snap.z, table.get("_eddington_source", "?")))
        fh.write("# %s\n" % "  ".join(TABLE_COLS))
        if not table.get("count"):
            return path
        for i in np.argsort(-table["mass_msun"]):
            fh.write("%d  %.5g  %.5g  %.4g  %.4g  %.4g  %.4g  %d  %d  %.5g  %.4g  %.4g\n" % tuple(
                table[k][i] for k in TABLE_COLS))
    print("  [bh_population] wrote %s (%d black holes)" % (path, table["count"]))
    return path


def plot_population(table, outpath, snap=None, title=None):
    """Fig 16: every BH at one snapshot."""
    if not table.get("count"):
        print("  [bh_population] no black holes -- skipping the population figure")
        return None
    cen = table["host_is_central"]
    host = table["host_subhalo"] >= 0
    groups = ((host & cen, "in a central galaxy", PALETTE["blue"], "o"),
              (host & ~cen, "in a satellite", PALETTE["vermillion"], "s"),
              (~host, "no host galaxy", PALETTE["grey"], "x"))
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.0))

    def scatter(a, x, y, logx=True, logy=True):
        for m, lab, col, mk in groups:
            ok = m & np.isfinite(x) & np.isfinite(y) & ((x > 0) if logx else True) & ((y > 0) if logy else True)
            if ok.any():
                a.scatter(x[ok], y[ok], s=26, color=col, marker=mk, label="%s (%d)" % (lab, ok.sum()),
                          edgecolors="none" if mk != "x" else None)
        if logx:
            a.set_xscale("log")
        if logy:
            a.set_yscale("log")

    a = ax[0, 0]
    scatter(a, table["host_mstar_msun"], table["mass_msun"])
    a.set_xlabel(r"host $M_\star$ [M$_\odot$]")
    a.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.legend(fontsize=7, loc="best")
    label_panel(a, "(a) BH mass vs host galaxy", loc="lower right")

    a = ax[0, 1]
    scatter(a, table["mass_msun"], table["eddington_ratio"])
    a.axhline(1.0, color=PALETTE["vermillion"], ls="--", lw=1.0)
    a.axhline(0.01, color=PALETTE["grey"], ls=":", lw=1.0)
    a.set_xlabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.set_ylabel(r"$\lambda_{\rm Edd}$ (%s)" % table.get("_eddington_source", ""))
    label_panel(a, "(b) Eddington ratio", loc="lower right")

    a = ax[1, 0]
    # A BH sitting exactly on its host's centre (offset 0) cannot go on a log
    # axis; draw everything below 10 pc at 10 pc and say so.
    floor = 0.01
    off = np.where(np.isfinite(table["offset_kpc"]), np.maximum(table["offset_kpc"], floor), np.nan)
    scatter(a, table["mass_msun"], off)
    a.axhline(floor, color=PALETTE["grey"], lw=0.6, ls=":")
    a.text(0.02, 0.04, "offsets < 10 pc drawn at 10 pc", transform=a.transAxes, fontsize=6.5, color=PALETTE["grey"])
    a.set_xlabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.set_ylabel("offset from the host centre [kpc]")
    label_panel(a, "(c) BH offsets", loc="lower right")

    a = ax[1, 1]
    scatter(a, table["nH_cgs"], table["mdot_msun_per_yr"])
    a.set_xlabel(r"$n_{\rm H}$ at the BH [cm$^{-3}$]")
    a.set_ylabel(r"$\dot M$ [M$_\odot$ yr$^{-1}$]")
    label_panel(a, "(d) accretion vs environment", loc="lower right")

    ttl = "%d black holes" % table["count"] + ("  (%s, z = %.2f)" % (snap.files[0].split("/")[-1], snap.z) if snap else "")
    fig.suptitle("%s  %s" % (title or "", ttl), y=0.995)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_tracks(details, mergers, units, outpath, params=None, max_bhs=20, nbins=120, title=None):
    """Fig 17: the history of every BH (up to ``max_bhs``, most massive first)
    from ``blackhole_details``, dt-weighted onto a common time grid, with the
    mergers from ``blackhole_mergers`` marked."""
    if details["time"].size == 0:
        print("  [bh_population] blackhole_details empty -- skipping the BH tracks")
        return None
    ids = bh_logs.unique_bh_ids(details)
    peak = np.array([details["bh_mass"][details["id"] == i].max() for i in ids])
    order = ids[np.argsort(-peak)]
    shown = order[:max_bhs]
    t_all = details["time"]
    bins = np.linspace(t_all.min(), t_all.max(), nbins + 1)

    fields = ["bh_mass", "mdot", "rho"]
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    ax = ax.ravel()
    for k, i in enumerate(shown):
        s = bh_logs.select_bh(details, i)
        b = bh_logs.bin_details_by_time(s, bins, fields=fields)
        ok = b["count"] > 0
        tg = units.age_gyr(b["time"][ok])
        col = CYCLE[k % len(CYCLE)]
        ls = "-" if k < len(CYCLE) else "--"
        m = b["bh_mass"][ok] * units.mass_to_msun
        md = b["mdot"][ok] * units.mdot_to_msun_per_yr
        ax[0].plot(tg, m, color=col, ls=ls, lw=1.0, label=str(i))
        ax[1].plot(tg, np.where(md > 0, md, np.nan), color=col, ls=ls, lw=0.9)
        lam = units.eddington_ratio(md, m)
        ax[2].plot(tg, np.where(lam > 0, lam, np.nan), color=col, ls=ls, lw=0.9)
        ax[3].plot(tg, b["rho"][ok] * units.density_to_nH_cgs(1.0), color=col, ls=ls, lw=0.9)   # details rho is physical
    if mergers is not None and np.size(mergers.get("time", [])):
        for tm in units.age_gyr(mergers["time"]):
            for a in ax:
                a.axvline(tm, color=PALETTE["grey"], lw=0.6, alpha=0.6)
    for a, lab, yl in ((ax[0], "(a) BH mass", r"$M_{\rm BH}$ [M$_\odot$]"),
                       (ax[1], "(b) accretion rate (pre-cap log value)", r"$\dot M$ [M$_\odot$ yr$^{-1}$]"),
                       (ax[2], r"(c) Eddington ratio", r"$\lambda_{\rm Edd}$"),
                       (ax[3], "(d) gas density at the BH", r"$n_{\rm H}$ [cm$^{-3}$]")):
        a.set_yscale("log")
        a.set_ylabel(yl)
        label_panel(a, lab)
    ax[2].axhline(1.0, color=PALETTE["vermillion"], ls="--", lw=1.0)
    for a in ax[2:]:
        a.set_xlabel("age of universe [Gyr]")
    ax[0].legend(fontsize=6, ncol=2, loc="lower right", title="BH ID", title_fontsize=6)
    ttl = "%d black holes in blackhole_details%s; grey lines = mergers" % (
        ids.size, (", the %d most massive shown" % len(shown)) if ids.size > len(shown) else "")
    fig.suptitle("%s  %s" % (title or "", ttl), y=0.995)
    fig.tight_layout()
    return savefig(fig, outpath)
