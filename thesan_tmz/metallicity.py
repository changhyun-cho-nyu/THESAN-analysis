"""
Metallicity of the target galaxy against the clean-central population and
observations (fig 05 MZR panels b-c, fig 24 stellar MZR, fig 25 iron; report
block METALLICITY), after Rahul's ``metallicity.py`` / ``k25_figures.py``.

Sample: the "clean centrals" of :func:`contamination.clean_flags` (central, no
type 2/3 particle within the group's R200c, >= 10 star particles); the target
galaxy is drawn separately.  Catalogue quantities (``SubhaloGasMetalFractions*``,
``Subhalo*MetallicityHalfRad``, ``SubhaloGasDustMetallicity*``) are those AREPO's
Subfind writes; a panel whose fields are missing says so.

Abundances from GFM mass fractions (element order H, He, C, N, O, Ne, Mg, Si,
Fe, ``begrun.cc:330``):
    12 + log(O/H) = 12 + log10((X_O / 16) / X_H)
    [X/H]          = log10((X_X / A_X) / X_H) + 12 - log eps_sun(X)
with Asplund+09 log eps(O) = 8.69, log eps(Fe) = 7.50, and Z_sun = 0.0127
(0.01359 in a GMC_SF_FB build), the value AREPO itself uses
(``GFM/cooling_metal_vars.h``).  Gas-phase metals exclude what is in dust.

Stellar <[Fe/H]> is the mass-weighted mean of the bound star particles' [Fe/H]
(floored at -5; Pop III / pristine particles have no Fe), from the
``SubhaloOffsetType`` slices -- the analogue of Kirby+13's mean [Fe/H] of
individual stars (theirs is weighted by measurement errors, at z = 0).
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from . import obs

H_IDX, O_IDX, FE_IDX = 0, 4, 8
A_O, A_FE = 16.0, 55.845
LOGEPS_O, LOGEPS_FE = 8.69, 7.50
FEH_FLOOR = -5.0
ZSTAR_FLOOR = -5.0


def solar_z(snap):
    """AREPO's GFM_SOLAR_METALLICITY for this build."""
    cfg = getattr(snap, "config", {}) or {}
    return 0.01359 if io.has_option(cfg, "GMC_SF_FB") else 0.0127


def log_oh(frac):
    """12 + log(O/H) from a metal-fraction array [..., >= 9]."""
    f = np.asarray(frac, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 12.0 + np.log10((f[..., O_IDX] / A_O) / f[..., H_IDX])


def x_h(frac, idx, a_x, logeps):
    """[X/H] from a metal-fraction array."""
    f = np.asarray(frac, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10((f[..., idx] / a_x) / f[..., H_IDX]) + 12.0 - logeps


def star_feh_means(metals, mass, tform, off, ln, subs):
    """Per subhalo in ``subs``: mass-weighted mean particle [Fe/H] (floored) and
    the mass fraction at the floor, from global star arrays and catalogue slices."""
    mean, fpris = np.full(len(subs), np.nan), np.full(len(subs), np.nan)
    for k, s in enumerate(subs):
        a, n = int(off[s]), int(ln[s])
        if n == 0:
            continue
        ok = tform[a:a + n] > 0
        if not ok.any():
            continue
        m = mass[a:a + n][ok]
        fe = x_h(metals[a:a + n][ok], FE_IDX, A_FE, LOGEPS_FE)
        fe = np.where(np.isfinite(fe), np.maximum(fe, FEH_FLOOR), FEH_FLOOR)
        mean[k] = float(np.sum(m * fe) / m.sum())
        fpris[k] = float(m[fe <= FEH_FLOOR].sum() / m.sum())
    return mean, fpris


def sample(snap):
    """Clean centrals + target from the catalogue of ``snap``.  Returns a dict
    (population arrays, ``clean`` mask, ``target`` row or None, notes)."""
    from . import catalog, contamination

    cat = catalog.load(snap)
    if cat is None:
        return None
    pop = cat.population(min_star_particles=10, use_inrad=True)
    if pop is None or not pop["count"]:
        return None
    flags, note = contamination.clean_flags(snap, cat)
    idx = pop["subhalo_index"]
    pop["clean"] = flags[idx] if flags is not None else np.zeros(pop["count"], dtype=bool)
    j = cat.target_index()
    t = np.nonzero(idx == j)[0] if j is not None else np.zeros(0, dtype=int)
    pop["target"] = int(t[0]) if t.size else None
    pop["clean"] = pop["clean"] & (np.arange(pop["count"]) != (pop["target"] if pop["target"] is not None else -1))
    pop["clean_note"] = note
    pop["zsun"] = solar_z(snap)
    pop["_cat"] = cat
    return pop


# ---------------------------------------------------------------------------
# fig 05, panels (b) and (c): gas MZR and dust-to-gas of the population
# ---------------------------------------------------------------------------


def draw_gas_mzr(ax, pop):
    """Gas 12+log(O/H) within R_1/2 against M*(<2R_1/2): clean centrals,
    target, JWST relations.  Returns False when the O fractions are missing."""
    from .plotstyle import PALETTE

    fr = pop.get("gasmetfrac_halfrad")
    if fr is None:
        return False
    c, t = pop["clean"], pop["target"]
    ms = pop["mstar"]
    ohr = log_oh(fr)
    ohs = log_oh(pop["gasmetfrac_sfrw"]) if pop.get("gasmetfrac_sfrw") is not None else None
    ok = c & np.isfinite(ohr) & (ms > 0)
    ax.scatter(ms[ok], ohr[ok], s=12, color=PALETTE["grey"], lw=0, label="clean centrals, $<R_{1/2}$ (%d)" % ok.sum())
    if ohs is not None:
        oks = c & np.isfinite(ohs) & (pop["sfr"] > 0) & (ms > 0)
        ax.scatter(ms[oks], ohs[oks], s=14, facecolor="none", edgecolor=PALETTE["grey"], lw=0.8,
                   label="clean centrals, SFR-weighted (%d)" % oks.sum())
    if t is not None and ms[t] > 0:
        ax.scatter(ms[t], ohr[t], s=90, color=PALETTE["vermillion"], edgecolor="white", lw=1.4, zorder=6,
                   label="this galaxy, $<R_{1/2}$")
        if ohs is not None and pop["sfr"][t] > 0:
            ax.scatter(ms[t], ohs[t], s=90, facecolor="none", edgecolor=PALETTE["vermillion"], lw=1.5, zorder=6,
                       label="this galaxy, SFR-weighted")
    g = np.linspace(*obs.CURTI24_LOGM_RANGE, 50)
    ax.plot(10 ** g, obs.curti24_mzr(g), color="k", lw=1.4, label="Curti+24 JADES, $z$ = 6-10")
    cb = np.array(obs.CURTI24_MEDIANS_Z6_10)
    ax.plot(10 ** cb[:, 0], cb[:, 1], "s", color="k", ms=5, label="Curti+24 medians, $z$ = 6-10")
    g = np.linspace(*obs.NAKAJIMA23_LOGM_RANGE, 50)
    ax.plot(10 ** g, obs.nakajima23_mzr(g), color="k", lw=1.2, ls="--", label="Nakajima+23, $z$ = 4-10")
    g = np.linspace(*obs.CHEMERYNSKA24_LOGM_RANGE, 50)
    ax.plot(10 ** g, obs.chemerynska24_mzr(g), color="k", lw=1.2, ls="-.", label="Chemerynska+24, $z$ = 6-8")
    ax.axhline(obs.OH_SOLAR, color=PALETTE["grey"], ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$M_\star(<2R_{1/2})$ [M$_\odot$]")
    ax.set_ylabel(r"$12+\log({\rm O/H})$, gas $<R_{1/2}$ (from $X_{\rm O}$)")
    return True


def draw_dgr(ax, pop):
    """log DGR against gas 12+log(O/H): clean centrals and target."""
    from .plotstyle import PALETTE

    fr, d1, d2 = pop.get("gasmetfrac_halfrad"), pop.get("dgr_halfrad"), pop.get("dgr_sfr")
    if fr is None or d1 is None:
        return False
    oh = log_oh(fr)
    c, t = pop["clean"], pop["target"]
    for d, filled, lab in ((d1, True, "$<R_{1/2}$"), (d2, False, "star-forming gas")):
        if d is None:
            continue
        ok = c & np.isfinite(oh) & (d > 0)
        kw = dict(color=PALETTE["grey"], lw=0) if filled else dict(facecolor="none", edgecolor=PALETTE["grey"], lw=0.8)
        ax.scatter(oh[ok], np.log10(d[ok]), s=12, label="clean centrals, %s" % lab, **kw)
        if t is not None and d[t] > 0:
            kw = (dict(color=PALETTE["vermillion"], edgecolor="white", lw=1.4) if filled
                  else dict(facecolor="none", edgecolor=PALETTE["vermillion"], lw=1.5))
            ax.scatter(oh[t], np.log10(d[t]), s=90, zorder=6, label="this galaxy, %s" % lab, **kw)
    ax.set_xlabel(r"$12+\log({\rm O/H})$, gas $<R_{1/2}$")
    ax.set_ylabel(r"$\log\,$DGR")
    return True


# ---------------------------------------------------------------------------
# fig 24: stellar MZR
# ---------------------------------------------------------------------------


def plot_stellar_mzr(pop, outpath, title=None, z=None):
    from .plotstyle import PALETTE, label_panel, plt, savefig

    zs, zg = pop.get("zstar_halfrad"), pop.get("zgas_halfrad")
    if zs is None:
        print("  [metallicity] no SubhaloStarMetallicityHalfRad -- skipping fig 24")
        return None
    c, t, ms, zsun = pop["clean"], pop["target"], pop["mstar"], pop["zsun"]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.6))
    a = ax[0]
    ok = c & (zs > 0) & (ms > 0)
    lms, lz = np.log10(ms[ok]), np.log10(zs[ok] / zsun)
    lo = lz < ZSTAR_FLOOR
    a.scatter(10 ** lms[~lo], lz[~lo], s=12, color=PALETTE["grey"], lw=0, label="clean centrals (%d)" % ok.sum())
    a.scatter(10 ** lms[lo], np.full(lo.sum(), ZSTAR_FLOOR + 0.12), s=16, marker="v", color=PALETTE["grey"], lw=0,
              label=r"$Z_\star < 10^{%g} Z_\odot$ (%d)" % (ZSTAR_FLOOR, lo.sum()))
    if lms.size:
        edges = np.arange(np.floor(lms.min() * 2) / 2, lms.max() + 0.5, 0.5)
        cen, med = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            b = (lms >= e0) & (lms < e1)
            if b.sum() >= 5:
                cen.append(np.median(lms[b]))
                med.append(np.median(lz[b]))
        if cen:
            a.plot(10 ** np.array(cen), med, color=PALETTE["blue"], lw=1.6, label=r"median (bins with $\geq$5)")
    if t is not None and zs[t] > 0:
        a.scatter(ms[t], np.log10(zs[t] / zsun), s=90, color=PALETTE["vermillion"], edgecolor="white", lw=1.4,
                  zorder=6, label="this galaxy, stars")
        if zg is not None and zg[t] > 0:
            a.scatter(ms[t], np.log10(zg[t] / zsun), s=90, facecolor="none", edgecolor=PALETTE["vermillion"], lw=1.5,
                      zorder=6, label="this galaxy, gas (reference)")
    a.axhline(0, color=PALETTE["grey"], lw=0.8, ls=":")
    a.set_ylim(ZSTAR_FLOOR - 0.3, 1.0)
    a.set_xscale("log")
    a.set_xlabel(r"$M_\star(<2R_{1/2})$ [M$_\odot$]")
    a.set_ylabel(r"$\log(Z_\star/Z_\odot)$, stars $<R_{1/2}$")
    a.legend(fontsize=7, loc="lower right")
    label_panel(a, "(a) stellar mass-metallicity", loc="upper left")
    b = ax[1]
    if zg is not None:
        ok2 = c & (zs > 0) & (zg > 0) & (ms > 0)
        r = np.log10(zs[ok2] / zg[ok2])
        lo2 = r < ZSTAR_FLOOR
        b.scatter(ms[ok2][~lo2], r[~lo2], s=12, color=PALETTE["grey"], lw=0, label="clean centrals (%d)" % ok2.sum())
        b.scatter(ms[ok2][lo2], np.full(lo2.sum(), ZSTAR_FLOOR + 0.12), s=16, marker="v", color=PALETTE["grey"], lw=0)
        if t is not None and zs[t] > 0 and zg[t] > 0:
            b.scatter(ms[t], np.log10(zs[t] / zg[t]), s=90, color=PALETTE["vermillion"], edgecolor="white", lw=1.4,
                      zorder=6, label="this galaxy")
        b.axhline(0, color=PALETTE["grey"], lw=0.8, ls=":")
        b.set_ylim(ZSTAR_FLOOR - 0.3, 1.0)
        b.legend(fontsize=7, loc="lower right")
    else:
        b.text(0.5, 0.5, "no SubhaloGasMetallicityHalfRad", transform=b.transAxes, ha="center", color="0.5")
    b.set_xscale("log")
    b.set_xlabel(r"$M_\star(<2R_{1/2})$ [M$_\odot$]")
    b.set_ylabel(r"$\log(Z_\star/Z_{\rm gas})$, $<R_{1/2}$")
    label_panel(b, "(b) stars vs gas", loc="upper left")
    fig.suptitle("%s%s   Z$_\\odot$ = %g (AREPO's GFM_SOLAR_METALLICITY); gas excludes dust"
                 % (title or "", ("   z = %.2f" % z) if z is not None else "", pop["zsun"]), fontsize=9)
    fig.tight_layout()
    return savefig(fig, outpath)


# ---------------------------------------------------------------------------
# fig 25: iron
# ---------------------------------------------------------------------------


def star_arrays(snap):
    st = snap.part(io.PT_STAR, ["Masses", "GFM_StellarFormationTime", "GFM_Metals"])
    if not st.get("count") or "GFM_Metals" not in st:
        return None
    return (np.asarray(st["GFM_Metals"], dtype=np.float64), np.asarray(st["Masses"], dtype=np.float64) * snap.units.mass_to_msun,
            np.asarray(st["GFM_StellarFormationTime"], dtype=np.float64))


def iron(snap, pop):
    """Per-population iron numbers and the target's [Fe/H], [O/Fe] particles.
    Returns a dict, or (None, reason)."""
    cat = pop["_cat"]
    off, ln = cat.s("offsettype"), cat.s("lentype")
    arr = star_arrays(snap)
    out = {"note": []}
    if arr is None:
        out["note"].append("no GFM_Metals for stars: stellar [Fe/H] panels left out")
    if off is None:
        out["note"].append("no SubhaloOffsetType: stellar [Fe/H] panels left out")
    idx = pop["subhalo_index"]
    if arr is not None and off is not None:
        metals, mass, tform = arr
        off4, ln4 = np.asarray(off)[:, io.PT_STAR], np.asarray(ln)[:, io.PT_STAR]
        mean, fpris = star_feh_means(metals, mass, tform, off4, ln4, idx)
        out.update(feh_star=mean, fpris=fpris)
        t = pop["target"]
        if t is not None:
            a, n = int(off4[idx[t]]), int(ln4[idx[t]])
            ok = tform[a:a + n] > 0
            met = metals[a:a + n][ok]
            out["target_m"] = mass[a:a + n][ok]
            out["target_feh"] = x_h(met, FE_IDX, A_FE, LOGEPS_FE)
            out["target_ofe"] = x_h(met, O_IDX, A_O, LOGEPS_O) - out["target_feh"]
    if pop.get("starmetfrac") is not None:
        out["feh_star_cat"] = x_h(pop["starmetfrac"], FE_IDX, A_FE, LOGEPS_FE)
    if pop.get("gasmetfrac_halfrad") is not None:
        out["feh_gas"] = x_h(pop["gasmetfrac_halfrad"], FE_IDX, A_FE, LOGEPS_FE)
        out["oh_gas_x"] = x_h(pop["gasmetfrac_halfrad"], O_IDX, A_O, LOGEPS_O)
    return out


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def plot_iron(pop, fe, outpath, title=None, z=None):
    from .plotstyle import LogNorm, PALETTE, label_panel, plt, savefig

    c, t = pop["clean"], pop["target"]
    mtot = pop.get("mstar_total", pop["mstar"])
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.8))
    a = ax[0]
    if "feh_star" in fe:
        m_ = fe["feh_star"]
        ok = c & np.isfinite(m_) & (mtot > 0)
        a.scatter(mtot[ok], m_[ok], s=14, color=PALETTE["grey"], lw=0, label=r"clean centrals, $\langle$[Fe/H]$\rangle_\star$ (%d)" % ok.sum())
        if "feh_star_cat" in fe:
            okc = ok & np.isfinite(fe["feh_star_cat"])
            a.scatter(mtot[okc], np.maximum(fe["feh_star_cat"][okc], FEH_FLOOR), s=10, facecolor="none",
                      edgecolor=PALETTE["grey"], lw=0.6, label="clean centrals, log of mean Fe fraction")
        if t is not None and np.isfinite(m_[t]):
            a.scatter(mtot[t], m_[t], s=90, color=PALETTE["vermillion"], edgecolor="white", lw=1.4, zorder=6,
                      label=r"this galaxy, $\langle$[Fe/H]$\rangle_\star$ = %+.2f" % m_[t])
        g = np.linspace(*obs.KIRBY13_LOGM_RANGE, 50)
        a.plot(10 ** g, obs.kirby13_feh(g), color="k", lw=1.4, label="Kirby+13 Local Group dwarfs ($z$ = 0)")
        a.fill_between(10 ** g, obs.kirby13_feh(g) - obs.KIRBY13[3], obs.kirby13_feh(g) + obs.KIRBY13[3], color="k",
                       alpha=0.1, lw=0)
        lo_hi = [np.log10(max(mtot[ok].min(), 1.0)) if ok.any() else 3.0, np.log10(mtot.max()) if mtot.size else 10.0]
        for e0, e1 in ((min(lo_hi[0], obs.KIRBY13_LOGM_RANGE[0]), obs.KIRBY13_LOGM_RANGE[0]),
                       (obs.KIRBY13_LOGM_RANGE[1], max(lo_hi[1], obs.KIRBY13_LOGM_RANGE[1]))):
            if e1 > e0:
                g = np.linspace(e0, e1, 20)
                a.plot(10 ** g, obs.kirby13_feh(g), color="k", lw=1.0, ls="--")
        a.axhline(FEH_FLOOR, color="0.6", lw=0.6)
        a.set_ylim(FEH_FLOOR - 0.3, 0.8)
        a.legend(fontsize=6.5, loc="upper left")
    else:
        a.text(0.5, 0.5, "\n".join(fe["note"]), transform=a.transAxes, ha="center", color="0.5", fontsize=8)
    a.set_xscale("log")
    a.set_xlabel(r"$M_\star$ (all bound stars) [M$_\odot$]")
    a.set_ylabel("stellar [Fe/H]")
    label_panel(a, "(a) stellar iron-mass (dashed: outside Kirby+13's range)", loc="lower right")

    b = ax[1]
    if "feh_gas" in fe:
        ms = pop["mstar"]
        fg = fe["feh_gas"]
        ok = c & np.isfinite(fg) & (ms > 0)
        lo = fg < FEH_FLOOR
        b.scatter(ms[ok & ~lo], fg[ok & ~lo], s=14, color=PALETTE["grey"], lw=0, label="clean centrals, gas $<R_{1/2}$ (%d)" % ok.sum())
        b.scatter(ms[ok & lo], np.full((ok & lo).sum(), FEH_FLOOR + 0.12), s=16, marker="v", color=PALETTE["grey"], lw=0)
        if t is not None and np.isfinite(fg[t]):
            b.scatter(ms[t], fg[t], s=90, color=PALETTE["vermillion"], edgecolor="white", lw=1.4, zorder=6,
                      label="this galaxy, gas [Fe/H] = %+.2f" % fg[t])
            b.scatter(ms[t], fe["oh_gas_x"][t], s=90, marker="D", facecolor="none", edgecolor=PALETTE["vermillion"],
                      lw=1.4, zorder=6, label="this galaxy, gas [O/H] = %+.2f" % fe["oh_gas_x"][t])
        b.axhline(0, color="0.6", lw=0.8, ls=":")
        b.set_ylim(FEH_FLOOR - 0.3, 0.8)
        b.legend(fontsize=7, loc="lower right")
    else:
        b.text(0.5, 0.5, "no SubhaloGasMetalFractionsHalfRad", transform=b.transAxes, ha="center", color="0.5")
    b.set_xscale("log")
    b.set_xlabel(r"$M_\star(<2R_{1/2})$ [M$_\odot$]")
    b.set_ylabel("gas-phase [Fe/H], $<R_{1/2}$ (dust excluded)")
    label_panel(b, "(b) gas-phase iron", loc="upper left")

    d = ax[2]
    if "target_feh" in fe:
        m, feh, ofe = fe["target_m"], fe["target_feh"], fe["target_ofe"]
        ok = np.isfinite(feh) & np.isfinite(ofe)
        if ok.any():
            H, xe, ye = np.histogram2d(feh[ok], ofe[ok], bins=[np.linspace(-4, 0.8, 97), np.linspace(-0.6, 1.4, 81)],
                                       weights=m[ok])
            d.pcolormesh(xe, ye, np.where(H > 0, H, np.nan).T, cmap="Greys", norm=LogNorm(), shading="auto")
            cen, med = [], []
            for e0 in np.arange(-4, 0.8, 0.25):
                s = ok & (feh >= e0) & (feh < e0 + 0.25)
                if s.sum() >= 20:
                    cen.append(e0 + 0.125)
                    med.append(_wmedian(ofe[s], m[s]))
            d.plot(cen, med, color=PALETTE["vermillion"], lw=1.8, label="mass-weighted median")
        d.axhspan(0.4, 0.6, xmin=0, xmax=3.0 / 4.8, color=PALETTE["blue"], alpha=0.18, lw=0,
                  label=r"MW halo $\alpha$-plateau, [O/Fe] $\approx$ +0.4 to +0.6 (reference only)")
        d.text(0.02, 0.03, "%d particles without Fe or O not shown (%.1f%% of mass)"
               % ((~ok).sum(), 100 * m[~ok].sum() / m.sum() if m.sum() > 0 else 0), transform=d.transAxes,
               fontsize=7, color="0.45")
        d.legend(fontsize=7, loc="upper right")
    else:
        d.text(0.5, 0.5, "\n".join(fe["note"]) or "no target galaxy", transform=d.transAxes, ha="center", color="0.5",
               fontsize=8)
    d.axhline(0, color="0.6", lw=0.8, ls=":")
    d.axvline(0, color="0.6", lw=0.8, ls=":")
    d.set_xlim(-4, 0.8)
    d.set_ylim(-0.6, 1.4)
    d.set_xlabel("[Fe/H] (star particles)")
    d.set_ylabel("[O/Fe]")
    label_panel(d, "(c) this galaxy's bound stars", loc="lower right")
    fig.suptitle("%s%s   Asplund+09: log eps(O) = %.2f, log eps(Fe) = %.2f; [Fe/H] floor %g; Kirby+13 is a z = 0 relation"
                 % (title or "", ("   z = %.2f" % z) if z is not None else "", LOGEPS_O, LOGEPS_FE, FEH_FLOOR), fontsize=9)
    fig.tight_layout()
    return savefig(fig, outpath)


def unavailable(pop, fe):
    """{"24": reason, "25": reason} for the figures that cannot be made."""
    if pop is None:
        return {"24": "no catalogue population", "25": "no catalogue population"}
    out = {}
    if pop.get("zstar_halfrad") is None:
        out["24"] = "catalogue has no SubhaloStarMetallicityHalfRad"
    if fe is None or not any(k in fe for k in ("feh_star", "feh_gas", "target_feh")):
        out["25"] = "; ".join(((fe or {}).get("note") or []) + ["no SubhaloGasMetalFractionsHalfRad"])
    return out


def report_lines(pop, fe):
    out = ["METALLICITY (clean centrals; Z_sun = %s)" % (pop["zsun"] if pop else "n/a")]
    if pop is None:
        return "\n".join(out + ["  no catalogue population"])
    out.append("  " + pop["clean_note"])
    t = pop["target"]
    if t is None:
        return "\n".join(out + ["  no target galaxy in the population"])
    ms = pop["mstar"]
    line = "  this galaxy: log M*(<2R1/2) = %.2f" % np.log10(max(ms[t], 1e-30))
    if pop.get("gasmetfrac_halfrad") is not None:
        line += ", 12+log(O/H) gas <R1/2 = %.2f" % log_oh(pop["gasmetfrac_halfrad"])[t]
    if pop.get("gasmetfrac_sfrw") is not None:
        line += " (SFR-weighted %.2f)" % log_oh(pop["gasmetfrac_sfrw"])[t]
    if pop.get("zstar_halfrad") is not None:
        line += ", Z*/Zsun = %.3g" % (pop["zstar_halfrad"][t] / pop["zsun"])
    if pop.get("zgas_halfrad") is not None:
        line += ", Zgas/Zsun = %.3g" % (pop["zgas_halfrad"][t] / pop["zsun"])
    out.append(line)
    if fe is not None and "feh_star" in fe and np.isfinite(fe["feh_star"][t]):
        mt = pop.get("mstar_total", ms)[t]
        out.append("  this galaxy: <[Fe/H]>* = %+.2f (Kirby+13 at its mass %+.2f), pristine mass fraction %.3f"
                   % (fe["feh_star"][t], obs.kirby13_feh(np.log10(mt)), fe["fpris"][t]))
    if fe is not None and "feh_gas" in fe:
        out.append("  this galaxy: gas [Fe/H] = %+.2f, gas [O/H] = %+.2f" % (fe["feh_gas"][t], fe["oh_gas_x"][t]))
    if fe is not None and "target_feh" in fe:
        m, feh, ofe = fe["target_m"], fe["target_feh"], fe["target_ofe"]
        ok = np.isfinite(feh) & np.isfinite(ofe)
        for lab, s in (("[Fe/H] < -1", ok & (feh < -1)), ("[Fe/H] > -0.5", ok & (feh > -0.5))):
            if s.any():
                out.append("  this galaxy's stars with %s: %.1f%% of mass, median [O/Fe] = %+.2f"
                           % (lab, 100 * m[s].sum() / m[ok].sum(), _wmedian(ofe[s], m[s])))
    c = pop["clean"]
    if fe is not None and "feh_star" in fe:
        mt = pop.get("mstar_total", ms)
        for lo, hi in ((3, 5), (5, 6), (6, 8), (8, 11)):
            b = c & np.isfinite(fe["feh_star"]) & (np.log10(np.maximum(mt, 1e-30)) >= lo) & (np.log10(np.maximum(mt, 1e-30)) < hi)
            if b.any():
                r = fe["feh_star"][b] - obs.kirby13_feh(np.log10(mt[b]))
                out.append("  clean centrals log M* %d-%d: N = %d, median <[Fe/H]>* = %+.2f, median offset from Kirby+13 = %+.2f"
                           % (lo, hi, b.sum(), np.median(fe["feh_star"][b]), np.median(r)))
    if pop.get("zstar_halfrad") is not None and pop.get("zgas_halfrad") is not None:
        b = c & (pop["zstar_halfrad"] > 0) & (pop["zgas_halfrad"] > 0)
        if b.any():
            out.append("  clean centrals: median Z*/Zgas = %.2f (%d)" % (np.median(pop["zstar_halfrad"][b] / pop["zgas_halfrad"][b]), b.sum()))
    return "\n".join(out)
