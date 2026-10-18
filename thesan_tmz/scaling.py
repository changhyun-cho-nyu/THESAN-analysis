"""
Scaling relations, compared against observational and semi-empirical constraints.

Two complementary views:

  * the **evolutionary track** of the central galaxy through each plane, from
    the snapshot history -- this is what tells you whether the model gets to
    the right place and by what route;
  * the **population** of resolved subhaloes in a single snapshot -- for a zoom
    this is only the handful of objects in the high-resolution region, so treat
    it as a sanity check rather than a statistical statement.

A resolution caveat that applies to all of them: a subhalo with fewer than
~100 star particles has a stellar mass that is dominated by Poisson noise and a
velocity dispersion that is meaningless.  :func:`population_from_catalog`
enforces a particle-count floor rather than a mass floor, so the cut follows
the actual resolution of the run.
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from . import catalog, history, obs
from .plotstyle import COMPONENT_COLORS as C
from .plotstyle import PALETTE, label_panel, plt, running_percentiles, savefig, shade_band

# ---------------------------------------------------------------------------
# Extracting a population from the Subfind catalogue
# ---------------------------------------------------------------------------

def population_from_catalog(snap, min_star_particles=10, use_inrad=True, centrals_only=False,
                            min_highres=None, cat=None):
    """Physical-unit table of *individual galaxies* from one snapshot.

    Every number here is a Subhalo property.  For a THESAN-Zoom run the FoF
    Group is the whole zoom system -- the main halo and all its satellites
    together -- so a Group-level stellar mass is the sum over several galaxies
    and belongs in no galaxy-scale plot.  The one Group quantity used is
    ``Group_M_Crit200``, attached to each galaxy through ``SubhaloGroupNr`` and
    reported as ``m200c_central`` only for the galaxies that are actually their
    group's central; for a satellite it is NaN, because a satellite's host halo
    mass is not the x-axis of the SMHM relation.

    ``use_inrad`` selects ``SubhaloMassInRadType`` (mass inside twice the
    stellar half-mass radius, the closest thing to an observed aperture) over
    the total bound mass.  The two differ by ~0.2 dex for a typical galaxy,
    which is the size of the offsets people argue about in the SMHM relation,
    so the choice is recorded in the returned ``aperture`` key.

    The floor is a *particle count*, so it follows the run's real resolution.
    ``min_highres`` additionally cuts on ``SubhaloHighResStellarFraction``,
    which is the only reliable way to keep contaminated haloes from a zoom's
    padding region out of the plot.
    """
    if cat is None:
        cat = catalog.load(snap)
    if cat is None:
        return None
    pop = cat.population(
        min_star_particles=min_star_particles,
        use_inrad=use_inrad,
        centrals_only=centrals_only,
        min_highres=min_highres,
    )
    if pop is None:
        return None
    pop["z"] = snap.z
    pop["age_gyr"] = snap.age_gyr
    # ``m200c`` on a satellite is its host's; only the central's is the SMHM
    # x-axis.  Point the plotting name at the masked version.
    if "m200c_central" in pop:
        pop["m200c_all"] = pop["m200c"]
        pop["m200c"] = pop["m200c_central"]
    if "central" in pop:
        pop["is_central"] = pop["central"]
    return pop


# ---------------------------------------------------------------------------
# Individual relation plots
# ---------------------------------------------------------------------------


def _aperture_label(hist):
    """How to describe the aperture the track was measured in.

    A catalogue history has none -- Subfind's own boundary is the aperture --
    and printing ``M* < nan kpc`` on the figure is worse than saying so.
    """
    radii = np.asarray(hist.get("_radii_kpc", []), dtype=np.float64)
    if radii.size and np.isfinite(radii).any():
        return r"$M_\star<%g$ kpc" % float(np.nanmax(radii))
    return "Subfind aperture"


def _track_colour(fig, ax, x, y, z, label=None, cmap="viridis", s=22):
    """Scatter an evolutionary track colour-coded by redshift, joined by a line."""
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.plot(x[good], y[good], "-", color=PALETTE["grey"], lw=0.8, zorder=2)
    sc = ax.scatter(x[good], y[good], c=z[good], cmap=cmap, s=s, zorder=3, edgecolor="k", linewidth=0.3, label=label)
    return sc


def plot_smhm(hist, units, outpath, pop=None, title=None):
    """Stellar mass -- halo mass, with Moster+13 and Behroozi+13."""
    R = float(np.max(hist["_radii_kpc"]))
    mh = hist.get("mhalo_msun")
    ms = history.key(hist, "mstar", R)
    if mh is None or not np.isfinite(mh).any():
        print("  [scaling] no halo masses (group catalogue missing) -- skipping SMHM")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    grid = np.logspace(8.5, 13.5, 200)
    # Panel (a) shows the whole track, so anchor its reference curves at z = 0;
    # panel (b) is one snapshot, so anchor them at that snapshot's redshift.
    z_pop = float(pop["z"]) if (pop is not None and pop.get("count")) else float(np.nanmin(hist["z"]))
    for a, zref in zip(ax, (0.0, z_pop)):
        for f, name, col in (
            (obs.moster13_smhm, "Moster+13", PALETTE["blue"]),
            (obs.behroozi13_smhm, "Behroozi+13", PALETTE["vermillion"]),
        ):
            y = f(grid, z=zref)
            a.plot(grid, y, color=col, lw=1.3, label=r"%s ($z=%.1f$)" % (name, zref))
            lo, hi = obs.scatter_band(y, 0.2)
            shade_band(a, grid, lo, hi, col, alpha=0.12)
        a.plot(grid, grid * units.OmegaBaryon / units.Omega0, color=PALETTE["grey"], ls=":", lw=1.0, label=r"$f_{\rm b}\,M_{\rm h}$")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(r"$M_{\rm 200c}$ [M$_\odot$]")
        a.set_ylabel(r"$M_\star$ [M$_\odot$]")

    sc = _track_colour(fig, ax[0], mh, ms, hist["z"], label="central, this run")
    fig.colorbar(sc, ax=ax[0], label="redshift")
    ax[0].legend(loc="upper left", fontsize=7)
    label_panel(ax[0], "(a) track of the central (%s)" % _aperture_label(hist), loc="lower right")

    if pop is not None and pop["count"] and "m200c" in pop:
        ax[1].scatter(pop["m200c"], pop["mstar"], s=14, color=PALETTE["black"], alpha=0.6, label=r"subhaloes, $z=%.2f$" % pop["z"])
    ax[1].legend(loc="upper left", fontsize=7)
    label_panel(ax[1], "(b) population at the final snapshot", loc="lower right")

    for a in ax:
        a.set_xlim(10 ** 8.5, 10 ** 13.5)
        a.set_ylim(1e4, 1e12)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_mbh_mstar(hist, units, outpath, pop=None, title=None, show_jwst_band=True):
    """M_BH -- M_star, with the local relations and the JWST high-z band."""
    R = float(np.max(hist["_radii_kpc"]))
    ms = history.key(hist, "mstar", R)
    mb = hist.get("bh_mass_msun")
    if mb is None or not np.isfinite(mb).any():
        print("  [scaling] no BH masses -- skipping M_BH-M_star")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2))
    grid = np.logspace(6.0, 12.0, 200)

    for a in ax:
        y = obs.kormendy_ho13_mbh_mbulge(grid)
        a.plot(grid, y, color=PALETTE["vermillion"], lw=1.3, label="Kormendy & Ho 13 (bulges)")
        lo, hi = obs.scatter_band(y, obs.KORMENDY_HO_13_SCATTER)
        shade_band(a, grid, lo, hi, PALETTE["vermillion"], alpha=0.12)

        y = obs.reines_volonteri15_agn(grid)
        a.plot(grid, y, color=PALETTE["blue"], lw=1.3, label="Reines & Volonteri 15 (AGN)")
        lo, hi = obs.scatter_band(y, obs.REINES_VOLONTERI_15_SCATTER)
        shade_band(a, grid, lo, hi, PALETTE["blue"], alpha=0.12)

        a.plot(grid, obs.reines_volonteri15_local(grid), color=PALETTE["green"], lw=1.0, ls="--", label="Reines & Volonteri 15 (local E)")

        if show_jwst_band:
            lo, hi = obs.jwst_overmassive_band(grid)
            shade_band(a, grid, lo, hi, PALETTE["purple"], alpha=0.14, label=r"JWST $z>4$ AGN (indicative)")

        a.plot(grid, 1e-3 * grid, color=PALETTE["grey"], ls=":", lw=0.9)
        a.text(grid[150], 1e-3 * grid[150] * 1.4, r"$10^{-3}M_\star$", fontsize=7, color=PALETTE["grey"], rotation=32)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(r"$M_\star$ [M$_\odot$]")
        a.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
        a.set_xlim(1e6, 1e12)
        a.set_ylim(1e3, 1e11)

    sc = _track_colour(fig, ax[0], ms, mb, hist["z"], label="central, this run")
    fig.colorbar(sc, ax=ax[0], label="redshift")
    ax[0].legend(loc="upper left", fontsize=6.5)
    label_panel(ax[0], "(a) evolutionary track", loc="lower right")

    if pop is not None and pop["count"]:
        m = pop["mbh"] > 0
        if m.any():
            ax[1].scatter(pop["mstar"][m], pop["mbh"][m], s=16, color=PALETTE["black"], alpha=0.7, label=r"subhaloes, $z=%.2f$" % pop["z"])
    ax[1].legend(loc="upper left", fontsize=6.5)
    label_panel(ax[1], "(b) population at the final snapshot", loc="lower right")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_mbh_sigma(hist, units, outpath, pop=None, title=None):
    """M_BH -- sigma, with Kormendy & Ho (2013)."""
    sig = hist.get("sigma_star_kms")
    mb = hist.get("bh_mass_msun")
    if sig is None or mb is None or not (np.isfinite(sig) & np.isfinite(mb)).any():
        print("  [scaling] no sigma or no BH -- skipping M_BH-sigma")
        return None

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    grid = np.logspace(0.7, 2.7, 200)
    y = obs.kormendy_ho13_mbh_sigma(grid)
    ax.plot(grid, y, color=PALETTE["vermillion"], lw=1.4, label="Kormendy & Ho 13")
    lo, hi = obs.scatter_band(y, obs.KORMENDY_HO_13_SCATTER)
    shade_band(ax, grid, lo, hi, PALETTE["vermillion"], alpha=0.14)

    sc = _track_colour(fig, ax, sig, mb, hist["z"], label="central, this run")
    fig.colorbar(sc, ax=ax, label="redshift")

    if pop is not None and pop["count"]:
        m = (pop["mbh"] > 0) & np.isfinite(pop["sigma"])
        if m.any():
            ax.scatter(pop["sigma"][m], pop["mbh"][m], s=14, color=PALETTE["grey"], alpha=0.6, label="subhaloes (final)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_\star$ [km s$^{-1}$]")
    ax.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
    ax.legend(loc="upper left", fontsize=7)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_main_sequence(hist, units, outpath, pop=None, title=None):
    """SFR -- M_star, with the Speagle+14 main sequence at several epochs."""
    R = float(np.max(hist["_radii_kpc"]))
    ms = history.key(hist, "mstar", R)
    sfr = history.key(hist, "sfr", R)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    grid = np.logspace(5.5, 11.5, 200)

    zshow = [z for z in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0) if z <= np.nanmax(hist["z"]) + 2]
    for a in ax:
        for zz in zshow:
            t = float(units.age_gyr(1.0 / (1.0 + zz)))
            a.plot(grid, obs.speagle14_main_sequence(grid, t), lw=1.0, alpha=0.8, label=r"Speagle+14 $z=%g$" % zz)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(r"$M_\star$ [M$_\odot$]")
        a.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
        a.set_xlim(10 ** 5.5, 10 ** 11.5)

    sc = _track_colour(fig, ax[0], ms, sfr, hist["z"], label="central, this run")
    fig.colorbar(sc, ax=ax[0], label="redshift")
    ax[0].legend(loc="upper left", fontsize=6.5, ncol=2)
    label_panel(ax[0], "(a) track", loc="lower right")

    if pop is not None and pop["count"]:
        m = pop["sfr"] > 0
        ax[1].scatter(pop["mstar"][m], pop["sfr"][m], s=14, color=PALETTE["black"], alpha=0.6, label=r"subhaloes, $z=%.2f$" % pop["z"])
        q = ~m
        if q.any():
            ax[1].scatter(pop["mstar"][q], np.full(q.sum(), 1e-5), s=14, marker="v", color=PALETTE["grey"], alpha=0.6, label="SFR = 0")
    ax[1].legend(loc="upper left", fontsize=6.5)
    label_panel(ax[1], "(b) population at the final snapshot", loc="lower right")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_mzr(hist, units, outpath, pop=None, title=None):
    """Mass -- metallicity, gas and stars, against Tremonti+04 / Andrews+13.

    The conversion from GFM's mass-fraction metallicity to 12+log(O/H) assumes
    a solar O/Z ratio, which is optimistic for young alpha-enhanced systems --
    read the offset, not the absolute normalisation.
    """
    R = float(np.max(hist["_radii_kpc"]))
    ms = history.key(hist, "mstar", R)
    Zg = history.key(hist, "Zgas", R)
    if not np.isfinite(Zg).any():
        print("  [scaling] no GFM_Metallicity -- skipping MZR")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    grid = np.logspace(7.0, 11.5, 200)

    for a in ax:
        a.plot(grid, obs.tremonti04_mzr(grid), color=PALETTE["vermillion"], lw=1.3, label="Tremonti+04 (strong line)")
        a.plot(grid, obs.andrews_martini13_mzr(grid), color=PALETTE["blue"], lw=1.3, ls="--", label=r"Andrews & Martini 13 ($T_e$)")
        a.axhline(obs.OH_SOLAR, color=PALETTE["grey"], ls=":", lw=0.9)
        a.text(grid[5], obs.OH_SOLAR + 0.03, "solar", fontsize=7, color=PALETTE["grey"])
        a.set_xscale("log")
        a.set_xlabel(r"$M_\star$ [M$_\odot$]")
        a.set_ylabel(r"$12+\log({\rm O/H})$")
        a.set_ylim(6.5, 9.5)

    sc = _track_colour(fig, ax[0], ms, obs.logOH_from_Zmass(Zg), hist["z"], label="central (gas)")
    fig.colorbar(sc, ax=ax[0], label="redshift")
    ax[0].set_yscale("linear")
    ax[0].legend(loc="lower right", fontsize=7)
    label_panel(ax[0], "(a) gas-phase MZR track")

    if pop is not None and pop["count"] and "Zgas" in pop:
        m = pop["Zgas"] > 0
        ax[1].scatter(pop["mstar"][m], obs.logOH_from_Zmass(pop["Zgas"][m]), s=14, color=C["gas"], alpha=0.7, label="gas")
    if pop is not None and pop["count"] and "Zstar" in pop:
        m = pop["Zstar"] > 0
        ax[1].scatter(pop["mstar"][m], obs.logOH_from_Zmass(pop["Zstar"][m]), s=14, color=C["stars"], alpha=0.7, marker="s", label="stars")
    ax[1].legend(loc="lower right", fontsize=7)
    label_panel(ax[1], r"(b) population at $z=%.2f$" % (pop["z"] if pop else np.nan))

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_gas_fraction_and_size(hist, units, outpath, pop=None, title=None):
    """Gas fraction and stellar size against stellar mass."""
    R = float(np.max(hist["_radii_kpc"]))
    ms = history.key(hist, "mstar", R)

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))

    sc = _track_colour(fig, ax[0], ms, hist.get("fgas", np.full_like(ms, np.nan)), hist["z"])
    fig.colorbar(sc, ax=ax[0], label="redshift")
    ax[0].set_xscale("log")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel(r"$M_\star$ [M$_\odot$]")
    ax[0].set_ylabel(r"$f_{\rm gas}$")
    label_panel(ax[0], "(a) gas fraction")

    rh = hist.get("rhalf_star_kpc")
    if rh is not None and np.isfinite(rh).any():
        sc = _track_colour(fig, ax[1], ms, rh, hist["z"])
        fig.colorbar(sc, ax=ax[1], label="redshift")
    if pop is not None and pop["count"] and "rhalf_star" in pop:
        m = pop["rhalf_star"] > 0
        ax[1].scatter(pop["mstar"][m], pop["rhalf_star"][m], s=12, color=PALETTE["grey"], alpha=0.6, label="subhaloes (final)")
        ax[1].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$M_\star$ [M$_\odot$]")
    ax[1].set_ylabel(r"$R_{1/2,\star}$ [kpc]")
    label_panel(ax[1], "(b) stellar half-mass radius")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_btfr(hist, units, outpath, title=None):
    """Baryonic Tully-Fisher, using the circular velocity at the outer aperture."""
    R = float(np.max(hist["_radii_kpc"]))
    v = history.key(hist, "vcirc", R)
    mb = history.key(hist, "mstar", R) + history.key(hist, "mgas", R)
    if not np.isfinite(v).any():
        return None

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    grid = np.logspace(0.7, 2.6, 100)
    ax.plot(grid, obs.mcgaugh12_btfr(grid), color=PALETTE["vermillion"], lw=1.3, label=r"McGaugh 12: $M_{\rm bar}=47V^4$")
    lo, hi = obs.scatter_band(obs.mcgaugh12_btfr(grid), 0.12)
    shade_band(ax, grid, lo, hi, PALETTE["vermillion"], alpha=0.14)
    sc = _track_colour(fig, ax, v, mb, hist["z"], label="central, this run")
    fig.colorbar(sc, ax=ax, label="redshift")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$v_{\rm circ}(<%g\ {\rm kpc})$ [km s$^{-1}$]" % R)
    ax.set_ylabel(r"$M_\star+M_{\rm gas}$ [M$_\odot$]")
    ax.legend(loc="upper left", fontsize=7)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_all(hist, units, outdir, pop=None, prefix="scaling", title=None):
    """Produce every scaling-relation figure this module knows how to make."""
    import os

    made = []
    for name, fn in (
        ("smhm", plot_smhm),
        ("mbh_mstar", plot_mbh_mstar),
        ("main_sequence", plot_main_sequence),
        ("mzr", plot_mzr),
        ("gasfrac_size", plot_gas_fraction_and_size),
    ):
        p = fn(hist, units, os.path.join(outdir, "%s_%s.png" % (prefix, name)), pop=pop, title=title)
        if p:
            made.append(p)
    for name, fn in (("mbh_sigma", plot_mbh_sigma), ("btfr", plot_btfr)):
        try:
            p = fn(hist, units, os.path.join(outdir, "%s_%s.png" % (prefix, name)), title=title) if fn is plot_btfr else fn(
                hist, units, os.path.join(outdir, "%s_%s.png" % (prefix, name)), pop=pop, title=title
            )
            if p:
                made.append(p)
        except Exception as exc:  # a missing field should not kill the batch
            print("  [scaling] %s failed: %s" % (name, exc))
    return made
