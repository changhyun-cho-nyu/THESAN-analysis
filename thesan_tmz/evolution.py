"""
Time-evolution figures: how the central galaxy and its black hole grow.

Two independent sources feed these plots and it is worth knowing which is
which, because they disagree in informative ways:

  * the **snapshots** (via :mod:`history`) give aperture-resolved masses but
    only at the snapshot cadence, typically tens of Myr apart;
  * the **logs** (``blackholes.txt``, ``sfr.txt``, ``blackhole_details``) are
    written every timestep and resolve bursts the snapshots alias away.

When a duty cycle looks smooth in the snapshot series and spiky in the log,
the log is right.
"""

from __future__ import annotations

import os

import numpy as np

from . import bh_logs, history, obs
from .plotstyle import COMPONENT_COLORS as C
from .plotstyle import PALETTE, add_redshift_axis, label_panel, plt, savefig, shade_band

# ---------------------------------------------------------------------------
# The 12-panel overview
# ---------------------------------------------------------------------------


def plot_overview(hist, units, outpath, radius=None, title=None, logs=None):
    """Twelve-panel summary of the central galaxy's evolution.

    Panels: BH mass, stellar mass, gas mass, gas by phase, SFR, BH accretion
    rate, Eddington ratio, mean gas density, gas volume density, gas fraction,
    specific SFR, BH-to-stellar mass ratio.

    ``logs`` is an optional dict from :func:`collect_logs`; when given, the
    SFR and accretion-rate panels also show the full-cadence log curves in
    grey behind the snapshot points.
    """
    R = float(np.max(hist["_radii_kpc"])) if radius is None else float(radius)
    t = hist["age_gyr"]
    z = hist["z"]

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 13.0), sharex=True)
    ax = axes.ravel()

    def K(base, r=R):
        return history.key(hist, base, r)

    # 1. black hole mass ----------------------------------------------------
    a = ax[0]
    a.plot(t, np.where(hist.get("bh_mass_msun", np.nan) > 0, hist["bh_mass_msun"], np.nan), color=C["bh"], label=r"$M_{\rm BH}$ (in aperture)")
    if "bh_mass_max_msun" in hist:
        a.plot(t, hist["bh_mass_max_msun"], color=C["bh"], ls="--", lw=1.0, label="most massive BH")
    if logs and logs.get("blackholes", {}).get("time", np.zeros(0)).size:
        b = logs["blackholes"]
        a.plot(units.age_gyr(b["time"]), b["mass_bh"] * units.mass_to_msun, color=PALETTE["grey"], lw=0.8, zorder=0, label="all BHs (log)")
    a.set_yscale("log")
    a.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.legend(loc="lower right")
    label_panel(a, "(a) black hole mass")

    # 2. stellar mass -------------------------------------------------------
    a = ax[1]
    for r, ls in zip(hist["_radii_kpc"], ["-", "--", "-.", ":", "-"]):
        a.plot(t, history.key(hist, "mstar", float(r)), ls=ls, lw=1.2, label=r"$<%g$ kpc" % r)
    a.set_yscale("log")
    a.set_ylabel(r"$M_\star$ [M$_\odot$]")
    a.legend(loc="lower right", ncol=2)
    label_panel(a, "(b) stellar mass")

    # 3. gas mass -----------------------------------------------------------
    a = ax[2]
    a.plot(t, K("mgas"), color=C["gas"], label=r"gas $<%g$ kpc" % R)
    a.plot(t, K("mstar"), color=C["stars"], lw=1.0, ls="--", label=r"stars $<%g$ kpc" % R)
    if "mhalo_msun" in hist:
        a.plot(t, hist["mhalo_msun"], color=C["dm"], lw=1.0, ls=":", label=r"$M_{\rm 200c}$")
    a.set_yscale("log")
    a.set_ylabel(r"mass [M$_\odot$]")
    a.legend(loc="lower right")
    label_panel(a, "(c) mass budget")

    # 4. gas by phase -------------------------------------------------------
    a = ax[3]
    for ph, lab in (("cold", r"$T<10^4$ K"), ("warm", r"$10^4$--$10^5$ K"), ("hot", r"$T>10^5$ K")):
        y = K("mgas_%s" % ph)
        if np.isfinite(y).any():
            a.plot(t, y, color=C["gas_%s" % ph], label=lab)
    y = K("mgas_dense")
    if np.isfinite(y).any():
        a.plot(t, y, color=PALETTE["purple"], ls="--", lw=1.0, label=r"cold & $n_{\rm H}>0.1$")
    a.set_yscale("log")
    a.set_ylabel(r"$M_{\rm gas}$ by phase [M$_\odot$]")
    a.legend(loc="lower right")
    label_panel(a, "(d) gas phases")

    # 5. SFR ----------------------------------------------------------------
    a = ax[4]
    if logs and logs.get("sfr", {}).get("time", np.zeros(0)).size:
        s = logs["sfr"]
        a.plot(units.age_gyr(s["time"]), s["sfr_msun_per_yr"], color=PALETTE["grey"], lw=0.7, zorder=0, label="sfr.txt (all gas)")
    a.plot(t, K("sfr"), color=C["sfr"], marker="o", ms=2.5, label=r"$<%g$ kpc (snapshots)" % R)
    a.set_yscale("log")
    a.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
    a.legend(loc="lower right")
    label_panel(a, "(e) star formation rate")

    # 6. BH accretion rate --------------------------------------------------
    a = ax[5]
    if logs and logs.get("blackholes", {}).get("time", np.zeros(0)).size:
        b = logs["blackholes"]
        a.plot(units.age_gyr(b["time"]), b["mdot_msun_per_yr"], color=PALETTE["grey"], lw=0.7, zorder=0, label="blackholes.txt")
    y = hist.get("bh_mdot_msun_per_yr")
    if y is not None:
        a.plot(t, np.where(y > 0, y, np.nan), color=C["bh"], marker="o", ms=2.5, label="snapshots")
    a.set_yscale("log")
    a.set_ylabel(r"$\dot M_{\rm BH}$ [M$_\odot$ yr$^{-1}$]")
    a.legend(loc="lower right")
    label_panel(a, "(f) accretion rate")

    # 7. Eddington ratio ----------------------------------------------------
    a = ax[6]
    y = hist.get("eddington_ratio")
    if y is not None:
        a.plot(t, np.where(y > 0, y, np.nan), color=C["bh"], marker="o", ms=2.5)
    a.axhline(1.0, color=PALETTE["vermillion"], ls="--", lw=1.0)
    a.axhline(0.01, color=PALETTE["grey"], ls=":", lw=1.0)
    a.text(0.98, 0.93, r"$\lambda_{\rm Edd}=1$", color=PALETTE["vermillion"], transform=a.transAxes, ha="right", fontsize=7)
    a.set_yscale("log")
    a.set_ylabel(r"$\lambda_{\rm Edd}=\dot M/\dot M_{\rm Edd}$")
    label_panel(a, "(g) Eddington ratio")

    # 8. mean gas density ---------------------------------------------------
    a = ax[7]
    for r in hist["_radii_kpc"]:
        y = history.key(hist, "nH_mean", float(r))
        if np.isfinite(y).any():
            a.plot(t, y, lw=1.2, label=r"$<%g$ kpc" % r)
    a.set_yscale("log")
    a.set_ylabel(r"$\langle n_{\rm H}\rangle$ [cm$^{-3}$]")
    a.legend(loc="lower left", ncol=2, fontsize=7)
    label_panel(a, "(h) mean gas density")

    # 9. gas volume density -------------------------------------------------
    a = ax[8]
    for r in hist["_radii_kpc"]:
        y = history.key(hist, "rho_gas", float(r))
        if np.isfinite(y).any():
            a.plot(t, y, lw=1.2, label=r"$<%g$ kpc" % r)
    a.set_yscale("log")
    a.set_ylabel(r"$\rho_{\rm gas}$ [M$_\odot$ kpc$^{-3}$]")
    label_panel(a, "(i) gas volume density")

    # 10. gas fraction ------------------------------------------------------
    a = ax[9]
    a.plot(t, hist.get("fgas", np.full_like(t, np.nan)), color=C["gas"])
    a.set_ylim(0, 1.05)
    a.set_ylabel(r"$f_{\rm gas}=M_{\rm gas}/(M_{\rm gas}+M_\star)$")
    a.set_xlabel("age of universe [Gyr]")
    label_panel(a, "(j) gas fraction")

    # 11. specific SFR ------------------------------------------------------
    a = ax[10]
    y = hist.get("ssfr_per_gyr")
    if y is not None:
        a.plot(t, np.where(y > 0, y, np.nan), color=C["sfr"], marker="o", ms=2.5, label="simulation")
    zz = np.clip(z, 0, 20)
    a.plot(t, obs.specific_sfr_from_cosmology(zz), color=PALETTE["grey"], ls="--", lw=1.0, label="halo growth limit")
    a.axhline(1.0 / units.age_gyr(1.0 / (1.0 + 0.0)), color=PALETTE["grey"], ls=":", lw=0.8)
    a.set_yscale("log")
    a.set_ylabel(r"sSFR [Gyr$^{-1}$]")
    a.set_xlabel("age of universe [Gyr]")
    a.legend(loc="lower left")
    label_panel(a, "(k) specific SFR")

    # 12. M_BH / M_star -----------------------------------------------------
    a = ax[11]
    y = hist.get("mbh_over_mstar")
    if y is not None:
        a.plot(t, np.where(y > 0, y, np.nan), color=C["bh"], marker="o", ms=2.5, label="simulation")
    a.axhline(obs.bh_mass_fraction_local(), color=PALETTE["vermillion"], ls="--", lw=1.0, label="local (KH13, bulges)")
    a.set_yscale("log")
    a.set_ylabel(r"$M_{\rm BH}/M_\star$")
    a.set_xlabel("age of universe [Gyr]")
    a.legend(loc="lower right")
    label_panel(a, "(l) BH-to-stellar mass")

    for a in ax:
        a.set_xlim(np.nanmin(t), np.nanmax(t))
    add_redshift_axis(ax[0], units)
    add_redshift_axis(ax[1], units)
    add_redshift_axis(ax[2], units)

    if title:
        fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout()
    return savefig(fig, outpath)


# ---------------------------------------------------------------------------
# Log-driven figures (full time resolution)
# ---------------------------------------------------------------------------


def collect_logs(output_dir, stdout_path=None, run_dir=None):
    """Read every ASCII log the run produced, tolerating missing ones.

    Returns a dict with keys 'blackholes', 'sfr', 'details', 'bipolar',
    'mergers', 'energy', 'injection' -- each an empty-array dict if the
    corresponding file is not there.
    """
    out = {}
    out["blackholes"] = bh_logs.read_blackholes_txt(output_dir)
    out["sfr"] = bh_logs.read_sfr_txt(output_dir)
    out["details"] = bh_logs.read_bh_details(output_dir)
    out["bipolar"] = bh_logs.read_bh_bipolar(output_dir)
    out["mergers"] = bh_logs.read_bh_mergers(output_dir)
    out["energy"] = bh_logs.read_energy_txt(output_dir)
    # A restart chain has one stdout per job; hand the parser all of them.
    if stdout_path:
        paths = [stdout_path] if isinstance(stdout_path, str) else list(stdout_path)
    else:
        paths = bh_logs.find_stdout(run_dir or os.path.dirname(output_dir.rstrip("/")))
    out["injection"] = bh_logs.parse_bipolar_stdout(paths)
    out["stdout_path"] = paths
    return out


def plot_bh_growth(logs, units, outpath, title=None):
    """Four-panel BH growth history straight from the logs (full cadence)."""
    b = logs["blackholes"]
    d = logs["details"]
    if b["time"].size == 0:
        print("  [evolution] blackholes.txt empty -- skipping BH growth figure")
        return None

    t = units.age_gyr(b["time"])
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)

    a = ax[0, 0]
    a.plot(t, b["mass_bh"] * units.mass_to_msun, color=C["bh"], label=r"$\sum M_{\rm BH}$ (subgrid)")
    a.plot(t, b["mass_real"] * units.mass_to_msun, color=PALETTE["grey"], ls="--", label=r"$\sum M_{\rm dyn}$ (particle)")
    a.set_yscale("log")
    a.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.legend(loc="lower right")
    label_panel(a, "(a) total BH mass")

    a = ax[0, 1]
    a.plot(t, b["mdot_msun_per_yr"], color=C["bh"], lw=0.8)
    a.set_yscale("log")
    a.set_ylabel(r"$\sum\dot M_{\rm BH}$ [M$_\odot$ yr$^{-1}$]")
    label_panel(a, "(b) total accretion rate")

    a = ax[1, 0]
    a.plot(t, b["n_bh"], color=PALETTE["green"], drawstyle="steps-post")
    if logs["mergers"]["time"].size:
        for tm in units.age_gyr(logs["mergers"]["time"]):
            a.axvline(tm, color=PALETTE["vermillion"], lw=0.5, alpha=0.5)
        label_panel(a, "(c) BH count (red = merger)", loc="upper left")
    else:
        label_panel(a, "(c) BH count")
    a.set_ylabel(r"$N_{\rm BH}$")
    a.set_xlabel("age of universe [Gyr]")

    a = ax[1, 1]
    if d["time"].size:
        ids = bh_logs.unique_bh_ids(d)
        # Plot the ten most massive tracks; more than that is unreadable.
        peak = np.array([d["bh_mass"][d["id"] == i].max() for i in ids])
        for i in ids[np.argsort(peak)[::-1][:10]]:
            s = bh_logs.select_bh(d, i)
            a.plot(units.age_gyr(s["time"]), s["bh_mass"] * units.mass_to_msun, lw=0.9, alpha=0.85)
        a.set_yscale("log")
        a.set_ylabel(r"$M_{\rm BH}$ [M$_\odot$]")
        label_panel(a, "(d) individual BH tracks (10 most massive)")
    a.set_xlabel("age of universe [Gyr]")

    add_redshift_axis(ax[0, 0], units)
    add_redshift_axis(ax[0, 1], units)
    if title:
        fig.suptitle(title, y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_sfh(logs, hist, units, outpath, title=None):
    """Star formation history from sfr.txt, snapshots, and stellar ages."""
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.8))

    s = logs["sfr"]
    a = ax[0]
    if s["time"].size:
        t = units.age_gyr(s["time"])
        a.plot(t, s["sfr_msun_per_yr"], color=C["sfr"], lw=0.7, label="sfr.txt (whole box)")
        # A running median makes the trend visible under the burstiness.
        if t.size > 50:
            w = max(3, t.size // 200)
            k = np.ones(w) / w
            a.plot(t, np.convolve(s["sfr_msun_per_yr"], k, mode="same"), color=PALETTE["black"], lw=1.2, label="running mean")
    R = float(np.max(hist["_radii_kpc"]))
    a.plot(hist["age_gyr"], history.key(hist, "sfr", R), "o", ms=3, color=PALETTE["vermillion"], label=r"snapshots $<%g$ kpc" % R)
    a.set_yscale("log")
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
    a.legend(loc="lower right")
    label_panel(a, "(a) SFR")

    a = ax[1]
    if s["time"].size:
        a.plot(units.age_gyr(s["time"]), s["cum_mass_stars"] * units.mass_to_msun, color=C["stars"], label="cumulative formed (log)")
        a.plot(units.age_gyr(s["time"]), s["mass_stars"] * units.mass_to_msun, color=PALETTE["grey"], ls="--", label="on-the-fly total")
    a.plot(hist["age_gyr"], history.key(hist, "mstar", R), "o", ms=3, color=PALETTE["vermillion"], label=r"snapshots $<%g$ kpc" % R)
    a.set_yscale("log")
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel(r"$M_\star$ [M$_\odot$]")
    a.legend(loc="lower right")
    label_panel(a, "(b) stellar mass growth")

    a = ax[2]
    b = logs["blackholes"]
    if s["time"].size and b["time"].size:
        # Interpolate the BH accretion rate onto the SFR time axis so the
        # ratio is well defined even though the two logs are written on
        # different cadences.
        ts = units.age_gyr(s["time"])
        tb = units.age_gyr(b["time"])
        mdot = np.interp(ts, tb, b["mdot_msun_per_yr"], left=np.nan, right=np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s["sfr_msun_per_yr"] > 0, mdot / s["sfr_msun_per_yr"], np.nan)
        # sfr.txt has a row per active-timestep write (often millions), so a
        # finite ratio (SFR>0 *and* an mdot to interpolate) is almost always an
        # isolated sample with NaN neighbours on both sides.  A bare line plot
        # only draws a segment between two *consecutive* finite points, so it
        # rendered this panel essentially blank; markers show every sample.
        a.plot(ts, ratio, ".", ms=1.2, color=PALETTE["purple"], alpha=0.5)
        a.axhline(1e-3, color=PALETTE["grey"], ls="--", lw=1.0)
        a.text(0.98, 0.06, r"$\dot M_{\rm BH}/{\rm SFR}=10^{-3}$", transform=a.transAxes, ha="right", fontsize=7, color=PALETTE["grey"])
    a.set_yscale("log")
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel(r"$\dot M_{\rm BH}\,/\,$SFR")
    label_panel(a, "(c) BH growth vs star formation")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_stellar_age_distribution(snap, centre, outpath, radius_kpc=30.0, title=None):
    """Archaeological SFH: histogram of stellar formation times in an aperture.

    Only meaningful with GFM_StellarFormationTime present.  Wind particles
    (negative formation time in the TNG convention) are excluded.
    """
    from . import arepo_io as io

    d = snap.part(io.PT_STAR, ["Coordinates", "Masses", "GFM_StellarFormationTime", "GFM_InitialMass"])
    if d.get("count", 0) == 0 or "GFM_StellarFormationTime" not in d:
        print("  [evolution] no stellar formation times -- skipping age distribution")
        return None

    u = snap.units
    r = io.radius_from(d["Coordinates"], centre, snap.boxsize) * u.length_to_kpc(snap.a)
    aform = d["GFM_StellarFormationTime"]
    sel = (r <= radius_kpc) & (aform > 0)  # aform <= 0 marks wind particles
    if sel.sum() < 10:
        print("  [evolution] fewer than 10 star particles in aperture -- skipping")
        return None

    m = (d.get("GFM_InitialMass", d["Masses"])[sel]) * u.mass_to_msun
    tform = u.age_gyr(aform[sel])

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bins = np.linspace(tform.min(), max(tform.max(), snap.age_gyr), 60)
    dt = np.diff(bins) * 1.0e9
    h, _ = np.histogram(tform, bins=bins, weights=m)
    ax[0].step(0.5 * (bins[1:] + bins[:-1]), h / dt, where="mid", color=C["stars"])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("formation time [Gyr]")
    ax[0].set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
    label_panel(ax[0], "(a) archaeological SFH, $<%g$ kpc" % radius_kpc)

    o = np.argsort(tform)
    ax[1].plot(tform[o], np.cumsum(m[o]) / m.sum(), color=C["stars"])
    ax[1].set_xlabel("formation time [Gyr]")
    ax[1].set_ylabel(r"cumulative $M_\star$ fraction")
    for f in (0.5, 0.9):
        tf = np.interp(f, np.cumsum(m[o]) / m.sum(), tform[o])
        ax[1].axvline(tf, color=PALETTE["grey"], ls=":", lw=0.9)
        ax[1].text(tf, 0.05, r"$t_{%d}$" % int(f * 100), fontsize=7, color=PALETTE["grey"])
    label_panel(ax[1], "(b) cumulative", loc="lower right")

    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_energy_budget(logs, units, outpath, hist=None, params=None, title=None):
    """Global energy bookkeeping from energy.txt, plus the real AGN budget.

    Panel (a) is genuinely from ``energy.txt``.  Panels (b) and (c) are not:
    ``energy.txt``'s ``e_injected`` column is ``All.EgyInjection``, a mesh
    refinement/derefinement energy-conservation tracker that no black-hole
    routine ever writes to (see :func:`bh_logs.agn_cumulative_energy`) -- an
    earlier version of this function plotted it here as if it were the AGN
    energy budget, which is why these panels used to show noise near zero
    instead of a rising curve.  The correct source is the stdout injection
    counters (``BH_BIPOLAR_INJECTION*`` builds) or the snapshot's
    ``BH_CumEgyInjection_QM``/``_RM`` fields (``BH_THERMALFEEDBACK`` builds).
    """
    e = logs["energy"]
    if e["time"].size == 0:
        print("  [evolution] energy.txt empty -- skipping energy figure")
        return None

    t = units.age_gyr(e["time"])
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.8))

    a = ax[0]
    a.plot(t, np.abs(e["e_int"]) * units.energy_to_erg, label="internal")
    a.plot(t, np.abs(e["e_kin"]) * units.energy_to_erg, label="kinetic")
    a.plot(t, np.abs(e["e_pot"]) * units.energy_to_erg, label="|potential|")
    a.set_yscale("log")
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel("energy [erg]")
    a.legend()
    label_panel(a, "(a) global energies")

    t_agn, e_agn, src = bh_logs.agn_cumulative_energy(hist, logs.get("injection"), units)

    a = ax[1]
    if t_agn is not None:
        a.plot(t_agn, e_agn, color=PALETTE["vermillion"])
        a.set_yscale("log")
        a.text(0.03, 0.05, "source: %s" % src, transform=a.transAxes, fontsize=6.5, color=PALETTE["grey"])
    else:
        a.text(0.5, 0.5, "no AGN energy source available\n(no injection counters, no BH_CumEgyInjection_*)",
               ha="center", va="center", transform=a.transAxes, fontsize=7.5, color=PALETTE["grey"])
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel(r"$E_{\rm AGN}$ [erg]")
    label_panel(a, "(b) cumulative injected energy")

    a = ax[2]
    b = logs["blackholes"]
    if b["time"].size and t_agn is not None and t_agn.size > 1:
        # dE/dt of the injection vs eps_f eps_r Mdot c^2, the theoretical rate.
        from .units import CLIGHT, SEC_PER_YEAR, SOLAR_MASS

        eps_r = float(params.get("BlackHoleRadiativeEfficiency", 0.1)) if params else 0.1
        eps_f = float(params.get("BlackHoleFeedbackFactor", 0.05)) if params else 0.05

        dE = np.gradient(e_agn, np.maximum(t_agn * 1e9 * SEC_PER_YEAR, 1))
        a.plot(t_agn, np.abs(dE), color=PALETTE["vermillion"], lw=0.8, label=r"$dE_{\rm AGN}/dt$")
        mdot_cgs = np.interp(t_agn, units.age_gyr(b["time"]), b["mdot_msun_per_yr"]) * SOLAR_MASS / SEC_PER_YEAR
        a.plot(t_agn, eps_f * eps_r * mdot_cgs * CLIGHT ** 2, color=PALETTE["grey"], ls="--",
               label=r"$\epsilon_f\epsilon_r\,\dot Mc^2$ ($\epsilon_f$=%.2g, $\epsilon_r$=%.2g)" % (eps_f, eps_r))
        a.set_yscale("log")
        a.legend(loc="lower right", fontsize=7)
    a.set_xlabel("age of universe [Gyr]")
    a.set_ylabel(r"power [erg s$^{-1}$]")
    label_panel(a, "(c) injection power")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_catalog_evolution(chist, units, outpath, tree=None, title=None):
    """Growth of the target galaxy as *Subfind* measured it.

    Every panel here is a subhalo quantity: Subfind decided which particles
    belong to this galaxy, and for a THESAN-Zoom run that decision is the whole
    difference between a galaxy and the zoom system it sits in.  The host
    halo's M200c is the one Group quantity shown, and it is drawn only while
    the galaxy is its group's central -- for a satellite the host halo mass is
    not a property of the galaxy, and joining across the transition would draw
    a jump that never happened.

    ``tree`` overlays the main branch from ``output/tree.hdf5``.  Where the two
    disagree the tree is right: it followed the progenitor through mergers,
    while a per-snapshot selection can step onto a different object.
    """
    z = chist.get("z")
    if z is None or not np.isfinite(z).any():
        return None

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharex=True)
    ax = axes.ravel()

    TREE_C = PALETTE["vermillion"]

    def panel(k, key, ylabel, log=True, colour=C["stars"], scale=1.0):
        a = ax[k]
        y = chist.get(key)
        if y is None or not np.isfinite(y).any():
            a.text(0.5, 0.5, "no %s" % key, ha="center", va="center", transform=a.transAxes, color=PALETTE["grey"])
            a.set_ylabel(ylabel)
            return a
        good = np.isfinite(y) & ((y > 0) if log else np.isfinite(y))
        vals = y[good] * scale
        if log and not vals.size and np.isfinite(y).any():
            # A field the Subfind catalogue writes but never populates for
            # this run (every subhalo reports exactly 0) used to fall through
            # silently to an empty, unlabelled panel -- indistinguishable from
            # a real plotting bug.  Say so explicitly instead.
            a.text(0.5, 0.5, "Subfind reports %s = 0\nfor every subhalo in this catalogue" % key,
                   ha="center", va="center", transform=a.transAxes, color=PALETTE["grey"], fontsize=7.5)
            a.set_ylabel(ylabel)
            return a
        a.plot(z[good], vals, "-o", ms=3, color=colour, label="Subfind")
        # A log axis over a quantity that barely varies renders as noise around
        # a degenerate range.  Fall back to linear with a real margin so a flat
        # curve reads as flat rather than as scatter.
        if log and vals.size and np.nanmax(vals) > np.nanmin(vals) * 1.05:
            a.set_yscale("log")
        elif vals.size:
            v = float(np.nanmean(vals))
            pad = max(abs(v) * 0.1, 1e-30)
            a.set_ylim(v - pad, v + pad)
        a.set_ylabel(ylabel)
        return a

    a0 = panel(0, "mstar_msun", r"$M_\star$  [M$_\odot$]", colour=C["stars"])
    mh = chist.get("mhalo_msun")
    cen = chist.get("is_central")
    if mh is not None and np.isfinite(mh).any():
        m = np.isfinite(mh) & (mh > 0)
        if cen is not None:
            m &= cen > 0
        a0.plot(z[m], mh[m], "-", color=C["dm"], lw=1.2, label=r"host $M_{200c}$ (central only)")
    if tree is not None and tree.mstar_msun() is not None:
        a0.plot(tree.z, tree.mstar_msun(), "--", color=TREE_C, lw=1.2, label="tree main branch")
    a0.legend(fontsize=8, frameon=False)
    label_panel(a0, "stellar and halo mass")

    a = panel(1, "sfr_msun_per_yr", r"SFR  [M$_\odot$ yr$^{-1}$]", colour=C["sfr"])
    if tree is not None and tree.sfr_msun_per_yr() is not None:
        a.plot(tree.z, tree.sfr_msun_per_yr(), "--", color=TREE_C, lw=1.2)
    label_panel(a, "star formation")

    a = panel(2, "bh_mass_msun", r"$M_{\rm BH}$  [M$_\odot$]", colour=C["bh"])
    if tree is not None and tree.mbh_msun() is not None:
        a.plot(tree.z, tree.mbh_msun(), "--", color=TREE_C, lw=1.2)
    label_panel(a, "black hole mass")

    label_panel(panel(3, "bh_mdot_msun_per_yr", r"$\dot M_{\rm BH}$  [M$_\odot$ yr$^{-1}$]", colour=C["bh"]),
                "accretion rate")
    label_panel(panel(4, "ssfr_per_gyr", r"sSFR  [Gyr$^{-1}$]", colour=C["sfr"]), "specific SFR")
    label_panel(panel(5, "fgas", r"$f_{\rm gas}$", log=False, colour=C["gas"]), "gas fraction")
    label_panel(panel(6, "rhalf_star_kpc", r"$R_{1/2,\star}$  [kpc]", colour=C["stars"]), "stellar half-mass radius")
    label_panel(panel(7, "sigma_kms", r"$\sigma$  [km s$^{-1}$]", colour=PALETTE["vermillion"]), "velocity dispersion")

    for a in ax[4:]:
        a.set_xlabel("redshift")
    # The axes are shared, so one inversion flips all eight.  Inverting each in
    # turn would flip an even number of times and leave time running backwards.
    ax[0].invert_xaxis()

    n = chist.get("n_subhalos")
    if n is not None and np.isfinite(n).any():
        lo, hi = int(np.nanmin(n)), int(np.nanmax(n))
        ax[0].text(0.02, 0.02,
                   "%d subhaloes in the catalogue" % lo if lo == hi
                   else "%d-%d subhaloes in the catalogue" % (lo, hi),
                   transform=ax[0].transAxes, fontsize=7, color=PALETTE["grey"])

    fig.suptitle(("%s -- " % title if title else "") + "galaxy growth from the Subfind catalogue", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return savefig(fig, outpath)
