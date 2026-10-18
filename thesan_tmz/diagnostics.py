"""
Model-specific health checks for the Arepo2-TMZ-newbh black hole modules.

These are the plots that do not exist in a generic AREPO analysis suite,
because they test things only this branch does:

  * BH_BIPOLAR_INJECTION -- did the energy actually go anywhere?  How often did
    the cone come up empty, or the axis fail to form?
  * BH_BIPOLAR_INJECTION_MODE_SWITCH -- does the thermal/kinetic split match
    what the Eddington threshold in the parameter file implies?
  * BH_ALPHA_DISK_ACCRETION -- where does the run sit in the (Omega, Q) plane
    that decides whether the disc or the Bondi rate limits?
  * BH_TORQUE_LIMITED_ACCRETION -- what are f_disc and f_gas actually doing,
    given that mdot goes as f_disc^{5/2}?
  * the collimation itself -- is there a real polar/equatorial asymmetry in the
    gas around the black hole, or is the "bipolar" feedback isotropic in
    practice?

The last one is the only test that looks at the *outcome* rather than the
bookkeeping, and it is the one worth trusting.

Cross-references to the code:
    blackhole_bipolar_injection.cc:736/741  the stdout counters
    blackhole_mdot.cc:277-292               the Eddington mode switch
    blackhole_mdot.cc:504-582               alpha-disc rate (Omega, Q)
    blackhole_mdot.cc:597-695               torque-limited rate (f_disc, f_gas)
"""

from __future__ import annotations

import os

import numpy as np

from . import arepo_io as io
from . import bh_logs
from .plotstyle import COMPONENT_COLORS as C
from .plotstyle import PALETTE, LogNorm, label_panel, plt, savefig, shade_band

# ---------------------------------------------------------------------------
# Parameters used by the model, read back out of the snapshot
# ---------------------------------------------------------------------------

BIPOLAR_PARAMS = [
    "BHBipolarTheta",
    "BHBipolarEfficiency",
    "BHBipolarColdFraction",
    "BHBipolarColdTemp",
    "BHBipolarInjColdTemp",
    "BHBipolarInjColdFraction",
    "BHBipolarInjEddThreshold",
    "BHBipolarInjThermalTheta",
    "BHBipolarInjThermalEfficiency",
    "BHBipolarInjKineticTheta",
    "BHBipolarInjKineticEfficiency",
    "BHBipolarInjMaxVel",
    "BHBipolarInjBurstFactor",
    "BlackHoleRadiativeEfficiency",
    "BlackHoleFeedbackFactor",
    "BlackHoleAccretionFactor",
    "BlackHoleEddingtonFactor",
]


def model_summary(snap):
    """What was this run actually configured to do?

    Pulls the relevant ``/Config`` options and ``/Parameters`` values out of a
    snapshot so every diagnostic figure can be annotated with the settings it
    is testing -- which removes the commonest confusion when comparing runs.
    """
    cfg, par = snap.config, snap.parameters
    accretion = "bondi"
    for opt, name in (
        ("BH_ALPHA_DISK_ACCRETION", "alpha-disc"),
        ("BH_TORQUE_LIMITED_ACCRETION", "torque-limited"),
        ("BH_FREEFALL_ACCRETION", "freefall"),
        ("BH_BONDI_DISK_VORTICITY", "bondi+vorticity"),
    ):
        if opt in cfg:
            accretion = name

    feedback = []
    for opt in (
        "BH_THERMALFEEDBACK",
        "BH_THERMALFEEDBACK_ACC",
        "BH_BIPOLAR_FEEDBACK",
        "BH_BIPOLAR_INJECTION",
        "BH_BIPOLAR_INJECTION_THERMAL",
        "BH_BIPOLAR_INJECTION_KINETIC",
        "BH_BIPOLAR_INJECTION_MODE_SWITCH",
        "BH_NEW_CENTERING",
        "LOCAL_INJECTION",
        "LOCAL_INJECTION_BLACK_HOLES",
        "LOCAL_INJECTION_WAKEUP_CELLS",
        "VORONOI_LOCAL_OCTANTS",
    ):
        if opt in cfg:
            feedback.append(opt)

    # -- how each kind of feedback is actually deposited ---------------------
    #
    # LOCAL_INJECTION builds a Voronoi mesh around the source and deposits over
    # its faces; without it the deposition is kernel-weighted over the SPH-style
    # neighbour list.  The two flags are independent: a run can use the local
    # mesh for stars and the kernel for black holes.  Which one is on decides
    # which diagnostics in this package mean anything.
    stellar_dep = "voronoi-local" if "LOCAL_INJECTION" in cfg else "kernel"
    bh_dep = "voronoi-local" if "LOCAL_INJECTION_BLACK_HOLES" in cfg else "kernel"

    channels = [n for opt, n in (
        ("BH_BIPOLAR_INJECTION_THERMAL", "thermal"),
        ("BH_BIPOLAR_INJECTION_KINETIC", "kinetic"),
    ) if opt in cfg]
    if not channels and "BH_BIPOLAR_INJECTION" in cfg:
        channels = ["bipolar"]
    collimated = bool(channels) or "BH_BIPOLAR_FEEDBACK" in cfg

    # Flags that make the "mdot" column of blackhole_details a different
    # number from BPP(n).BH_Mdot -- the value that actually grows the BH, is
    # summed into blackholes.txt, and decides the BH_BIPOLAR_INJECTION_MODE_SWITCH
    # split.  blackhole_bondi_rate()/blackhole_alpha_disk_rate()/
    # blackhole_torque_limited_rate() fprintf their *own* local "mdot" before
    # returning it to blackhole_calculate_mdot(), which then still applies the
    # Eddington cap (blackhole_mdot.cc:189-190) and, if BH_EXACT_INTEGRATION is
    # on, throws that value away entirely and substitutes the analytic
    # sub-step-averaged accretion rate (blackhole_mdot.cc:162-217) -- so the
    # logged "mdot" is the raw pre-cap, pre-integration estimate, not what
    # happened.  See plot_accretion_internals / check_mode_switch.
    mdot_caveats = [f for f in ("BH_EXACT_INTEGRATION", "BH_ADIOS_WIND", "BH_PRESSURE_CRITERION") if f in cfg]

    params = {k: par[k] for k in BIPOLAR_PARAMS if k in par}
    return {
        "accretion": accretion,
        "feedback": feedback,
        "params": params,
        "config": cfg,
        "stellar_deposition": stellar_dep,
        "bh_deposition": bh_dep,
        "bipolar_channels": channels,
        "collimated": collimated,
        "mode_switch": "BH_BIPOLAR_INJECTION_MODE_SWITCH" in cfg,
        "mdot_caveats": mdot_caveats,
        "scenario": _scenario_name(stellar_dep, bh_dep, collimated),
    }


def _scenario_name(stellar_dep, bh_dep, collimated):
    """One phrase naming the deposition combination, for figure captions."""
    s = "Voronoi-local" if stellar_dep == "voronoi-local" else "kernel"
    b = "Voronoi-local" if bh_dep == "voronoi-local" else "kernel"
    tail = ", collimated" if collimated else ", isotropic"
    if s == b:
        return "%s deposition for stars and black holes%s" % (s, tail)
    return "%s deposition for stars, %s for black holes%s" % (s, b, tail)


def format_model_summary(summary):
    lines = [
        "configuration: %s" % summary.get("scenario", "?"),
        "  accretion            : %s" % summary["accretion"],
        "  stellar feedback     : %s deposition" % summary.get("stellar_deposition", "?"),
        "  black hole feedback  : %s deposition" % summary.get("bh_deposition", "?"),
        "  collimation channels : %s" % (", ".join(summary.get("bipolar_channels", [])) or "none (isotropic)"),
        "feedback options: %s" % (", ".join(summary["feedback"]) or "(none)"),
    ]
    for k, v in sorted(summary["params"].items()):
        lines.append("  %-32s %s" % (k, v))
    caveats = summary.get("mdot_caveats") or []
    if caveats:
        lines.append("mdot caveat: %s active -- blackhole_details' mdot column is the raw" % "/".join(caveats))
        lines.append("  pre-cap estimate, not what the BH actually accreted; figs 08/mode-switch")
        lines.append("  check clip it at the Eddington rate for display, but a build with")
        lines.append("  BH_EXACT_INTEGRATION time-averages within the step in a way no ASCII log")
        lines.append("  records -- trust blackholes.txt / fig 09 for the true accretion rate.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Did the injection happen?
# ---------------------------------------------------------------------------


def plot_injection_counters(inj, units, outpath, params=None, title=None):
    """The BH_BIPOLAR_INJECTION per-step counters scraped from stdout.

    Silence is the failure mode this catches: a black hole that holds energy
    in its reservoir but never finds a target injects nothing, and in every
    other output that is indistinguishable from a quiet black hole.  If the
    'empty cone' counter is a large fraction of the steps, widen the opening
    angle or raise the neighbour number.
    """
    if inj is None or inj.get("time", np.zeros(0)).size == 0:
        print("  [diagnostics] no BH_BIPOLAR_INJECTION lines found in stdout -- skipping")
        return None

    t = units.age_gyr(inj["time"])
    n = t.size
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.0))

    # (a) energy injected per step
    a = ax[0, 0]
    a.plot(t, np.where(inj["thermal"] > 0, inj["thermal"], np.nan), ".", ms=2.5, color=C["thermal"], label="thermal")
    a.plot(t, np.where(inj["kinetic"] > 0, inj["kinetic"], np.nan), ".", ms=2.5, color=C["kinetic"], label="kinetic")
    a.set_yscale("log")
    a.set_ylabel("energy per step [code units]")
    a.set_xlabel("age of universe [Gyr]")
    a.legend()
    label_panel(a, "(a) injected energy per step")

    # (b) cumulative
    a = ax[0, 1]
    a.plot(t, np.cumsum(np.nan_to_num(inj["thermal"])), color=C["thermal"], label="thermal")
    a.plot(t, np.cumsum(np.nan_to_num(inj["kinetic"])), color=C["kinetic"], label="kinetic")
    tot = np.cumsum(np.nan_to_num(inj["thermal"]) + np.nan_to_num(inj["kinetic"]))
    a.plot(t, tot, color=PALETTE["black"], ls="--", label="total")
    a.set_yscale("log")
    a.set_ylabel("cumulative energy [code units]")
    a.set_xlabel("age of universe [Gyr]")
    a.legend()
    label_panel(a, "(b) cumulative injection")

    # (c) mode split
    a = ax[0, 2]
    if np.isfinite(inj.get("n_thermal", np.array([np.nan]))).any():
        # Rows are per-BH-step at wildly uneven cadence (bursts of thousands
        # of steps within a few Myr) -- plotted raw, those bursts alias into
        # an opaque block that hides the title, the legend, and the fraction
        # trace underneath it.  Bin onto a coarse, even time grid first.
        nbins = int(np.clip(n // 50, 20, 150))
        edges = np.linspace(t.min(), t.max(), nbins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ib = np.clip(np.digitize(t, edges) - 1, 0, centres.size - 1)
        nth = np.bincount(ib, weights=np.nan_to_num(inj["n_thermal"]), minlength=centres.size)
        nki = np.bincount(ib, weights=np.nan_to_num(inj["n_kinetic"]), minlength=centres.size)
        a.plot(centres, nth, color=C["thermal"], lw=1.2, label="thermal mode (binned mean)")
        a.plot(centres, nki, color=C["kinetic"], lw=1.2, label="kinetic mode (binned mean)")
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(nth + nki > 0, nth / (nth + nki), np.nan)
        a2 = a.twinx()
        a2.plot(centres, frac, color=PALETTE["grey"], lw=1.4)
        a2.set_ylabel("thermal fraction (binned)", color=PALETTE["grey"])
        a2.set_ylim(0, 1)
        a.legend(loc="upper right", fontsize=6.5)
        a.set_title("(c) mode switch (Eddington)", fontsize=9, loc="left")
    else:
        a.text(0.5, 0.5, "no MODE_SWITCH\n(single-mode build)", ha="center", va="center", transform=a.transAxes, color=PALETTE["grey"])
        a.set_title("(c) mode switch", fontsize=9, loc="left")
    a.set_ylabel(r"$N_{\rm BH}$ in each mode (binned mean)")
    a.set_xlabel("age of universe [Gyr]")

    # (d) failure counters as a running fraction of steps
    a = ax[1, 0]
    for k, lab, col in (
        ("empty_cone", "empty cone", PALETTE["vermillion"]),
        ("no_axis", "no axis", PALETTE["purple"]),
        ("cold_fallback", "cold-disk fallback", PALETTE["orange"]),
    ):
        y = np.nan_to_num(inj[k])
        a.plot(t, np.cumsum(y) / np.arange(1, n + 1), color=col, label="%s (cum. mean)" % lab)
    a.set_ylabel("counts per step (running mean)")
    a.set_xlabel("age of universe [Gyr]")
    a.legend(fontsize=7)
    label_panel(a, "(d) failure counters")

    # (e) histogram of per-step counters
    a = ax[1, 1]
    labels, values = [], []
    for k, lab in (("empty_cone", "empty cone"), ("no_axis", "no axis"), ("cold_fallback", "cold fallback"), ("clipped", "clipped cells")):
        labels.append(lab)
        values.append(float(np.nansum(inj[k])))
    steps_with_injection = int(np.sum((np.nan_to_num(inj["thermal"]) + np.nan_to_num(inj["kinetic"])) > 0))
    labels.append("steps w/ injection")
    values.append(float(steps_with_injection))
    labels.append("total steps")
    values.append(float(n))
    a.barh(range(len(values)), values, color=[PALETTE["vermillion"], PALETTE["purple"], PALETTE["orange"], PALETTE["skyblue"], PALETTE["green"], PALETTE["grey"]])
    a.set_yticks(range(len(values)))
    a.set_yticklabels(labels, fontsize=8)
    a.set_xscale("symlog")
    a.set_xlabel("total count over the run")
    label_panel(a, "(e) totals", loc="lower right")

    # (f) cost
    a = ax[1, 2]
    a.plot(t, inj["seconds"], ".", ms=2, color=PALETTE["grey"])
    a.set_yscale("log")
    a.set_ylabel("wall time per injection step [s]")
    a.set_xlabel("age of universe [Gyr]")
    label_panel(a, "(f) cost")

    if title:
        fig.suptitle(title, y=0.99)

    # The configured parameters go in a reserved strip below the axes, so they
    # annotate the figure without ever sitting on top of a panel.
    txt = ""
    if params:
        items = [(k.replace("BHBipolarInj", ""), v) for k, v in sorted(params.items()) if "BipolarInj" in k]
        txt = "   ".join("%s=%s" % (k, v) for k, v in items)
    if txt:
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.text(0.5, 0.012, txt, fontsize=7, ha="center", va="bottom", color=PALETTE["grey"], family="monospace")
    else:
        fig.tight_layout()
    return savefig(fig, outpath)


def injection_report(inj, logs=None, units=None):
    """One-screen text summary of the injection health.  Returns a string."""
    if inj is None or inj.get("time", np.zeros(0)).size == 0:
        return "BH_BIPOLAR_INJECTION: no stdout diagnostics found."

    n = inj["time"].size
    th, kin = np.nan_to_num(inj["thermal"]), np.nan_to_num(inj["kinetic"])
    active = (th + kin) > 0
    lines = [
        "BH_BIPOLAR_INJECTION summary",
        "  injection steps parsed        : %d" % n,
        "  steps with non-zero injection : %d  (%.1f%%)" % (active.sum(), 100.0 * active.mean()),
        "  cumulative thermal energy     : %.6g (code units)" % th.sum(),
        "  cumulative kinetic energy     : %.6g (code units)" % kin.sum(),
    ]
    if th.sum() + kin.sum() > 0:
        lines.append("  thermal / total               : %.3f" % (th.sum() / (th.sum() + kin.sum())))
    for k, lab in (("no_axis", "no-axis events"), ("cold_fallback", "cold-disk fallbacks"), ("empty_cone", "empty-cone events"), ("clipped", "clipped cells")):
        v = np.nan_to_num(inj[k])
        lines.append("  %-29s: %d total, %.3f per step" % (lab, v.sum(), v.mean()))
    if np.isfinite(inj.get("n_thermal", np.array([np.nan]))).any():
        nt, nk = np.nansum(inj["n_thermal"]), np.nansum(inj["n_kinetic"])
        tot = nt + nk
        lines.append("  mode switch (BH-steps)        : thermal %d (%.1f%%) / kinetic %d (%.1f%%)" % (nt, 100 * nt / max(tot, 1), nk, 100 * nk / max(tot, 1)))

    empty = np.nan_to_num(inj["empty_cone"]).sum()
    if empty > 0.2 * max(active.sum(), 1):
        lines.append("  WARNING: the cone was empty on a large fraction of steps.")
        lines.append("           Widen BHBipolarInj*Theta or raise DesNumNgbBlackHole.")
    if np.nan_to_num(inj["no_axis"]).sum() > 0.1 * n:
        lines.append("  WARNING: the axis failed to form often -- the neighbour gas has")
        lines.append("           little net angular momentum.  Check the cold-disk cut.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Is the axis stable?
# ---------------------------------------------------------------------------


def plot_axis_stability(bipolar, units, outpath, bh_id=None, title=None):
    """How much does the jet axis wander between timesteps?

    A jet axis that reorients by tens of degrees between steps does not drill a
    channel -- it sprays, and the "bipolar" feedback becomes an expensive way of
    doing isotropic heating.  The distribution of step-to-step reorientation
    angles is the cleanest way to see this.
    """
    if bipolar["time"].size == 0:
        print("  [diagnostics] no blackhole_bipolar log -- skipping axis stability")
        return None

    ids = bh_logs.unique_bh_ids(bipolar)
    if bh_id is None:
        # The BH with the most log rows is the one worth looking at.
        counts = [np.sum(bipolar["id"] == i) for i in ids]
        bh_id = int(ids[int(np.argmax(counts))])
    s = bh_logs.select_bh(bipolar, bh_id)

    n = s["jhat"]
    ok = np.linalg.norm(n, axis=1) > 0
    t = units.age_gyr(s["time"])

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.0))

    # (a) components of the unit axis
    a = ax[0, 0]
    for k, lab in enumerate(("x", "y", "z")):
        a.plot(t[ok], n[ok, k], lw=0.8, label=r"$\hat n_%s$" % lab)
    a.set_ylim(-1.05, 1.05)
    a.set_ylabel("axis direction")
    a.set_xlabel("age of universe [Gyr]")
    a.legend(ncol=3, fontsize=7)
    label_panel(a, "(a) axis components, BH %d" % bh_id)

    # (b) step-to-step reorientation
    a = ax[0, 1]
    nn = n[ok]
    if nn.shape[0] > 2:
        # The axis is a *direction*, not an orientation: n and -n describe the
        # same cone pair, so fold the angle into [0, 90] deg.
        cosang = np.clip(np.abs(np.sum(nn[1:] * nn[:-1], axis=1)), -1.0, 1.0)
        dang = np.degrees(np.arccos(cosang))
        a.plot(t[ok][1:], dang, ".", ms=2, color=PALETTE["purple"])
        a.axhline(np.median(dang), color=PALETTE["black"], ls="--", lw=1.0, label="median = %.1f deg" % np.median(dang))
        a.legend(fontsize=7)
    a.set_ylabel("reorientation per step [deg]")
    a.set_xlabel("age of universe [Gyr]")
    label_panel(a, "(b) axis wander")

    # (c) angle from the time-averaged axis
    a = ax[1, 0]
    if nn.shape[0] > 2:
        mean_axis = nn.mean(axis=0)
        mean_axis /= np.linalg.norm(mean_axis)
        ang = np.degrees(np.arccos(np.clip(np.abs(nn @ mean_axis), -1, 1)))
        a.plot(t[ok], ang, lw=0.8, color=PALETTE["blue"])
        a.axhline(30, color=PALETTE["vermillion"], ls=":", lw=1.0)
        a.text(0.98, 0.9, "30 deg", transform=a.transAxes, ha="right", fontsize=7, color=PALETTE["vermillion"])
    a.set_ylabel("angle from mean axis [deg]")
    a.set_xlabel("age of universe [Gyr]")
    label_panel(a, "(c) drift from the mean direction")

    # (d) cold disk fraction against the threshold
    a = ax[1, 1]
    a.plot(t, s["cold_fraction"], lw=0.8, color=C["gas_cold"], label="cold fraction")
    a.plot(t, s["cold_flag"], lw=0.8, color=PALETTE["vermillion"], alpha=0.6, label="above threshold")
    a.set_ylim(-0.05, 1.15)
    a.set_ylabel("cold gas fraction in the kernel")
    a.set_xlabel("age of universe [Gyr]")
    a.legend(fontsize=7, loc="lower right")
    label_panel(a, "(d) cold-disk criterion")

    if title:
        fig.suptitle(title, y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)


# ---------------------------------------------------------------------------
# 3. Accretion-model internals
# ---------------------------------------------------------------------------


def plot_accretion_internals(details, units, outpath, params=None, summary=None, title=None):
    """Model-dependent panels from ``blackhole_details``.

    The layout is detected from the column count, so this works unchanged for
    the Bondi, alpha-disc and torque-limited variants.

    ``blackhole_details``'s "mdot" column is logged *inside*
    ``blackhole_bondi_rate()`` (or the alpha-disc/torque equivalent), before
    the caller applies the Eddington cap (``blackhole_mdot.cc:189-190``) and,
    if ``BH_EXACT_INTEGRATION`` is compiled in, before that cap value is
    itself discarded in favour of an analytically sub-step-averaged rate
    (``blackhole_mdot.cc:162-217``).  So the raw column can run many orders of
    magnitude above the Eddington rate even though the black hole never
    actually accreted anywhere near that fast -- every panel here therefore
    also shows ``mdot`` clipped at ``BlackHoleEddingtonFactor * mdot_Edd``,
    which is what the simulation actually enforced, and uses the clipped
    value for lambda_Edd (panels b, e) so the mode-switch fraction quoted in
    panel (b) is not inflated by the un-clipped tail.  This clip still cannot
    reproduce BH_EXACT_INTEGRATION's sub-step averaging exactly -- see the
    caveat printed on the figure when that flag is set, and
    :func:`check_mode_switch`.
    """
    if details["time"].size == 0:
        print("  [diagnostics] blackhole_details empty -- skipping accretion internals")
        return None

    layout = details.get("_layout", "default")
    t = units.age_gyr(details["time"])
    mdot_raw = details["mdot"] * units.mdot_to_msun_per_yr
    mbh = details["bh_mass"] * units.mass_to_msun
    eps_r = params.get("BlackHoleRadiativeEfficiency", 0.1) if params else 0.1
    edd_factor = float(params.get("BlackHoleEddingtonFactor", 1.0)) if params else 1.0
    edd = units.eddington_rate_msun_per_yr(np.maximum(mbh, 1e-30), eps_r)
    with np.errstate(invalid="ignore"):
        mdot = np.where(edd > 0, np.minimum(mdot_raw, edd_factor * edd), mdot_raw)  # Eddington-capped, as the code enforces
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(edd > 0, mdot / edd, np.nan)
    caveats = (summary or {}).get("mdot_caveats") or []

    ncol = 3 if layout == "default" else 4
    fig, ax = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7.0))
    ax = np.atleast_2d(ax)

    # (a) mdot vs time -- raw estimate faint in the background, the
    # Eddington-capped value (what the code actually used) on top.
    a = ax[0, 0]
    a.plot(t, np.where(mdot_raw > 0, mdot_raw, np.nan), ".", ms=1.0, color=PALETTE["grey"], alpha=0.15, label="raw (pre-cap) estimate")
    a.plot(t, np.where(mdot > 0, mdot, np.nan), ".", ms=1.5, color=C["bh"], alpha=0.5, label="Eddington-capped")
    a.legend(fontsize=6, markerscale=4, loc="upper right")
    if caveats:
        a.text(0.02, 0.02, "caveat: %s active -- true rate is further time-averaged" % "/".join(caveats),
               transform=a.transAxes, fontsize=5.5, color=PALETTE["grey"])
    a.set_yscale("log")
    a.set_ylabel(r"$\dot M$ [M$_\odot$ yr$^{-1}$]")
    a.set_xlabel("age [Gyr]")
    label_panel(a, "(a) accretion rate")

    # (b) Eddington ratio distribution
    a = ax[0, 1]
    good = np.isfinite(lam) & (lam > 0)
    if good.any():
        a.hist(np.log10(lam[good]), bins=50, color=C["bh"], alpha=0.8)
        a.axvline(0.0, color=PALETTE["vermillion"], ls="--", lw=1.0)
        if params and "BHBipolarInjEddThreshold" in params:
            thr = float(params["BHBipolarInjEddThreshold"])
            if thr > 0:
                a.axvline(np.log10(thr), color=PALETTE["blue"], ls="-", lw=1.4, label=r"mode switch $\lambda=%.3g$" % thr)
                a.legend(fontsize=7)
                above = float(np.mean(lam[good] >= thr))
                a.text(0.03, 0.78, "%.1f%% of rows above\n=> thermal mode" % (100 * above), transform=a.transAxes, fontsize=7)
    a.set_xlabel(r"$\log_{10}\lambda_{\rm Edd}$")
    a.set_ylabel("rows")
    label_panel(a, "(b) Eddington ratio")

    # (c) the BH's environment.  The torque-limited log replaces the sound
    # speed with f_disk/f_gas, so fall back to rho against mdot there.
    a = ax[0, 2]
    if "csnd" in details:
        good = (details["rho"] > 0) & (details["csnd"] > 0)
        if good.any():
            h = a.hexbin(
                np.log10(details["rho"][good] * units.density_to_nH_cgs(1.0)),
                np.log10(details["csnd"][good] * units.velocity_to_kms),
                bins="log",
                gridsize=45,
                cmap="magma",
            )
            fig.colorbar(h, ax=a, label=r"$\log N$")
        a.set_ylabel(r"$\log_{10} c_s$ [km s$^{-1}$]")
        label_panel(a, "(c) BH environment")
    else:
        good = (details["rho"] > 0) & (mdot > 0)
        if good.any():
            h = a.hexbin(
                np.log10(details["rho"][good] * units.density_to_nH_cgs(1.0)), np.log10(mdot[good]), bins="log", gridsize=45, cmap="magma"
            )
            fig.colorbar(h, ax=a, label=r"$\log N$")
        a.set_ylabel(r"$\log_{10}\dot M$ [M$_\odot$ yr$^{-1}$]")
        label_panel(a, "(c) BH environment (no $c_s$ in this log)")
    a.set_xlabel(r"$\log_{10} n_{\rm H}$ at the BH [cm$^{-3}$]")

    # (d) mdot vs M_BH, with the Eddington locus
    a = ax[1, 0]
    good = (mbh > 0) & (mdot > 0)
    if good.any():
        a.plot(mbh[good], mdot[good], ".", ms=1.5, alpha=0.4, color=C["bh"])
        mg = np.logspace(np.log10(mbh[good].min()), np.log10(mbh[good].max()), 50)
        a.plot(mg, units.eddington_rate_msun_per_yr(mg), color=PALETTE["vermillion"], lw=1.3, label="Eddington")
        a.plot(mg, 0.01 * units.eddington_rate_msun_per_yr(mg), color=PALETTE["grey"], ls=":", lw=1.0, label=r"$0.01\,\dot M_{\rm Edd}$")
        a.legend(fontsize=7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"$M_{\rm BH}$ [M$_\odot$]")
    a.set_ylabel(r"$\dot M$ [M$_\odot$ yr$^{-1}$]")
    label_panel(a, "(d) accretion vs mass")

    # (e) duty cycle
    a = ax[1, 1]
    if good.any():
        thresholds = np.logspace(-6, 0.5, 40)
        duty = [np.mean(lam[np.isfinite(lam)] >= th) for th in thresholds]
        a.plot(thresholds, duty, color=C["bh"])
        a.set_xscale("log")
        a.set_ylabel(r"fraction of rows with $\lambda>x$")
        a.set_xlabel(r"$\lambda_{\rm Edd}$ threshold")
    label_panel(a, "(e) duty cycle")

    # (f) sound speed / density time series -- the Bondi ingredients
    a = ax[1, 2]
    a.plot(t, details["rho"] * units.density_to_nH_cgs(1.0), ".", ms=1.2, color=C["gas"], alpha=0.4)
    a.set_yscale("log")
    a.set_ylabel(r"$n_{\rm H}$ at the BH [cm$^{-3}$]")
    a.set_xlabel("age [Gyr]")
    label_panel(a, "(f) BH gas density")

    # Model-specific extra column ------------------------------------------
    if layout == "alpha":
        a = ax[0, 3]
        good = np.isfinite(details["toomre_q"]) & (details["toomre_q"] > 0)
        if good.any():
            a.hist(np.log10(details["toomre_q"][good]), bins=50, color=PALETTE["green"], alpha=0.85)
            a.axvline(0.0, color=PALETTE["vermillion"], ls="--", lw=1.2, label="Q = 1 (marginal stability)")
            a.legend(fontsize=7)
        a.set_xlabel(r"$\log_{10} Q$")
        a.set_ylabel("rows")
        label_panel(a, "(g) Toomre Q")

        a = ax[1, 3]
        good = (details["omega"] > 0) & (details["toomre_q"] > 0)
        if good.any():
            h = a.hexbin(np.log10(details["omega"][good]), np.log10(details["toomre_q"][good]), bins="log", gridsize=45, cmap="viridis")
            fig.colorbar(h, ax=a, label=r"$\log N$")
            a.axhline(0.0, color=PALETTE["vermillion"], ls="--", lw=1.0)
        a.set_xlabel(r"$\log_{10}\Omega$ [code]")
        a.set_ylabel(r"$\log_{10} Q$")
        label_panel(a, r"(h) $\Omega$-$Q$ plane")

    elif layout == "torque":
        a = ax[0, 3]
        for k, lab, col in (("f_disk", r"$f_{\rm disc}$", PALETTE["blue"]), ("f_gas", r"$f_{\rm gas}$", PALETTE["green"])):
            good = np.isfinite(details[k]) & (details[k] > 0)
            if good.any():
                a.hist(details[k][good], bins=50, range=(0, 1), histtype="step", lw=1.4, color=col, label=lab)
        a.legend(fontsize=8)
        a.set_xlabel("fraction")
        a.set_ylabel("rows")
        label_panel(a, "(g) disc and gas fractions")

        a = ax[1, 3]
        good = (details["f_disk"] > 0) & (mdot > 0)
        if good.any():
            h = a.hexbin(np.log10(details["f_disk"][good]), np.log10(mdot[good]), bins="log", gridsize=45, cmap="viridis")
            fig.colorbar(h, ax=a, label=r"$\log N$")
            # mdot goes as f_disc^{5/2} at fixed everything else -- overplot
            # that slope so a departure from it is visible by eye.
            x = np.linspace(np.log10(details["f_disk"][good]).min(), 0.0, 10)
            y0 = np.median(np.log10(mdot[good]))
            a.plot(x, y0 + 2.5 * (x - np.median(np.log10(details["f_disk"][good]))), color=PALETTE["vermillion"], lw=1.3, label=r"slope $5/2$")
            a.legend(fontsize=7, loc="lower right")
        a.set_xlabel(r"$\log_{10} f_{\rm disc}$")
        a.set_ylabel(r"$\log_{10}\dot M$ [M$_\odot$ yr$^{-1}$]")
        label_panel(a, r"(h) $\dot M \propto f_{\rm disc}^{5/2}$")

    if title:
        fig.suptitle("%s  [%s accretion]" % (title, layout), y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)


# ---------------------------------------------------------------------------
# 4. Is the feedback actually collimated?
# ---------------------------------------------------------------------------


def bh_axis_from_snapshot(snap, centre, radius_kpc=None, cold_only=True, T_cold=1.0e4):
    """Recompute the jet axis the way the code does: net J of the kernel gas.

    ``radius_kpc`` defaults to the BH's own smoothing length, which is what the
    module actually uses.  ``cold_only`` mirrors the cold-disk selection.
    Returns the unit axis, or None if the gas has no net angular momentum.
    """
    u = snap.units
    L = u.length_to_kpc(snap.a)

    if radius_kpc is None:
        bh = snap.part(io.PT_BH, ["Coordinates", "BH_Hsml"])
        if bh.get("count", 0) and "BH_Hsml" in bh:
            r = io.radius_from(bh["Coordinates"], centre, snap.boxsize)
            radius_kpc = float(bh["BH_Hsml"][int(np.argmin(r))] * L)
        else:
            radius_kpc = 1.0

    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "InternalEnergy", "ElectronAbundance"])
    if gas.get("count", 0) == 0:
        return None, radius_kpc

    rel = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
    r = np.linalg.norm(rel, axis=1)
    sel = r <= radius_kpc
    if cold_only and "InternalEnergy" in gas:
        T = u.temperature(gas["InternalEnergy"], gas.get("ElectronAbundance"))
        sel &= T < T_cold
    if sel.sum() < 8:
        return None, radius_kpc

    vc = np.average(gas["Velocities"][sel], axis=0, weights=gas["Masses"][sel])
    dv = (gas["Velocities"][sel] - vc) * u.peculiar_velocity_to_kms(snap.a)
    J = np.sum(gas["Masses"][sel][:, None] * np.cross(rel[sel], dv), axis=0)
    m = np.linalg.norm(J)
    return (J / m if m > 0 else None), radius_kpc


def plot_collimation(snap, centre, axis, outpath, rmax_kpc=10.0, rmin_kpc=0.0, title=None, params=None):
    """Angle-resolved outflow test: is there a real polar/equatorial contrast?

    For every gas cell within ``rmax_kpc`` we compute the angle
    ``alpha = arccos(|r_hat . n_hat|)`` from the jet axis (folded to [0, 90]
    because the feedback is bipolar) and look at how the radial velocity,
    temperature and mass are distributed in it.

    What success looks like: outflowing (positive v_r), hot gas concentrated at
    small alpha, with the mass still dominated by the equatorial disc.  What
    failure looks like: v_r and T flat in alpha, meaning the energy thermalised
    isotropically before it went anywhere.

    An important caveat: this measures the *state* of the gas now, which is the
    integral of past injections plus everything else that happened.  It is
    conclusive when the contrast is large and only suggestive when it is not.
    """
    if axis is None:
        print("  [diagnostics] no jet axis available -- skipping collimation test")
        return None

    u = snap.units
    L = u.length_to_kpc(snap.a)
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "Density", "InternalEnergy", "ElectronAbundance"])
    if gas.get("count", 0) == 0:
        return None

    rel = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
    r = np.linalg.norm(rel, axis=1)
    sel = (r <= rmax_kpc) & (r > max(rmin_kpc, 1e-6))
    if sel.sum() < 50:
        print("  [diagnostics] fewer than 50 cells in the aperture -- skipping collimation test")
        return None

    rel, r = rel[sel], r[sel]
    m = gas["Masses"][sel] * u.mass_to_msun
    T = u.temperature(gas["InternalEnergy"][sel], gas["ElectronAbundance"][sel] if "ElectronAbundance" in gas else None)
    nH = gas["Density"][sel] * u.density_to_nH_cgs(snap.a)

    vc = np.average(gas["Velocities"][sel], axis=0, weights=gas["Masses"][sel])
    dv = (gas["Velocities"][sel] - vc) * u.peculiar_velocity_to_kms(snap.a)
    rhat = rel / r[:, None]
    vr = np.sum(rhat * dv, axis=1)

    n = np.asarray(axis, dtype=np.float64)
    n /= np.linalg.norm(n)
    cosa = np.abs(np.sum(rhat * n, axis=1))
    alpha = np.degrees(np.arccos(np.clip(cosa, 0.0, 1.0)))  # 0 = pole, 90 = equator

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))

    abins = np.linspace(0, 90, 19)
    ac = 0.5 * (abins[1:] + abins[:-1])
    idx = np.clip(np.digitize(alpha, abins) - 1, 0, ac.size - 1)

    def wmean(v):
        num = np.bincount(idx, weights=m * v, minlength=ac.size)
        den = np.bincount(idx, weights=m, minlength=ac.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    # Solid angle per bin goes as sin(alpha), so equal-alpha bins are NOT equal
    # solid angle; normalising by it is what makes panel (c) interpretable.
    dOmega = np.cos(np.radians(abins[:-1])) - np.cos(np.radians(abins[1:]))

    a = ax[0, 0]
    a.plot(ac, wmean(vr), "o-", color=PALETTE["blue"])
    a.axhline(0, color=PALETTE["grey"], lw=0.8)
    a.set_xlabel(r"angle from jet axis $\alpha$ [deg]")
    a.set_ylabel(r"mass-weighted $v_r$ [km s$^{-1}$]")
    label_panel(a, "(a) radial velocity vs angle")

    a = ax[0, 1]
    a.plot(ac, wmean(np.log10(np.maximum(T, 1))), "o-", color=PALETTE["vermillion"])
    a.set_xlabel(r"$\alpha$ [deg]")
    a.set_ylabel(r"$\langle\log_{10}T\rangle$ [K]")
    label_panel(a, "(b) temperature vs angle")

    a = ax[0, 2]
    mass_per_sr = np.bincount(idx, weights=m, minlength=ac.size) / np.maximum(dOmega, 1e-12)
    a.plot(ac, mass_per_sr, "o-", color=C["gas"])
    a.set_yscale("log")
    a.set_xlabel(r"$\alpha$ [deg]")
    a.set_ylabel(r"gas mass per steradian [M$_\odot$]")
    label_panel(a, "(c) mass distribution vs angle")

    a = ax[1, 0]
    h = a.hexbin(alpha, vr, C=m, reduce_C_function=np.sum, gridsize=40, bins="log", cmap="magma")
    fig.colorbar(h, ax=a, label=r"$\log_{10}$ mass [M$_\odot$]")
    a.axhline(0, color="w", lw=0.7)
    a.set_xlabel(r"$\alpha$ [deg]")
    a.set_ylabel(r"$v_r$ [km s$^{-1}$]")
    label_panel(a, "(d) mass-weighted distribution", loc="upper right", color="w", stroke=True)

    a = ax[1, 1]
    # Outflow rate per unit solid angle: the physically meaningful summary.
    out = vr > 0
    mdot_out = np.bincount(idx[out], weights=(m[out] * vr[out] / np.maximum(r[out], 1e-3)), minlength=ac.size)
    mdot_out *= 1.0227e-3  # (Msun km/s / kpc) -> Msun/yr
    a.plot(ac, mdot_out / np.maximum(dOmega, 1e-12), "o-", color=PALETTE["purple"])
    a.set_yscale("log")
    a.set_xlabel(r"$\alpha$ [deg]")
    a.set_ylabel(r"$d\dot M_{\rm out}/d\Omega$ [M$_\odot$ yr$^{-1}$ sr$^{-1}$]")
    label_panel(a, "(e) outflow rate vs angle")

    a = ax[1, 2]
    # Contrast summary: polar vs equatorial, with the configured opening angles.
    pol = alpha < 30
    equ = alpha > 60
    rows = [
        ("mass fraction", m[pol].sum() / m.sum(), m[equ].sum() / m.sum()),
        (r"$\langle v_r\rangle$ [km/s]", np.average(vr[pol], weights=m[pol]) if pol.any() else np.nan, np.average(vr[equ], weights=m[equ]) if equ.any() else np.nan),
        (r"$\langle\log T\rangle$", np.average(np.log10(np.maximum(T[pol], 1)), weights=m[pol]) if pol.any() else np.nan, np.average(np.log10(np.maximum(T[equ], 1)), weights=m[equ]) if equ.any() else np.nan),
        (r"outflow mass frac.", (m[pol & out].sum() / max(m[pol].sum(), 1e-30)), (m[equ & out].sum() / max(m[equ].sum(), 1e-30))),
    ]
    a.axis("off")
    txt = ["%-22s %10s %10s" % ("", "polar<30", "equator>60")]
    for lab, p, e in rows:
        txt.append("%-22s %10.3g %10.3g" % (lab.replace("$", "").replace("\\langle", "<").replace("\\rangle", ">").replace("\\", ""), p, e))
    if params:
        for k in ("BHBipolarInjThermalTheta", "BHBipolarInjKineticTheta"):
            if k in params:
                txt.append("%-22s %10s" % (k.replace("BHBipolarInj", ""), params[k]))
    a.text(0.0, 0.80, "\n".join(txt), family="monospace", fontsize=8, va="top", transform=a.transAxes)
    a.text(0.0, 0.95, "(f) polar / equatorial contrast", fontsize=9, va="top", transform=a.transAxes)

    ttl = title or ""
    fig.suptitle("%s  (axis = [%.2f, %.2f, %.2f], r < %g kpc, z = %.2f)" % (ttl, n[0], n[1], n[2], rmax_kpc, snap.z), y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)


# ---------------------------------------------------------------------------
# 5. Mode-switch consistency
# ---------------------------------------------------------------------------


def check_mode_switch(details, inj, units, params, summary=None):
    """Does the observed thermal/kinetic split match the Eddington threshold?

    The switch (blackhole_mdot.cc:277-292) sends a BH to the *thermal* channel
    when its accretion rate exceeds ``BHBipolarInjEddThreshold`` times the
    Eddington rate, and to the *kinetic* channel otherwise.  We can predict the
    split from ``blackhole_details`` and compare against the counters the
    injection routine printed.  A large disagreement means one of the two is
    being computed on a different quantity than you think.

    lambda_Edd here is ``mdot / mdot_Edd`` -- the same quantity
    ``plot_accretion_internals`` histograms in its panel (b) -- with no
    division by ``BlackHoleAccretionFactor``.  (An earlier version of this
    check divided by it on the assumption that the switch does; that made the
    two diagnostics disagree with each other.)  ``mdot`` is additionally
    clipped at ``BlackHoleEddingtonFactor * mdot_Edd``, matching the cap the
    code itself applies (blackhole_mdot.cc:189-190) before the switch is
    evaluated -- but this cannot fix a build with ``BH_EXACT_INTEGRATION``
    (see the module docstring): there the value the switch actually acts on
    is a sub-step time average that ``blackhole_details`` never logs (only the
    pre-integration instantaneous rate is), so a residual disagreement with
    the observed split is *expected*, not a bug, whenever that flag is set.

    Returns a report string.
    """
    if details["time"].size == 0:
        return "mode-switch check: no blackhole_details rows."
    if not params:
        return "mode-switch check: no snapshot was read, so the run's parameters are unknown."
    thr = float(params.get("BHBipolarInjEddThreshold", np.nan))
    if not np.isfinite(thr):
        return "mode-switch check: BHBipolarInjEddThreshold is not among the run's parameters (single-mode build)."

    eps_r = float(params.get("BlackHoleRadiativeEfficiency", 0.1)) if params else 0.1
    edd_factor = float(params.get("BlackHoleEddingtonFactor", 1.0))
    caveats = (summary or {}).get("mdot_caveats") or []

    mbh = details["bh_mass"] * units.mass_to_msun
    mdot_raw = details["mdot"] * units.mdot_to_msun_per_yr
    edd = units.eddington_rate_msun_per_yr(np.maximum(mbh, 1e-30), eps_r)
    with np.errstate(invalid="ignore"):
        mdot = np.where(edd > 0, np.minimum(mdot_raw, edd_factor * edd), mdot_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(edd > 0, mdot / edd, np.nan)

    good = np.isfinite(lam)
    if good.any():
        # Detail rows are written on each BH's own timestep (see
        # bh_logs.bin_details_by_time): a plain mean over-weights whichever
        # black hole is taking short steps, so weight by the time to the next
        # row of the same black hole instead.
        w = np.ones_like(details["time"])
        for i in bh_logs.unique_bh_ids(details):
            m = np.where(details["id"] == i)[0]
            tm = details["time"][m]
            dt = np.diff(tm)
            w[m] = np.append(dt, dt[-1] if dt.size else 1.0)
        w = np.maximum(w, 0.0)
        wgood = w[good]
        pred_thermal = float(np.average((lam[good] >= thr).astype(float), weights=wgood)) if wgood.sum() > 0 else float(np.mean(lam[good] >= thr))
    else:
        pred_thermal = np.nan

    lines = [
        "mode-switch consistency check",
        "  BHBipolarInjEddThreshold      : %g" % thr,
        "  predicted thermal fraction    : %.3f  (from %d detail rows, dt-weighted per BH)" % (pred_thermal, int(good.sum())),
    ]
    if inj is not None and np.isfinite(inj.get("n_thermal", np.array([np.nan]))).any():
        nt, nk = np.nansum(inj["n_thermal"]), np.nansum(inj["n_kinetic"])
        obs_frac = nt / max(nt + nk, 1)
        lines.append("  observed thermal fraction     : %.3f  (%d thermal / %d kinetic BH-steps)" % (obs_frac, nt, nk))
        if np.isfinite(pred_thermal) and abs(obs_frac - pred_thermal) > 0.15:
            lines.append("  WARNING: the two disagree by more than 0.15.")
            if "BH_EXACT_INTEGRATION" in caveats:
                lines.append("           expected for this build: BH_EXACT_INTEGRATION is on, so the mdot")
                lines.append("           the switch actually compares to BHBipolarInjEddThreshold is a")
                lines.append("           sub-step time average (blackhole_mdot.cc:162-217) that never")
                lines.append("           reaches blackhole_details -- only the pre-integration instantaneous")
                lines.append("           rate does.  Not fixable from these logs; blackholes.txt's aggregate")
                lines.append("           mdot (fig 09) is the one quantity guaranteed to match what happened.")
            else:
                lines.append("           blackhole_details is written every BH step, the injection")
                lines.append("           counter only on steps where the reservoir drained -- some")
                lines.append("           difference is expected, this much is worth investigating.")
    else:
        lines.append("  observed thermal fraction     : unavailable (no MODE_SWITCH counters in stdout)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Conservation / sanity
# ---------------------------------------------------------------------------


def plot_conservation(logs, units, outpath, params=None, hist=None, title=None):
    """Cross-check the mass and energy books across independent outputs.

    Three checks, each of which has caught a real bug at some point:

      1. ``sum(BH_Mass)`` from blackholes.txt against the time integral of
         ``sum(BH_Mdot)`` -- these must agree up to mergers and the (1-eps_r)
         radiative loss;
      2. the true AGN injected energy (see :func:`bh_logs.agn_cumulative_energy`
         -- *not* energy.txt's ``e_injected``, which is unrelated hydro
         bookkeeping) against eps_f eps_r Mdot c^2 accumulated over the run;
      3. the total BH count against the number of mergers, which must only
         ever decrease by mergers.
    """
    b, e = logs["blackholes"], logs["energy"]
    if b["time"].size == 0:
        return None

    from .units import CLIGHT, SEC_PER_YEAR, SOLAR_MASS

    t = units.age_gyr(b["time"])
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))

    a = ax[0]
    m = b["mass_bh"] * units.mass_to_msun
    integ = np.zeros_like(m)
    dt_yr = np.diff(t) * 1.0e9
    rate = b["mdot_msun_per_yr"]
    integ[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt_yr)
    a.plot(t, m - m[0], color=C["bh"], label=r"$\Delta\sum M_{\rm BH}$ (log)")
    a.plot(t, integ, color=PALETTE["vermillion"], ls="--", label=r"$\int\sum\dot M\,dt$")
    eps_r = float(params.get("BlackHoleRadiativeEfficiency", 0.1)) if params else 0.1
    a.plot(t, (1.0 - eps_r) * integ, color=PALETTE["grey"], ls=":", label=r"$(1-\epsilon_r)\int\dot M\,dt$")
    a.set_yscale("log")
    a.set_xlabel("age [Gyr]")
    a.set_ylabel(r"mass grown [M$_\odot$]")
    a.legend(fontsize=7, loc="lower right")
    label_panel(a, "(a) BH mass vs integrated accretion")

    a = ax[1]
    if integ[-1] > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(integ > 0, (m - m[0]) / integ, np.nan)
        a.plot(t, ratio, color=PALETTE["purple"])
        a.axhline(1.0, color=PALETTE["grey"], ls="--", lw=1.0)
        a.axhline(1.0 - eps_r, color=PALETTE["vermillion"], ls=":", lw=1.0, label=r"$1-\epsilon_r$")
        a.legend(fontsize=7)
        # A BH-BH merger adds the secondary's whole mass to sum(BH_Mass) in one
        # step without contributing to integ(Mdot), so the ratio legitimately
        # spikes well above 1 at a merger -- a fixed (0, 2) ceiling used to
        # clip that spike and make it look like clipped/broken data instead of
        # the real, documented "up to mergers" caveat above.
        finite = ratio[np.isfinite(ratio)]
        ymax = max(2.0, float(np.nanpercentile(finite, 99.5)) * 1.1) if finite.size else 2.0
        a.set_ylim(0, ymax)
    a.set_xlabel("age [Gyr]")
    a.set_ylabel(r"$\Delta M / \int\dot M dt$")
    label_panel(a, "(b) growth ratio")

    a = ax[2]
    te, e_agn, src = bh_logs.agn_cumulative_energy(hist, logs.get("injection"), units)
    if te is not None:
        a.plot(te, e_agn, color=PALETTE["vermillion"], label=src)
        eps_f = float(params.get("BlackHoleFeedbackFactor", 0.05)) if params else 0.05
        mdot_cgs = np.interp(te, t, rate) * SOLAR_MASS / SEC_PER_YEAR
        pred = np.zeros_like(te)
        dts = np.diff(te) * 1.0e9 * SEC_PER_YEAR
        p = eps_f * eps_r * mdot_cgs * CLIGHT ** 2
        pred[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * dts)
        a.plot(te, pred, color=PALETTE["grey"], ls="--", label=r"$\epsilon_f\epsilon_r\int\dot Mc^2dt$")
        a.set_yscale("log")
        a.legend(fontsize=6.5, loc="lower right")
    else:
        a.text(0.5, 0.5, "no AGN energy source available", ha="center", va="center",
               transform=a.transAxes, fontsize=7.5, color=PALETTE["grey"])
    a.set_xlabel("age [Gyr]")
    a.set_ylabel(r"$E$ [erg]")
    label_panel(a, "(c) injected energy budget")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return savefig(fig, outpath)


def write_report(path, sections):
    """Write the collected text reports to a file and echo them to stdout."""
    text = "\n\n".join(s for s in sections if s)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print("\n  wrote %s" % path)
    return path


# ---------------------------------------------------------------------------
# Which figures mean anything for this run?
# ---------------------------------------------------------------------------

#: ``id -> (filename stem, what it needs)``.  ``needs`` is checked against the
#: model summary and the logs, so a figure that is absent can be classified as
#: "does not apply to this build" rather than "something went wrong".
FIGURE_REQUIREMENTS = (
    ("01", "overview", "snapshot history"),
    ("02", "bh_growth", "blackholes.txt or blackhole_details"),
    ("03", "star_formation", "sfr.txt and the snapshot history"),
    ("04", "energy", "energy.txt"),
    ("05", "scaling_*", "a history (snapshot or catalogue)"),
    ("06", "injection_counters", "BH_BIPOLAR_INJECTION counters on stdout"),
    ("07", "axis_stability", "blackhole_bipolar log"),
    ("08", "accretion_internals", "blackhole_details"),
    ("09", "conservation", "blackholes.txt and energy.txt"),
    ("10", "phase_diagram", "a snapshot"),
    ("11", "radial_profiles", "a snapshot"),
    ("12", "projections", "a snapshot"),
    ("13", "collimation", "a snapshot"),
    ("14", "stellar_ages", "a snapshot with star formation times"),
    ("15", "catalog_evolution", "group catalogues"),
)

NOT_APPLICABLE = "n/a"
SKIPPED = "skipped"
PRODUCED = "produced"
MISSING = "MISSING"


def figure_applicability(summary, logs, produced, has_snapshots=True, has_history=True,
                         has_catalog=True, skipped=()):
    """Classify every figure as produced / not applicable / missing.

    The distinction is the point.  A run built without ``BH_BIPOLAR_INJECTION``
    has no jet axis and no injection counters, so figures 06 and 07 *cannot*
    exist and their absence says nothing is wrong.  A run that has the counters
    but produced no figure from them has a problem.  Before this existed both
    looked identical -- a name quietly missing from the output list -- which is
    the failure mode that makes a whole analysis untrustworthy.

    ``produced`` is the list of files ``run_all`` actually wrote; ``skipped`` is
    the ids the user deliberately turned off (``--no-maps``, ``--catalog-only``),
    which must not be reported as gaps.

    Returns a list of ``(id, name, status, reason)``.
    """
    stems = {os.path.basename(p) for p in produced if p}
    collimated = bool(summary and summary.get("collimated"))
    bh_dep = (summary or {}).get("bh_deposition", "kernel")

    inj = (logs or {}).get("injection") or {}
    bip = (logs or {}).get("bipolar") or {}
    n_inj = int(np.size(inj.get("time", []))) if inj else 0
    n_bip = int(np.size(bip.get("time", []))) if bip else 0

    rows = []
    for fid, stem, needs in FIGURE_REQUIREMENTS:
        made = any(fn.startswith(fid + "_") for fn in stems)
        na = None

        if fid in skipped and not made:
            na = "skipped on request"
        elif fid in ("06", "07") and not collimated:
            na = ("no collimated channel in this build: BH feedback is isotropic, "
                  "so there is no jet axis and no cone counters")
        elif fid == "06" and collimated and n_inj == 0:
            na = ("the bipolar channel is on but stdout carries no injection counters -- "
                  "pass --stdout, or the run wrote them somewhere this did not look")
        elif fid == "07" and collimated and n_bip == 0:
            if "BH_BIPOLAR_FEEDBACK" not in ((summary or {}).get("config") or ()):
                na = ("blackhole_bipolar is written only by the legacy BH_BIPOLAR_FEEDBACK "
                      "path (blackhole_mdot.cc:486-493, gated on BH_BipolarJ); this run uses "
                      "BH_BIPOLAR_INJECTION*, which computes its own axis (BH_BipolarInjJ) and "
                      "never writes it to that log, by design (see the DESIGN NOTE in "
                      "blackhole_bipolar_injection.cc) -- this is a structural n/a for any "
                      "BH_BIPOLAR_INJECTION* build, independent of the deposition method")
            else:
                na = ("the bipolar channel is on but blackhole_bipolar is empty; with "
                      "LOCAL_INJECTION_BLACK_HOLES the cone pass is disabled and may not log")
        elif fid in ("01", "03") and not has_history:
            na = "needs the particle snapshot history, which was not built"
        elif fid in ("10", "11", "12", "13", "14") and not has_snapshots:
            na = "needs a readable snapshot"
        elif fid == "15" and not has_catalog:
            na = "needs a Subfind group catalogue"

        if made:
            note = ""
            if fid == "13":
                if not collimated:
                    note = ("expect FLAT profiles -- this is the CONTROL, not a jet test. "
                            "Angular structure here would mean the isotropic deposition is not isotropic")
                elif bh_dep == "voronoi-local":
                    note = ("expect a broad but real polar/equatorial contrast: the Voronoi-face "
                            "deposition has a smooth angular profile, so there is no sharp cone edge")
                else:
                    note = ("expect a sharp transition near BHBipolarInj*Theta: the kernel/cone "
                            "deposition has a hard edge at the opening angle")
            rows.append((fid, stem, PRODUCED, note))
        elif na:
            rows.append((fid, stem, SKIPPED if na == "skipped on request" else NOT_APPLICABLE, na))
        else:
            rows.append((fid, stem, MISSING, "needs %s -- this one is a real gap" % needs))
    return rows


def format_applicability(rows, summary=None):
    """Render :func:`figure_applicability` as a block for stdout and report.txt."""
    out = ["FIGURE APPLICABILITY"]
    if summary:
        out.append("  configuration: %s" % summary.get("scenario", "?"))
    width = max(len(n) for _, n, _, _ in rows)
    for fid, name, status, reason in rows:
        line = "  %s %-*s %-9s" % (fid, width, name, status)
        out.append(line.rstrip() if not reason else line + " -- " + reason)
    bad = [r for r in rows if r[2] == MISSING]
    out.append("")
    out.append("  %d produced, %d not applicable to this build, %d skipped on request, %d missing"
               % (sum(1 for r in rows if r[2] == PRODUCED),
                  sum(1 for r in rows if r[2] == NOT_APPLICABLE),
                  sum(1 for r in rows if r[2] == SKIPPED), len(bad)))
    if bad:
        out.append("  the %d marked MISSING are the ones worth investigating." % len(bad))
    return "\n".join(out)
