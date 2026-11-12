"""
Stellar density profiles of the target galaxy against compact JWST systems
(fig 23, report block DENSITY), after Rahul's ``plot_density_profile.py``.

Stars are the galaxy's bound stars (:func:`kinematics.bound_stars`: the
``SubhaloOffsetType`` slice, group order checked), all ages and younger than
10 Myr.  Radii run from half the stellar softening to ``10 r_1/2`` (r_1/2 =
the catalogue's 3D stellar half-mass radius), in pc:

(a) mean enclosed 3D density   rho(<r) = M(<r) / (4/3 pi r^3)       [Msun pc^-3]
(b) mean enclosed projected    Sigma(<R) = M(<R) / (pi R^2)          [Msun pc^-2],
    the mean of the x, y and z projections, stars within +-10 r_1/2 along the line of sight.

The band r < 2.8 eps_* (stellar Plummer-equivalent softening) is not resolved.

The galaxy's own half-mass point uses the stars within 10 r_1/2: M = their mass,
r_e = their projected half-mass radius (mean of the three projections).

Observations (``obs``; values checked against the papers) are half-mass points:
(a) rho = (M/2) / (4/3 pi r_3D^3) at r_3D = 1.34 r_e (Rahul's deprojection, close
to the r_1/2 ~ 4/3 R_e of Wolf et al. 2010); (b) Sigma_eff = (M/2) / (pi r_e^2) at
R = r_e.  Sizes are circularised, r_e sqrt(b/a), to match the circular apertures;
Baggen+23 masses converted Salpeter -> Chabrier (-0.24 dex).  Observed sizes are
light-weighted, the simulation mass-weighted.
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from . import obs

R3D_OVER_RE = 1.34
OUTER_RHALF = 10.0
YOUNG_MYR = 10.0
NGRID = 120


def rho_half(m, re):
    """(M/2) / (4/3 pi (1.34 r_e)^3)."""
    return 0.5 * np.asarray(m) / (4.0 / 3.0 * np.pi * (R3D_OVER_RE * np.asarray(re, dtype=float)) ** 3)


def sigma_eff(m, re):
    """(M/2) / (pi r_e^2)."""
    return 0.5 * np.asarray(m) / (np.pi * np.asarray(re, dtype=float) ** 2)


def enclosed_rho(r, m, grid):
    """Mean enclosed 3D density on ``grid`` (same length units as r)."""
    if r.size == 0:
        return np.zeros_like(grid)
    o = np.argsort(r)
    M = np.interp(grid, r[o], np.cumsum(m[o]), left=0.0)
    return M / (4.0 / 3.0 * np.pi * grid ** 3)


def enclosed_sigma(pos, m, grid, depth):
    """Mean enclosed projected density on ``grid``, averaged over the x, y, z
    projections, elements within +-``depth`` along each line of sight."""
    out = []
    for ax in range(3):
        keep = np.abs(pos[:, ax]) < depth
        q = np.delete(pos[keep], ax, axis=1)
        R = np.hypot(q[:, 0], q[:, 1])
        if R.size == 0:
            out.append(np.zeros_like(grid))
            continue
        o = np.argsort(R)
        out.append(np.interp(grid, R[o], np.cumsum(m[keep][o]), left=0.0) / (np.pi * grid ** 2))
    return np.mean(out, axis=0)


def projected_half_mass_radius(pos, m):
    """Mean over x, y, z projections of the radius enclosing half the mass."""
    res = []
    for ax in range(3):
        q = np.delete(pos, ax, axis=1)
        R = np.hypot(q[:, 0], q[:, 1])
        o = np.argsort(R)
        res.append(R[o][np.searchsorted(np.cumsum(m[o]), 0.5 * m.sum())])
    return float(np.mean(res))


def compute(pos_pc, m, age_myr, rhalf_pc, eps_pc=None, young_myr=YOUNG_MYR):
    """Profiles and half-mass numbers from centred star arrays (pc, Msun, Myr)."""
    r = np.linalg.norm(pos_pc, axis=1)
    rout = OUTER_RHALF * rhalf_pc
    lo = 0.5 * eps_pc if eps_pc else 0.01 * rhalf_pc
    grid = np.logspace(np.log10(lo), np.log10(rout), NGRID)
    young = age_myr < young_myr
    prof = {}
    for lab, sel in (("all", np.ones(len(m), bool)), ("young", young)):
        prof[lab] = dict(rho=enclosed_rho(r[sel], m[sel], grid),
                         sigma=enclosed_sigma(pos_pc[sel], m[sel], grid, rout), n=int(sel.sum()))
    g = r < rout
    Mg = float(m[g].sum())
    re = projected_half_mass_radius(pos_pc[g], m[g]) if g.sum() > 1 else np.nan
    rho_at = {x: float(enclosed_rho(r, m, np.array([x]))[0]) for x in (10.0, 20.0, 50.0)}
    sig100 = float(enclosed_sigma(pos_pc, m, np.array([100.0]), rout)[0])
    return dict(grid=grid, prof=prof, rhalf_pc=rhalf_pc, eps_pc=eps_pc, rout_pc=rout, m_out=Mg, re_pc=re,
                sigma_eff=float(sigma_eff(Mg, re)), rho_half=float(rho_half(Mg, re)), rho_at=rho_at,
                sigma_100=sig100, young_myr=young_myr)


def analyse(snap, centre, young_myr=YOUNG_MYR):
    """Density profiles of the target galaxy's bound stars about ``centre``.
    Returns (table or None, note)."""
    from . import kinematics, maps

    st, note = kinematics.bound_stars(snap)
    if st is None:
        return None, note
    u, a = snap.units, snap.a
    L = u.length_to_kpc(a)
    pos = io.periodic_delta(st["Coordinates"], centre, snap.boxsize) * L * 1e3
    m = np.asarray(st["Masses"], dtype=np.float64) * u.mass_to_msun
    age = (snap.age_gyr - u.age_gyr(np.asarray(st["GFM_StellarFormationTime"], dtype=np.float64))) * 1e3
    eps = maps.softening_kpc(snap, parttypes=(io.PT_STAR,))
    return compute(pos, m, age, st["rhalf_kpc"] * 1e3, eps * 1e3 if eps else None, young_myr), note


def report_lines(tab, note):
    out = ["DENSITY (bound stars of the target galaxy; mass-weighted)", "  " + note]
    if tab is None:
        return "\n".join(out)
    out += ["  stellar softening eps* = %s; unresolved r < 2.8 eps* = %s" % (
                ("%.3g pc" % tab["eps_pc"]) if tab["eps_pc"] else "n/a",
                ("%.3g pc" % (2.8 * tab["eps_pc"])) if tab["eps_pc"] else "n/a"),
            "  3D r_1/2 (catalogue) = %.3g pc; within %g r_1/2: M* = %.3g Msun, projected r_e = %.3g pc"
            % (tab["rhalf_pc"], OUTER_RHALF, tab["m_out"], tab["re_pc"]),
            "  Sigma_eff = %.3g Msun/pc^2; rho_half (r_3D = 1.34 r_e) = %.3g Msun/pc^3; Sigma(<100 pc) = %.3g Msun/pc^2"
            % (tab["sigma_eff"], tab["rho_half"], tab["sigma_100"]),
            "  rho(<10 pc) = %.3g, rho(<20 pc) = %.3g, rho(<50 pc) = %.3g Msun/pc^3"
            % (tab["rho_at"][10.0], tab["rho_at"][20.0], tab["rho_at"][50.0])]
    if tab["eps_pc"] and tab["eps_pc"] * 2.8 > 10.0:
        out.append("  (radii below %.3g pc are inside the softening kernel)" % (2.8 * tab["eps_pc"]))
    return "\n".join(out)


def observation_points():
    """Half-mass points of the JWST samples: dict name -> (re_pc array, M array,
    extra) with circularised sizes and Chabrier masses."""
    b23 = np.array([row[2:] for row in obs.BAGGEN23], dtype=float)   # logM, re, +, -, b/a
    sq = np.sqrt(b23[:, 4])
    b24 = np.array([row[2:] for row in obs.BAGGEN24], dtype=float)   # med, min, max, re, b/a
    m25 = obs.MESSA25
    return {
        "baggen23": dict(re=b23[:, 1] * sq, m=10 ** (b23[:, 0] + obs.SALPETER_TO_CHABRIER_DEX),
                         re_err=np.vstack([b23[:, 3], b23[:, 2]]) * sq),
        "baggen24": dict(re=b24[:, 3] * np.sqrt(b24[:, 4]), m=10 ** b24[:, 0], m_lo=10 ** b24[:, 1], m_hi=10 ** b24[:, 2]),
        "gnz11": dict(re=np.array([obs.GNZ11["r_e"]]), re_err=np.array([obs.GNZ11["r_e_err"]]),
                      m=np.array([10 ** obs.GNZ11["logm"]]), m_lo=np.array([10 ** (obs.GNZ11["logm"] - obs.GNZ11["logm_lo"])]),
                      m_hi=np.array([10 ** (obs.GNZ11["logm"] + obs.GNZ11["logm_hi"])])),
        "messa25": dict(re=np.array([x[2] for x in m25]), m=10 ** np.array([x[1] for x in m25]),
                        upper=np.array([x[3] for x in m25])),
    }


def plot(tab, outpath, title=None, z=None):
    """Fig 23: (a) rho(<r), (b) Sigma(<R), with the JWST half-mass points."""
    from .plotstyle import PALETTE, label_panel, plt, savefig

    O = observation_points()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0))
    g = tab["grid"]
    for i, (a, key, xf, fn, ylab) in enumerate((
            (ax[0], "rho", R3D_OVER_RE, rho_half, r"$\langle\rho_\star\rangle(<r)$ [M$_\odot$ pc$^{-3}$]"),
            (ax[1], "sigma", 1.0, sigma_eff, r"$\langle\Sigma_\star\rangle(<R)$ [M$_\odot$ pc$^{-2}$]"))):
        if tab["eps_pc"]:
            a.axvspan(1e-3, 2.8 * tab["eps_pc"], color="0.9", lw=0, zorder=0)
            a.text(2.8 * tab["eps_pc"], 0.03 if i == 1 else 0.97, "unresolved\n($<2.8\\,\\epsilon_\\star$) ",
                   transform=a.get_xaxis_transform(), ha="right", va="bottom" if i == 1 else "top", fontsize=7,
                   color="0.45")
        for lab, col, txt in (("all", PALETTE["black"], "all bound stars"),
                              ("young", PALETTE["blue"], "stars < %g Myr" % tab["young_myr"])):
            y = tab["prof"][lab][key]
            a.plot(g, np.where(y > 0, y, np.nan), color=col, lw=2, label=txt)
        a.plot(xf * tab["re_pc"], fn(tab["m_out"], tab["re_pc"]), marker="*", ms=15, color=PALETTE["black"],
               mec="white", mew=1.0, ls="none", zorder=6, label="this galaxy, half-mass point ($r_e$ = %.3g pc)" % tab["re_pc"])
        b = O["baggen23"]
        a.errorbar(xf * b["re"], fn(b["m"], b["re"]), xerr=xf * b["re_err"], fmt="s", ms=6, color=PALETTE["vermillion"],
                   mec="white", mew=0.7, elinewidth=1, label="Baggen+23, $z_{\\rm phot}\\sim$7-9")
        b = O["baggen24"]
        y = fn(b["m"], b["re"])
        a.errorbar(xf * b["re"], y, yerr=[y - fn(b["m_lo"], b["re"]), fn(b["m_hi"], b["re"]) - y], fmt="D", ms=6,
                   color=PALETTE["green"], mec="white", mew=0.7, elinewidth=1,
                   label="Baggen+24, $z$ = 6.7-8.4 (bars: max-AGN to no-AGN mass)")
        b = O["gnz11"]
        y = fn(b["m"], b["re"])
        a.errorbar(xf * b["re"], y, xerr=xf * b["re_err"], yerr=[y - fn(b["m_lo"], b["re"]), fn(b["m_hi"], b["re"]) - y],
                   fmt="^", ms=7, color=PALETTE["orange"], mec="white", mew=0.7, elinewidth=1,
                   label="GN-z11, $z$ = 10.6 (Tacchella+23)")
        b = O["messa25"]
        y = fn(b["m"], b["re"])
        a.errorbar(xf * b["re"], y, yerr=np.where(b["upper"], 0.4 * y, 0.0), lolims=b["upper"], fmt="o", ms=5,
                   color=PALETTE["purple"], mec="white", mew=0.7, elinewidth=1, capsize=2,
                   label="Cosmic Gems clusters, $z$ = 9.6 (Messa+25; arrows: $R_{\\rm eff}$ upper limits)")
        if key == "sigma":
            a.axhline(obs.HOPKINS10_SIGMA_MAX_PC2, color="0.45", lw=1, ls="--")
            a.text(0.98, obs.HOPKINS10_SIGMA_MAX_PC2 * 1.15, r"$\Sigma_{\rm max}$ (Hopkins+10)", fontsize=7.5,
                   color="0.45", ha="right", va="bottom", transform=a.get_yaxis_transform())
        a.set_xscale("log")
        a.set_yscale("log")
        # the axis reaches down to the smallest observed size (the Cosmic Gems clusters, ~0.4 pc);
        # the curves themselves start at half the softening
        xmin = min(g[0], 0.5 * xf * min(float(np.min(O[k]["re"])) for k in O))
        a.set_xlim(xmin, max(g[-1], 3.0 * tab["re_pc"]))
        a.set_ylabel(ylab)
        label_panel(a, "(a) mean enclosed density" if i == 0 else "(b) mean enclosed surface density", loc="upper right")
    ax[0].set_xlabel(r"$r$ [pc]  (points at $r_{\rm 3D} = 1.34\,r_e$)")
    ax[1].set_xlabel(r"$R$ [pc]  (points at $r_e$; mean of x, y, z projections)")
    ax[0].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.suptitle("%s%s   bound stars, mass-weighted; observations light-weighted, sizes circularised, Chabrier"
                 % (title or "", ("   z = %.2f" % z) if z is not None else ""), fontsize=9)
    fig.subplots_adjust(bottom=0.12, top=0.9, wspace=0.22, left=0.07, right=0.98)
    return savefig(fig, outpath)
