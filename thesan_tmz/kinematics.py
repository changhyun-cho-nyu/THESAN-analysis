"""
Stellar and gas kinematics of the target galaxy (fig 22), after
``angmom_axes.py`` (spin directions) and ``plot_faceon.py`` (v/sigma).

Stars are the galaxy's **bound** stars: the slice
``[SubhaloOffsetType[j][4], + SubhaloLenType[j][4])`` of the group-ordered
snapshot (AREPO writes ``SubhaloOffsetType``, ``subfind.cc:865``), wind
particles dropped.  Before using the slice we check that the snapshot really is
group-ordered: half of the slice's stellar mass must lie within the catalogue's
own ``SubhaloHalfmassRadType[4]`` of ``SubhaloPos`` (0.4-0.6 accepted).  Gas is
every cell inside the radius (bound or not).

Radii are multiples of the stellar half-mass radius r_1/2 (``RADII_RHALF``).
Velocities are physical peculiar km/s about the mass-weighted mean velocity of
stars + gas within the largest radius.

Per sphere r < R:
    L_* , L_gas, L_SF-gas = sum m (r x v);  the angles between them, and of L_*
    and L_gas to the galaxy's spin axis (L of the bound stars within 3 r_1/2,
    the same sphere fig 20 uses).
Per shell (and per sphere), for all stars and stars younger than 10 Myr, in
the frame of the spin axis:
    v = mass-weighted <v_phi>;  sigma_z = rms v_z;
    sigma_1D = sqrt((sigma_R^2 + sigma_phi^2 + sigma_z^2) / 3), sigma_R = rms v_R,
    sigma_phi = dispersion of v_phi about its mean.
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io

RADII_RHALF = (0.5, 1.0, 2.0, 3.0, 5.0)
YOUNG_MYR = 10.0
AXIS_RHALF = 3.0
MIN_STARS = 10
HALF_MASS_OK = (0.4, 0.6)


def angle_deg(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (na > 0 and nb > 0) or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def spin(pos, vel, m, sel):
    """sum m (r x v) over ``sel``; NaN when empty."""
    if not np.any(sel):
        return np.full(3, np.nan)
    return np.sum(m[sel, None] * np.cross(pos[sel], vel[sel]), axis=0)


def v_over_sigma(pos, vel, m, axis, sel):
    """(v, sigma_z, sigma_1D, n) of the elements in ``sel``, in the frame whose
    z is ``axis``.  NaN when fewer than ``MIN_STARS``."""
    n = int(np.sum(sel))
    if n < MIN_STARS:
        return np.nan, np.nan, np.nan, n
    e3 = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(tmp, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.vstack([e1, e2, e3])
    q, vq, w = pos[sel] @ B.T, vel[sel] @ B.T, m[sel]
    R = np.hypot(q[:, 0], q[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        vphi = np.where(R > 0, (q[:, 0] * vq[:, 1] - q[:, 1] * vq[:, 0]) / R, 0.0)
        vR = np.where(R > 0, (q[:, 0] * vq[:, 0] + q[:, 1] * vq[:, 1]) / R, 0.0)
    W = w.sum()
    v = float(np.sum(w * vphi) / W)
    sz = float(np.sqrt(np.sum(w * vq[:, 2] ** 2) / W))
    sR = float(np.sqrt(np.sum(w * vR ** 2) / W))
    sphi = float(np.sqrt(np.sum(w * (vphi - v) ** 2) / W))
    return v, sz, float(np.sqrt((sR ** 2 + sphi ** 2 + sz ** 2) / 3.0)), n


def compute(stars, gas, rhalf_kpc, radii_rhalf=RADII_RHALF, young_myr=YOUNG_MYR, axis=None):
    """The kinematics table from centred arrays.

    ``stars``: pos [kpc], vel [km/s], mass, age_myr;  ``gas``: pos, vel, mass, sfr.
    Positions relative to the galaxy centre, velocities not yet in any frame.
    Returns a dict (radii in kpc and r_1/2, spheres, shells, axis, frame velocity)."""
    radii = np.asarray(radii_rhalf, dtype=np.float64) * rhalf_kpc
    rmax = radii.max()
    rs, rg = np.linalg.norm(stars["pos"], axis=1), np.linalg.norm(gas["pos"], axis=1)
    fs, fg = rs < rmax, rg < rmax
    msum = stars["mass"][fs].sum() + gas["mass"][fg].sum()
    vframe = ((np.sum(stars["mass"][fs, None] * stars["vel"][fs], axis=0)
               + np.sum(gas["mass"][fg, None] * gas["vel"][fg], axis=0)) / msum) if msum > 0 else np.zeros(3)
    vs, vg = stars["vel"] - vframe, gas["vel"] - vframe
    if axis is None:
        axis = spin(stars["pos"], vs, stars["mass"], rs < AXIS_RHALF * rhalf_kpc)
    axis = np.asarray(axis, dtype=np.float64)
    young = stars["age_myr"] < young_myr
    sf = gas["sfr"] > 0

    spheres = []
    for R in radii:
        s, g = rs < R, rg < R
        Ls, Lg, Lsf = spin(stars["pos"], vs, stars["mass"], s), spin(gas["pos"], vg, gas["mass"], g), \
            spin(gas["pos"], vg, gas["mass"], g & sf)
        row = dict(r_kpc=float(R), mstar=float(stars["mass"][s].sum()), mgas=float(gas["mass"][g].sum()),
                   msfgas=float(gas["mass"][g & sf].sum()),
                   ang_star_gas=angle_deg(Ls, Lg), ang_star_sfgas=angle_deg(Ls, Lsf),
                   ang_star_axis=angle_deg(Ls, axis), ang_gas_axis=angle_deg(Lg, axis))
        for lab, sel in (("all", s), ("young", s & young)):
            v, sz, s1, n = v_over_sigma(stars["pos"], vs, stars["mass"], axis, sel)
            row.update({"%s_v" % lab: v, "%s_sigz" % lab: sz, "%s_sig1d" % lab: s1, "%s_n" % lab: n})
        spheres.append(row)

    shells = []
    edges = np.concatenate([[0.0], radii])
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rs >= lo) & (rs < hi)
        row = dict(r_lo_kpc=float(lo), r_hi_kpc=float(hi))
        for lab, sel in (("all", s), ("young", s & young)):
            v, sz, s1, n = v_over_sigma(stars["pos"], vs, stars["mass"], axis, sel)
            row.update({"%s_v" % lab: v, "%s_sigz" % lab: sz, "%s_sig1d" % lab: s1, "%s_n" % lab: n})
        shells.append(row)
    return dict(rhalf_kpc=float(rhalf_kpc), radii_rhalf=list(radii_rhalf), spheres=spheres, shells=shells,
                axis=axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis, vframe=vframe,
                young_myr=young_myr)


def bound_stars(snap, cat=None, j=None):
    """The target galaxy's bound star particles (winds dropped) from the
    group-ordered snapshot.  Returns (dict or None, reason/info string)."""
    from . import catalog

    cat = cat if cat is not None else catalog.load(snap)
    if cat is None:
        return None, "no group catalogue"
    j = cat.target_index() if j is None else j
    if j is None:
        return None, "no target galaxy in the catalogue"
    off, ln = cat.s("offsettype"), cat.s("lentype")
    if off is None or ln is None:
        return None, "catalogue has no SubhaloOffsetType: bound stars cannot be picked"
    o, n = int(np.asarray(off)[j, io.PT_STAR]), int(np.asarray(ln)[j, io.PT_STAR])
    if n < MIN_STARS:
        return None, "target galaxy has %d star particles" % n
    fields = ["Coordinates", "Masses", "Velocities", "GFM_StellarFormationTime"]
    st = snap.part(io.PT_STAR, fields)
    if int(st.get("count", 0)) < o + n:
        return None, "SubhaloOffsetType points past the end of the star particles"
    d = {k: np.asarray(st[k])[o:o + n] for k in fields if k in st}

    # group-ordering check against the catalogue's own half-mass radius
    L = snap.units.length_to_kpc(snap.a)
    hr = cat.s("halfmassradtype")
    pos_j = np.asarray(cat.s("pos"))[j]
    rh = float(np.asarray(hr)[j, io.PT_STAR]) * L if hr is not None else np.nan
    r = np.linalg.norm(io.periodic_delta(d["Coordinates"], pos_j, snap.boxsize) * L, axis=1)
    m = np.asarray(d["Masses"], dtype=np.float64)
    frac = float(m[r <= rh].sum() / m.sum()) if rh > 0 and m.sum() > 0 else np.nan
    if not (HALF_MASS_OK[0] <= frac <= HALF_MASS_OK[1]):
        return None, ("the slice is not the galaxy's stars (mass within its own r_1/2 = %.2f, not ~0.5): "
                      "snapshot not group-ordered?" % frac)
    if "GFM_StellarFormationTime" in d:
        keep = np.asarray(d["GFM_StellarFormationTime"]) > 0
        d = {k: v[keep] for k, v in d.items()}
    d["rhalf_kpc"] = rh
    return d, "subhalo %d: %d bound star particles, %.2f of their mass within r_1/2 = %.3g kpc" % (j, n, frac, rh)


def analyse(snap, centre, young_myr=YOUNG_MYR):
    """Kinematics of the target galaxy about ``centre``.  Returns (table or
    None, note)."""
    st, note = bound_stars(snap)
    if st is None:
        return None, note
    u, a = snap.units, snap.a
    L, vconv = u.length_to_kpc(a), u.peculiar_velocity_to_kms(a)
    rh = st["rhalf_kpc"]
    rmax = max(RADII_RHALF) * rh
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "StarFormationRate"])
    gpos = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
    near = np.linalg.norm(gpos, axis=1) < rmax
    tform = np.asarray(st["GFM_StellarFormationTime"], dtype=np.float64)
    stars = dict(pos=io.periodic_delta(st["Coordinates"], centre, snap.boxsize) * L,
                 vel=np.asarray(st["Velocities"], dtype=np.float64) * vconv,
                 mass=np.asarray(st["Masses"], dtype=np.float64) * u.mass_to_msun,
                 age_myr=(snap.age_gyr - u.age_gyr(tform)) * 1e3)
    g = dict(pos=gpos[near], vel=np.asarray(gas["Velocities"], dtype=np.float64)[near] * vconv,
             mass=np.asarray(gas["Masses"], dtype=np.float64)[near] * u.mass_to_msun,
             sfr=np.asarray(gas["StarFormationRate"])[near] if "StarFormationRate" in gas else np.zeros(int(near.sum())))
    return compute(stars, g, rh, young_myr=young_myr), note


def report_lines(tab, note):
    """KINEMATICS block for report.txt."""
    out = ["KINEMATICS (bound stars of the target galaxy; gas = all cells in the sphere)", "  " + note]
    if tab is None:
        return "\n".join(out)
    out.append("  spin axis (L of the bound stars within %g r_1/2) = %s" % (AXIS_RHALF, np.round(tab["axis"], 3)))
    out.append("  %8s %6s %9s %9s %9s | %9s %9s %9s %9s" % ("r[kpc]", "r/r1/2", "M*", "Mgas", "Msfgas", "ang(*,g)",
                                                          "ang(*,sf)", "ang(*,ax)", "ang(g,ax)"))
    for x, k in zip(tab["spheres"], tab["radii_rhalf"]):
        out.append("  %8.3g %6g %9.3g %9.3g %9.3g | %9.1f %9.1f %9.1f %9.1f" % (
            x["r_kpc"], k, x["mstar"], x["mgas"], x["msfgas"], x["ang_star_gas"], x["ang_star_sfgas"],
            x["ang_star_axis"], x["ang_gas_axis"]))
    out.append("  v/sigma (v = <v_phi>, km/s; young = < %g Myr)" % tab["young_myr"])
    out.append("  %-17s %8s %8s %8s %8s %8s | %8s %8s %8s %8s" % ("", "v", "sig_z", "v/sig_z", "v/sig1D", "n",
                                                              "v_yng", "v/sig_z", "v/sig1D", "n_yng"))

    def line(lab, x):
        f = lambda p: (x[p + "_v"] / x[p + "_sigz"], x[p + "_v"] / x[p + "_sig1d"]) if x[p + "_sigz"] > 0 else (np.nan,
                                                                                                             np.nan)
        a_, y_ = f("all"), f("young")
        return "  %-17s %8.1f %8.1f %8.2f %8.2f %8d | %8.1f %8.2f %8.2f %8d" % (
            lab, x["all_v"], x["all_sigz"], a_[0], a_[1], x["all_n"], x["young_v"], y_[0], y_[1], x["young_n"])

    for x in tab["shells"]:
        out.append(line("shell %.3g-%.3g" % (x["r_lo_kpc"], x["r_hi_kpc"]), x))
    for x in tab["spheres"]:
        out.append(line("r < %.3g" % x["r_kpc"], x))
    return "\n".join(out)


def plot(tab, outpath, title=None, z=None):
    """Fig 22: (a) spin misalignment against radius, (b) v/sigma in shells."""
    from .plotstyle import PALETTE, label_panel, plt, savefig

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))
    r = np.array([x["r_kpc"] for x in tab["spheres"]])
    a = ax[0]
    for key, lab, col, ls in (("ang_star_gas", r"$L_\star$ vs $L_{\rm gas}$", PALETTE["blue"], "-"),
                              ("ang_star_sfgas", r"$L_\star$ vs $L_{\rm SF\,gas}$", PALETTE["green"], "-"),
                              ("ang_star_axis", r"$L_\star(<r)$ vs spin axis", PALETTE["blue"], ":"),
                              ("ang_gas_axis", r"$L_{\rm gas}(<r)$ vs spin axis", PALETTE["orange"], ":")):
        a.plot(r, [x[key] for x in tab["spheres"]], ls=ls, marker="o", ms=4, color=col, label=lab)
    a.set_xscale("log")
    a.set_ylim(0, 180)
    a.set_xlabel("r [kpc]  (spheres at %s r$_{1/2,\\star}$)" % ", ".join("%g" % k for k in tab["radii_rhalf"]))
    a.set_ylabel("angle [deg]")
    a.axvline(tab["rhalf_kpc"], color="0.6", lw=0.8, ls="--")
    a.legend(fontsize=8, loc="upper left")
    label_panel(a, "(a) spin directions within r", loc="upper right")

    b = ax[1]
    mid = np.array([np.sqrt(max(x["r_lo_kpc"], 0.5 * x["r_hi_kpc"]) * x["r_hi_kpc"]) for x in tab["shells"]])
    for p, col, lab in (("all", PALETTE["blue"], "all stars"),
                        ("young", PALETTE["orange"], "stars < %g Myr" % tab["young_myr"])):
        with np.errstate(invalid="ignore", divide="ignore"):
            vz = np.array([x[p + "_v"] / x[p + "_sigz"] for x in tab["shells"]])
            v1 = np.array([x[p + "_v"] / x[p + "_sig1d"] for x in tab["shells"]])
        b.plot(mid, vz, "-o", ms=4, color=col, label=r"%s: $v/\sigma_z$" % lab)
        b.plot(mid, v1, "--s", ms=3, color=col, label=r"%s: $v/\sigma_{\rm 1D}$" % lab)
    b.axhline(1.0, color="0.5", lw=0.8)
    b.axvline(tab["rhalf_kpc"], color="0.6", lw=0.8, ls="--")
    b.set_xscale("log")
    b.set_xlabel("r [kpc]  (shells between the spheres)")
    b.set_ylabel(r"$\langle v_\phi\rangle / \sigma$")
    b.legend(fontsize=8, loc="best")
    label_panel(b, "(b) rotation support", loc="lower right")
    fig.suptitle("%s%s   bound stars; r$_{1/2,\\star}$ = %.3g kpc (dashed)" % (
        title or "", ("   z = %.2f" % z) if z is not None else "", tab["rhalf_kpc"]), fontsize=9)
    fig.subplots_adjust(bottom=0.15, top=0.9, wspace=0.25)
    return savefig(fig, outpath)
