"""
How well SLR_BH refines the mesh around a black hole, from a snapshot.

SLR_BH (modules/SLR/blackhole_refinement.cc) gives every gas cell within a
refinement radius R_ref of a BH a target cell radius that grows linearly from
r_min at the BH to r_max at R_ref:

    target(r) = r_min + (r / R_ref) (r_max - r_min)                     (:245-246)

    R_ref = RefBHRadiusHSML x BH_Hsml          (SLR_BH_FIXED: RefBHRadius)
    r_max = RefBHMaxCellRadiusHSML x BH_Hsml   (SLR_BH_FIXED: RefBHMaxCellRadius)
    r_min = RefBHMinCellRadiusRBondi x R_B     (SLR_BH_FIXED: RefBHMinCellRadius)
    R_B   = 50 pc (M_BH / 1e7 Msun) (c_s / 30 km/s)^-2, c_s^2 = gamma (gamma-1) BH_U   (:55-58)

with r_min = r_max / 1e4 when r_min > r_max or R_B is not finite (:237-241), and
the smallest target wherever two BH regions overlap (:249-253).  A cell inside a
region is refined when its radius (3V/4pi)^(1/3) exceeds the target and its mass
exceeds RefBHMinCellMass (criterion_refinement.cc:207-237), and de-refined when its
mass is below RefBHMinCellMass/4, or its radius below target/RefBHLowerFactorC
with a mass below TargetGasMass/2 (criterion_derefinement.cc:169-177).  Outside,
the usual mass criterion holds: split above 2 TargetGasMass, merge below half.

The flags are not written to snapshots, but everything they are computed from is
(positions, masses, densities, BH_Hsml, BH_Mass, BH_U), so this module rebuilds
them and shows, for one BH:

  (a) cell radius vs distance, each cell coloured by what the refinement should do
      with it, against the target ramp;
  (b) cell mass vs distance, against RefBHMinCellMass and the TargetGasMass band;
  (c) the fraction of cells on target per radial bin;
  (d) a slice through the BH showing the actual Voronoi cells (each pixel takes
      the nearest generating point, which is exactly the Voronoi cell), coloured
      by cell radius, with R_ref and R_B drawn; (e) the same, zoomed in;
  (f) the numbers.

With SLR_BH = 0 a cell is only split when can_this_cell_be_split() allows it
(e.g. its timestep is synchronised), which a snapshot cannot tell: a few "too big"
cells right after the refinement region moved are normal.

    python3 -m thesan_tmz.slr_mesh OUTPUT_DIR [-s SNAP] [--bh-id ID] [--out slr_mesh.png]
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from .plotstyle import label_panel, plt, savefig
from .units import GAMMA, GAMMA_MINUS1, SOLAR_MASS

PARSEC_CM = 3.085678e18

# what the refinement should do with a cell
OUTSIDE, ON_TARGET, TOO_BIG, AT_FLOOR, TOO_SMALL, BELOW_FLOOR = range(6)
STATUS = {
    ON_TARGET: ("on target", "#2a9d8f"),
    TOO_BIG: ("too big: should be split", "#e63946"),
    AT_FLOOR: ("too big, but at RefBHMinCellMass", "#f4a261"),
    TOO_SMALL: ("too small: should be merged", "#457b9d"),
    BELOW_FLOOR: ("below RefBHMinCellMass/4: merged", "#6d597a"),
    OUTSIDE: ("outside the region", "#b0b0b0"),
}


def slr_settings(snap):
    """The SLR_BH switches and parameters of this run, or None without SLR_BH."""
    cfg, par = snap.config, snap.parameters
    if not io.has_option(cfg, "SLR_BH"):
        return None

    def num(k):
        return float(par[k]) if k in par else None

    s = dict(force=str(cfg.get("SLR_BH", "")).strip() in ("1", "1.0"), fixed=io.has_option(cfg, "SLR_BH_FIXED"),
             min_mass=num("RefBHMinCellMass") or 0.0, lower_c=num("RefBHLowerFactorC") or 1.0)
    if s["fixed"]:
        s.update(radius=num("RefBHRadius"), rmax=num("RefBHMaxCellRadius"), rmin=num("RefBHMinCellRadius"))
    else:
        s.update(radius_hsml=num("RefBHRadiusHSML"), rmax_hsml=num("RefBHMaxCellRadiusHSML"),
                 rmin_rb=num("RefBHMinCellRadiusRBondi"))
    ref, fac = num("ReferenceGasPartMass") or 0.0, num("TargetGasMassFactor") or 1.0
    s["target_mass"] = ref * fac if ref > 0 else None
    return s


def code_bondi_radius(bh_mass, bh_u, units):
    """R_B exactly as blackhole_refinement.cc:55-58 computes it (code length)."""
    cs = np.sqrt(GAMMA * GAMMA_MINUS1 * np.asarray(bh_u, dtype=np.float64))
    m = np.asarray(bh_mass, dtype=np.float64)              # float32 in snapshots: convert before the units
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (50.0 * PARSEC_CM / units.UnitLength_in_cm
                * (m * units.UnitMass_in_g / units.h) / (1.0e7 * SOLAR_MASS)
                * (cs * units.UnitVelocity_in_cm_per_s / 30.0e5) ** -2)


def bh_region(s, hsml, rb):
    """(R_ref, r_min, r_max) of one BH (code length), as the C++ sets them."""
    if s["fixed"]:
        R, rmax, rmin = s["radius"], s["rmax"], s["rmin"]
    else:
        R, rmax, rmin = s["radius_hsml"] * hsml, s["rmax_hsml"] * hsml, s["rmin_rb"] * rb
    if not (rmin <= rmax) or not np.isfinite(rb):
        rmin = rmax / 1.0e4
    return R, rmin, rmax


def target_radius(r, R, rmin, rmax):
    """Target cell radius at distance r inside the region (code length)."""
    return rmin + (np.asarray(r) / R) * (rmax - rmin)


def classify(rcell, mass, target, s, target_mass):
    """What the refinement criteria do with each cell (see STATUS).  ``target`` is
    np.inf outside every BH region."""
    st = np.full(rcell.shape, OUTSIDE, dtype=int)
    inside = np.isfinite(target)
    big = inside & (rcell > target)
    st[inside] = ON_TARGET
    st[big & (mass > s["min_mass"])] = TOO_BIG
    st[big & (mass <= s["min_mass"])] = AT_FLOOR
    small = inside & (rcell < target / s["lower_c"]) & (mass < 0.5 * target_mass)
    st[small & ~big] = TOO_SMALL
    st[inside & (mass < s["min_mass"] / 4.0)] = BELOW_FLOOR
    return st


def choose_bh(snap, hide_lowres=True):
    """The most massive BH of the target galaxy (host = the catalogue's target), else
    the most massive BH in the box that is not in low-res gas (see :mod:`lowres`).
    Returns (index into PartType5, note); index None when there is no candidate."""
    bh = snap.part(io.PT_BH, ["ParticleIDs", "BH_Mass"])
    if not bh.get("count"):
        return None, "no black holes"
    try:
        from . import bh_population, catalog

        cat = catalog.load(snap)
        tgt = cat.target_index()
        t = bh_population.bh_table(snap, cat, hide_lowres=hide_lowres)
        mine = np.where(np.asarray(t["host_subhalo"]) == tgt)[0]
        if mine.size:
            bid = int(t["id"][mine[np.argmax(np.asarray(t["mass_msun"])[mine])]])
            return int(np.where(bh["ParticleIDs"] == bid)[0][0]), "the most massive BH of the target galaxy"
    except Exception:
        pass
    mass = np.asarray(bh["BH_Mass"], dtype=np.float64)
    if hide_lowres:
        from . import lowres

        low = lowres.lowres_ids(snap)
        mass = np.where(np.isin(np.asarray(bh["ParticleIDs"], dtype=np.int64), list(low)), -np.inf, mass)
        if not np.isfinite(mass).any():
            return None, "every BH is in low-res gas"
    return int(np.argmax(mass)), "the most massive BH in the box (no catalogue match)"


def analyse(snap, k=None, s=None):
    """Everything the figure needs, for BH number ``k`` (index into PartType5)."""
    s = s or slr_settings(snap)
    if s is None:
        raise RuntimeError("this snapshot was not made with SLR_BH")
    note = ""
    if k is None:
        k, note = choose_bh(snap)
        if k is None:
            raise RuntimeError("no BH to show: %s" % note)
    bh = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs", "BH_Mass", "BH_Hsml", "BH_U"])
    box = snap.boxsize or 0.0
    u = snap.units
    rb_all = code_bondi_radius(bh["BH_Mass"], bh["BH_U"], u)
    regions = [bh_region(s, float(bh["BH_Hsml"][j]), float(rb_all[j])) for j in range(bh["count"])]
    R, rmin, rmax = regions[k]

    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Density", "Volume"])
    d = io.periodic_delta(gas["Coordinates"], bh["Coordinates"][k], box)
    r = np.linalg.norm(d, axis=1)
    near = np.where(r < 4.0 * R)[0]
    vol = (gas["Volume"][near] if "Volume" in gas
           else gas["Masses"][near] / np.maximum(gas["Density"][near], 1e-300))
    rcell = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    mass = gas["Masses"][near].astype(np.float64)

    # the smallest target of every BH region a cell is in (blackhole_refinement.cc:249-253)
    target = np.full(near.size, np.inf)
    for j, (Rj, rminj, rmaxj) in enumerate(regions):
        dj = np.linalg.norm(io.periodic_delta(gas["Coordinates"][near], bh["Coordinates"][j], box), axis=1)
        inj = dj < Rj
        target[inj] = np.minimum(target[inj], target_radius(dj[inj], Rj, rminj, rmaxj))

    tmass = s["target_mass"]
    if tmass is None:                      # TargetGasMass not in the parameters: the mass of the cells just outside
        shell = (r[near] > 1.5 * R) & (r[near] < 3.0 * R) & ~np.isfinite(target)
        tmass = float(np.median(mass[shell])) if np.any(shell) else float(np.median(mass))
    st = classify(rcell, mass, target, s, tmass)
    return dict(id=int(bh["ParticleIDs"][k]), k=k, note=note, settings=s, R=R, rmin=rmin, rmax=rmax,
                rb=float(rb_all[k]), hsml=float(bh["BH_Hsml"][k]), bh_mass=float(bh["BH_Mass"][k]),
                pos=np.asarray(bh["Coordinates"][k], dtype=np.float64), idx=near, d=d[near], r=r[near],
                rcell=rcell, mass=mass, target=target, status=st, target_mass=tmass,
                target_mass_given=s["target_mass"] is not None, nbh_regions=len(regions))


def voronoi_slice(d, rcell, half, n=500, normal=2):
    """Voronoi cells on the plane through the BH perpendicular to axis ``normal``:
    each pixel gets the nearest generating point.  Returns (extent, label, rcell map)."""
    from scipy.spatial import cKDTree

    pad = half * np.sqrt(2.0)
    cand = np.where(np.max(np.abs(d), axis=1) < pad + 2.0 * np.max(rcell))[0]
    ax = [a for a in range(3) if a != normal]
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    q = np.zeros((n * n, 3))
    q[:, ax[0]], q[:, ax[1]] = X.ravel(), Y.ravel()
    _, nn = cKDTree(d[cand]).query(q)
    lab = cand[nn].reshape(n, n)
    return (-half, half, -half, half), lab, rcell[lab]


def plot(res, snap, outpath, title=None):
    """The six-panel figure (see the module docstring)."""
    u = snap.units
    pc = u.length_to_kpc(snap.a) * 1e3                     # code length -> physical pc
    msun = u.mass_to_msun
    R, rmin, rmax, rb = res["R"], res["rmin"], res["rmax"], res["rb"]
    r, rc, m, st = res["r"] * pc, res["rcell"] * pc, res["mass"] * msun, res["status"]
    inside = st != OUTSIDE

    fig = plt.figure(figsize=(16.5, 10.5))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.32)

    # (a) cell radius vs distance
    ax = fig.add_subplot(gs[0, 0])
    for code in (OUTSIDE, ON_TARGET, TOO_SMALL, BELOW_FLOOR, AT_FLOOR, TOO_BIG):
        sel = st == code
        if np.any(sel):
            ax.scatter(r[sel], rc[sel], s=3 if code == OUTSIDE else 5, color=STATUS[code][1],
                       alpha=0.5 if code == OUTSIDE else 0.8, lw=0, label=f"{STATUS[code][0]} ({sel.sum()})")
    rr = np.linspace(0, R, 200)
    tg = target_radius(rr, R, rmin, rmax)
    ax.plot(rr * pc, tg * pc, color="k", lw=1.8, label="target (refine above)")
    ax.plot(rr * pc, tg / res["settings"]["lower_c"] * pc, color="k", lw=1, ls="--",
            label=f"target / RefBHLowerFactorC = {res['settings']['lower_c']:g}")
    for (x, nm, ls), yt in zip(((R, "R_ref", "-"), (rb, "R_B", ":"), (res["hsml"], "BH_Hsml", "-.")),
                               (0.97, 0.90, 0.83)):                 # staggered, so close lines stay readable
        ax.axvline(x * pc, color="0.3", ls=ls, lw=1)
        ax.text(x * pc, yt, f" {nm}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="0.3")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(max(np.min(r[r > 0]) * 0.8, 1e-3 * R * pc), 4 * R * pc)
    ax.set_ylim(min(np.min(rc), rmin * pc) / 2, max(np.max(rc), rmax * pc) * 2)
    ax.set_xlabel("distance from the BH [pc]"); ax.set_ylabel("cell radius $(3V/4\\pi)^{1/3}$ [pc]")
    ax.legend(fontsize=7, loc="lower right", markerscale=2)
    label_panel(ax, "(a)")

    # (b) cell mass vs distance
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(r[~inside], m[~inside], s=3, color=STATUS[OUTSIDE][1], alpha=0.5, lw=0)
    ax.scatter(r[inside], m[inside], s=5, c=[STATUS[c][1] for c in st[inside]], alpha=0.8, lw=0)
    tm = res["target_mass"] * msun
    ax.axhspan(0.5 * tm, 2 * tm, color="0.85", zorder=0,
               label="TargetGasMass ×[½, 2]" + ("" if res["target_mass_given"] else " (estimated)"))
    mm = res["settings"]["min_mass"] * msun
    if mm > 0:
        ax.axhline(mm, color="#f4a261", lw=1.2, label="RefBHMinCellMass")
        ax.axhline(mm / 4, color="#6d597a", lw=1, ls="--", label="RefBHMinCellMass / 4")
    ax.axvline(R * pc, color="0.3", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*fig.axes[0].get_xlim())
    ax.set_xlabel("distance from the BH [pc]"); ax.set_ylabel("cell mass [M$_\\odot$]")
    ax.legend(fontsize=7, loc="lower right")
    label_panel(ax, "(b)")

    # (c) fractions per radial bin
    ax = fig.add_subplot(gs[0, 2])
    edges = np.linspace(0, 2 * R, 17)
    mids = 0.5 * (edges[1:] + edges[:-1]) * pc
    bottom = np.zeros(mids.size)
    for code in (ON_TARGET, TOO_BIG, AT_FLOOR, TOO_SMALL, BELOW_FLOOR, OUTSIDE):
        cnt = np.array([np.sum((st == code) & (res["r"] >= a) & (res["r"] < b)) for a, b in zip(edges[:-1], edges[1:])])
        tot = np.array([np.sum((res["r"] >= a) & (res["r"] < b)) for a, b in zip(edges[:-1], edges[1:])])
        frac = np.where(tot > 0, cnt / np.maximum(tot, 1), 0)
        ax.bar(mids, frac, width=np.diff(edges) * pc * 0.95, bottom=bottom, color=STATUS[code][1],
               label=STATUS[code][0])
        bottom += frac
    ax.axvline(R * pc, color="k", lw=1)
    ax.set_xlim(0, 2 * R * pc); ax.set_ylim(0, 1.45); ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("distance from the BH [pc]"); ax.set_ylabel("fraction of cells")
    ax.legend(fontsize=7, loc="upper right", ncol=2)                # above the bars, not on them
    ax.set_title("(c) what the refinement does, per radial bin", fontsize=9, loc="left")

    # (d, e) Voronoi slices
    try:
        for col, half, tag in ((0, 1.3 * R, "(d) slice through the BH"),
                               (1, max(3.0 * rmin, 0.15 * R), "(e) the same, zoomed in")):
            ax = fig.add_subplot(gs[1, col])
            ext, lab, rmap = voronoi_slice(res["d"], res["rcell"], half)
            im = ax.imshow(np.log10(rmap * pc), origin="lower", extent=[e * pc for e in ext], cmap="viridis",
                           interpolation="nearest")
            edge = np.zeros(lab.shape, dtype=bool)
            edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
            edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
            ax.imshow(np.ma.masked_where(~edge, edge), origin="lower", extent=[e * pc for e in ext],
                      cmap="Greys", vmin=0, vmax=1, interpolation="nearest", alpha=0.6)
            for rad, ls in ((R, "-"), (rb, ":")):
                ax.add_patch(plt.Circle((0, 0), rad * pc, fill=False, color="w", ls=ls, lw=1.5))
            ax.plot(0, 0, marker="*", color="red", ms=9, mec="k")
            ax.set_xlim(ext[0] * pc, ext[1] * pc); ax.set_ylim(ext[2] * pc, ext[3] * pc)
            ax.set_xlabel("x [pc]   (plane z = z$_{BH}$)"); ax.set_ylabel("y [pc]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="log$_{10}$ cell radius [pc]")
            label_panel(ax, tag, stroke=True)
    except ImportError:
        fig.add_subplot(gs[1, 0]).text(0.5, 0.5, "slices need scipy", ha="center")

    # (f) the numbers
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    s = res["settings"]
    nin = int(np.sum(inside))
    frac_ok = np.sum(st == ON_TARGET) / max(nin, 1)
    core = res["r"] < rb
    rcore = np.median(res["rcell"][core]) if np.any(core) else np.nan
    lines = [
        f"BH {res['id']}  ({res['note']})" if res["note"] else f"BH {res['id']}",
        f"M_BH = {res['bh_mass'] * msun:.3g} M$_\\odot$,  BH_Hsml = {res['hsml'] * pc:.4g} pc",
        f"SLR_BH = {1 if s['force'] else 0}{'  (SLR_BH_FIXED)' if s['fixed'] else ''}",
        f"R_ref = {R * pc:.4g} pc,  R_B (code formula) = {rb * pc:.4g} pc",
        f"target radius: r_min = {rmin * pc:.4g} pc at the BH -> r_max = {rmax * pc:.4g} pc at R_ref",
        f"RefBHMinCellMass = {s['min_mass'] * msun:.3g} M$_\\odot$,  LowerFactorC = {s['lower_c']:g}",
        f"TargetGasMass = {res['target_mass'] * msun:.3g} M$_\\odot$" + ("" if res["target_mass_given"] else " (estimated)"),
        "",
        f"cells inside R_ref: {nin};  on target {100 * frac_ok:.1f}%",
        f"  too big (should split): {np.sum(st == TOO_BIG)};  at the mass floor: {np.sum(st == AT_FLOOR)}",
        f"  too small (should merge): {np.sum(st == TOO_SMALL)};  below floor/4: {np.sum(st == BELOW_FLOOR)}",
        f"cells inside R_B: {int(np.sum(core))};  median cell radius there {rcore * pc:.3g} pc",
        f"  = R_B / {rb / rcore:.3g}" if np.isfinite(rcore) and rcore > 0 else "",
        f"BH regions in the box: {res['nbh_regions']}",
        "",
        "slices (d, e): black lines = Voronoi cell edges;",
        "white circles: R_ref (solid), R_B (dotted)",
    ]
    if not s["force"]:
        lines += ["", "SLR_BH = 0: a cell is split only when can_this_cell_be_split()",
                  "allows it, which a snapshot cannot show."]
    ax.text(-0.05, 1.0, "\n".join(lines), va="top", fontsize=8.5, family="monospace", transform=ax.transAxes)

    fig.suptitle((title + "  " if title else "") + f"SLR_BH refinement around BH {res['id']}"
                 f"  ({snap.files[0].split('/')[-1]}, t = {snap.time:.4g})", fontsize=13)
    return savefig(fig, outpath)


def slr_mesh_figure(snap, outpath, bh_id=None, title=None, hide_lowres=True):
    """For run_all: the figure for the target galaxy's most massive BH (or ``bh_id``).
    Returns the path, or None when the run has no SLR_BH or no BH."""
    s = slr_settings(snap)
    if s is None:
        return None
    k = None
    if bh_id is not None:
        bh = snap.part(io.PT_BH, ["ParticleIDs"])
        hit = np.where(bh["ParticleIDs"] == bh_id)[0]
        k = int(hit[0]) if hit.size else None
    if k is None and not snap.part(io.PT_BH, ["ParticleIDs"]).get("count"):
        return None
    if k is None:
        k0, note = choose_bh(snap, hide_lowres=hide_lowres)
        if k0 is None:
            print("  [slr_mesh] %s -- skipping" % note)
            return None
    return plot(analyse(snap, k, s), snap, outpath, title=title)


def main(argv=None):
    import argparse
    import os

    ap = argparse.ArgumentParser(description="How well SLR_BH refines the mesh around a BH.")
    ap.add_argument("outdir")
    ap.add_argument("-s", "--snap", type=int, default=None, help="snapshot number (default: the last)")
    ap.add_argument("--bh-id", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from .plotstyle import apply_style

    apply_style()
    nums = io.available_snapshots(a.outdir)
    snap = io.Snapshot(a.outdir, a.snap if a.snap is not None else nums[-1])
    if slr_settings(snap) is None:
        raise SystemExit("this snapshot was not made with SLR_BH")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.outdir)), "slr_mesh_%03d.png" % snap.num)
    path = slr_mesh_figure(snap, out, bh_id=a.bh_id)
    print("  [slr_mesh] wrote %s" % path if path else "  [slr_mesh] no black hole")


if __name__ == "__main__":
    main()
