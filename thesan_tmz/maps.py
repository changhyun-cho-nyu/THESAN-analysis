"""
Projections, phase diagrams and maps.

The projections spread each element over its own size (:func:`project_adaptive`)
with the projected cubic-spline kernel of compact support h (the SPH convention):
a gas cell gets h = 2.5 x its radius (3V/4pi)^(1/3), a star particle h = the
distance to its 16th-nearest star.  Large diffuse CGM cells then fill the space
they occupy instead of showing up as isolated pixel-sized squares, the small
cells of the disc keep the fine structure, and nothing spreads beyond h, so
edges stay sharp.  It is not a Voronoi ray-trace, but it conserves mass and is
fast.  :func:`project` (point deposit) remains for callers that have no sizes.

Two frames: the *jet frame* (fig 12; z along the BH's axis, for the outflow
check) and the *galaxy frame* (fig 20; z along the galaxy's own spin, for
face-on and edge-on views of the disc, see :func:`galaxy_axis`).

The most useful view for this branch is the *jet frame*: rotate so the black
hole's angular-momentum axis is vertical, then look edge-on.  A working
bipolar model shows two hot, fast, low-density lobes above and below a dense
equatorial disc; an isotropic one shows a round blob.
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from .plotstyle import COMPONENT_COLORS as C
from .plotstyle import LogNorm, PALETTE, label_panel, plt, savefig

try:
    from scipy.ndimage import gaussian_filter

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def rotation_to_axis(axis):
    """Rotation matrix taking ``axis`` to +z.

    Rows are the new basis vectors, so ``R @ r`` expresses ``r`` in the frame
    where the jet points along z.  Degenerate when ``axis`` is already +-z,
    which is handled by picking an arbitrary perpendicular.
    """
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # Pick any vector not parallel to n.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(tmp, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.vstack([e1, e2, n])


def to_frame(rel, axis=None):
    """Express positions/velocities in the jet frame (z along ``axis``)."""
    if axis is None:
        return rel
    return rel @ rotation_to_axis(axis).T


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project(pos_kpc, weights, extent_kpc, npix=400, depth_kpc=None, plane=(0, 1), normal=2, smooth_px=0.8, values=None):
    """Deposit into a 2D grid, returning (image, extent).

    ``values=None``  -> the image is the summed weight per pixel divided by the
                        pixel area, i.e. a surface density in [weight]/kpc^2.
    ``values`` given -> the image is the weight-weighted mean of ``values``,
                        which is how you make a mass-weighted temperature map.
    """
    x = pos_kpc[:, plane[0]]
    y = pos_kpc[:, plane[1]]
    sel = (np.abs(x) <= extent_kpc) & (np.abs(y) <= extent_kpc)
    if depth_kpc is not None:
        sel &= np.abs(pos_kpc[:, normal]) <= depth_kpc
    if sel.sum() == 0:
        return np.full((npix, npix), np.nan), (-extent_kpc, extent_kpc, -extent_kpc, extent_kpc)

    bins = np.linspace(-extent_kpc, extent_kpc, npix + 1)
    w = np.asarray(weights)[sel]
    H, _, _ = np.histogram2d(x[sel], y[sel], bins=[bins, bins], weights=w)

    if values is None:
        px = (2.0 * extent_kpc / npix) ** 2
        img = H.T / px
    else:
        V, _, _ = np.histogram2d(x[sel], y[sel], bins=[bins, bins], weights=w * np.asarray(values)[sel])
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(H.T > 0, V.T / H.T, np.nan)

    if smooth_px and _HAVE_SCIPY:
        if values is None:
            img = gaussian_filter(img, smooth_px)
        else:
            num = gaussian_filter(np.nan_to_num(V.T), smooth_px)
            den = gaussian_filter(H.T, smooth_px)
            with np.errstate(invalid="ignore", divide="ignore"):
                img = np.where(den > 0, num / den, np.nan)

    return img, (-extent_kpc, extent_kpc, -extent_kpc, extent_kpc)


#: cubic-spline smoothing lengths (compact support h, the Arepo/Gadget convention):
#: a gas cell gets h = GAS_HSML_FAC x its radius (3V/4pi)^(1/3) -- about 16
#: cells inside the kernel sphere; a star gets h = distance to its STAR_KNN-th
#: nearest star
GAS_HSML_FAC = 2.5
STAR_KNN = 16
#: width of the smoothing-length classes in log10 h (the kernel of a class is
#: drawn at the geometric mean h of its members, so h is off by < 7 %)
H_CLASS_DEX = 0.03


def _spline_profile(n=401, nz=2001):
    """Line-of-sight integral of the 3D cubic spline of support 1, against the
    projected radius q = R/h (unnormalised; the stamps are normalised)."""
    q = np.linspace(0.0, 1.0, n)
    z = np.linspace(0.0, 1.0, nz)
    r = np.hypot(q[:, None], z[None, :])
    w = np.where(r < 0.5, 1.0 - 6.0 * r ** 2 + 6.0 * r ** 3, np.where(r < 1.0, 2.0 * (1.0 - r) ** 3, 0.0))
    return q, np.trapezoid(w, z, axis=1)


_SPL_Q, _SPL_F = _spline_profile()


def spline_stamp(h_px, sub=None):
    """Unit-sum image of the projected cubic spline of support ``h_px`` pixels.
    Each pixel is sampled ``sub`` x ``sub`` times so kernels of a few pixels are
    not aliased; by default 4 x 4 below 8 px and once per pixel above (a wide
    kernel is smooth on the pixel scale, and sub-sampling it costs sub^2)."""
    if sub is None:
        sub = 4 if h_px < 8.0 else 1
    r = int(np.ceil(h_px))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    g = (np.arange(-r, r + 1)[:, None] + off[None, :]).ravel()
    R = np.hypot(g[:, None], g[None, :]) / h_px
    k = np.interp(R, _SPL_Q, _SPL_F, right=0.0)
    k = k.reshape(2 * r + 1, sub, 2 * r + 1, sub).sum(axis=(1, 3))
    return k / k.sum()


def project_adaptive(pos_kpc, weights, size_kpc, extent_kpc, npix=400, depth_kpc=None, plane=(0, 1), normal=2,
                     values=None, kernel="spline", max_h_frac=1.0):
    """Deposit each element over its own size.  Same return convention as
    :func:`project`; without scipy it falls back to :func:`project`.

    ``kernel='spline'`` (default): the projected cubic spline of compact
    support ``size_kpc`` = h, the SPH convention -- sharp edges, nothing beyond
    h.  Elements are grouped in classes of h ``H_CLASS_DEX`` wide; each class is
    histogrammed and convolved once with its kernel.  Elements with h below half
    a pixel go straight to their pixel.  h is capped at ``max_h_frac`` x the
    frame width (a huge CGM cell cannot make the image larger).

    ``kernel='gauss'``: a Gaussian of width ``size_kpc`` (the earlier scheme,
    sqrt(2) classes, kept for comparison).

    Elements outside the frame whose kernels reach into it are included, so the
    frame is complete; mass is conserved for elements whose kernel lies inside.
    """
    if not _HAVE_SCIPY:
        return project(pos_kpc, weights, extent_kpc, npix, depth_kpc, plane, normal, values=values)
    from scipy.signal import fftconvolve

    ext_out = (-extent_kpc, extent_kpc, -extent_kpc, extent_kpc)
    px = 2.0 * extent_kpc / npix
    x, y = pos_kpc[:, plane[0]], pos_kpc[:, plane[1]]
    h = np.asarray(size_kpc, dtype=np.float64) * np.ones(len(x))
    if kernel == "spline":
        h_px = np.minimum(h / px, max_h_frac * npix)
        reach = h_px                                          # compact support
    elif kernel == "gauss":
        h_px = np.maximum(h / px, 0.5)
        reach = 3.0 * h_px                                    # gaussian_filter truncate=3
    else:
        raise ValueError("kernel must be 'spline' or 'gauss'")
    sel = (np.abs(x) - reach * px <= extent_kpc) & (np.abs(y) - reach * px <= extent_kpc)
    if depth_kpc is not None:
        sel &= np.abs(pos_kpc[:, normal]) <= depth_kpc
    if not sel.any():
        return np.full((npix, npix), np.nan), ext_out
    x, y, h_px = x[sel], y[sel], h_px[sel]
    w = np.asarray(weights, dtype=np.float64)[sel]
    v = None if values is None else np.asarray(values, dtype=np.float64)[sel]

    P = int(np.ceil(reach[sel].max())) + 1                     # padding, pixels
    n = npix + 2 * P
    lo = -extent_kpc - P * px
    ix = np.floor((x - lo) / px).astype(np.int64)
    iy = np.floor((y - lo) / px).astype(np.int64)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    ix, iy, h_px, w = ix[inside], iy[inside], h_px[inside], w[inside]
    v = v[inside] if v is not None else None
    flat = iy * n + ix                                        # row = y, column = x

    def layer(m, wt):
        return np.bincount(flat[m], weights=wt[m], minlength=n * n).reshape(n, n)

    den = np.zeros((n, n))
    num = np.zeros((n, n)) if v is not None else None
    wv = w * v if v is not None else None
    if kernel == "spline":
        point = h_px < 0.5
        if point.any():
            den += layer(point, w)
            if v is not None:
                num += layer(point, wv)
        big = ~point
        if big.any():
            lh = np.log10(h_px[big])
            cls = np.floor(lh / H_CLASS_DEX).astype(np.int64)
            idx = np.nonzero(big)[0]
            for c in np.unique(cls):
                m = np.zeros(len(h_px), dtype=bool)
                m[idx[cls == c]] = True
                st = spline_stamp(10.0 ** lh[cls == c].mean())          # geometric mean h of the class
                den += fftconvolve(layer(m, w), st, mode="same")
                if v is not None:
                    num += fftconvolve(layer(m, wv), st, mode="same")
        # FFT round-off leaves ~1e-16 x peak (of either sign) where no kernel reaches
        den[den < 1e-12 * den.max()] = 0.0
    else:
        cls = np.floor(2.0 * np.log2(h_px / 0.5)).astype(int)
        for c in np.unique(cls):
            m = cls == c
            s_px = 0.5 * 2.0 ** ((c + 0.5) / 2.0)
            den += gaussian_filter(layer(m, w), s_px, mode="constant", truncate=3.0)
            if v is not None:
                num += gaussian_filter(layer(m, wv), s_px, mode="constant", truncate=3.0)

    den = den[P:P + npix, P:P + npix]
    if v is None:
        return den / px ** 2, ext_out
    num = num[P:P + npix, P:P + npix]
    floor = 1e-6 * den.max() if den.size and den.max() > 0 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > floor, num / den, np.nan), ext_out


def gas_cell_radius_kpc(gas, units, a):
    """Physical radius (3V/4pi)^(1/3) of each gas cell, from Masses/Density."""
    vol = np.asarray(gas["Masses"], dtype=np.float64) / np.maximum(np.asarray(gas["Density"], dtype=np.float64), 1e-300)
    return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) * units.length_to_kpc(a)


def gas_hsml_kpc(gas, units, a):
    """Cubic-spline smoothing length of each gas cell: GAS_HSML_FAC x its radius."""
    return GAS_HSML_FAC * gas_cell_radius_kpc(gas, units, a)


def star_smoothing_kpc(pos_kpc, k=STAR_KNN):
    """Cubic-spline smoothing length of each star: the distance to its k-th
    nearest star (physical kpc), so dense clusters stay compact.  A fixed
    0.1 kpc without scipy."""
    n = len(pos_kpc)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return np.full(n, 0.1)
    if n < 2:
        return np.full(n, 0.1)
    d, _ = cKDTree(pos_kpc).query(pos_kpc, k=min(k + 1, n))
    return np.atleast_2d(d)[:, -1]


def stellar_half_mass_radius_kpc(snap):
    """3D stellar half-mass radius of the target galaxy (Subfind
    ``SubhaloHalfmassRadType[4]``), physical kpc; None without a catalogue."""
    try:
        from . import catalog

        cat = catalog.load(snap)
        j = cat.target_index() if cat is not None else None
        hr = cat.s("halfmassradtype") if j is not None else None
        if hr is not None and np.asarray(hr)[j, io.PT_STAR] > 0:
            return float(np.asarray(hr)[j, io.PT_STAR]) * snap.units.length_to_kpc(snap.a)
    except Exception:
        pass
    return None


#: fig 20 frame: half-width = GALAXY_VIEW_NRH stellar half-mass radii -- the
#: same sphere the spin axis is measured in -- clamped to this range (kpc)
GALAXY_VIEW_NRH = 3.0
GALAXY_VIEW_RANGE_KPC = (0.1, 20.0)


def galaxy_view_extent_kpc(snap, fallback_kpc=20.0):
    """Half-width of the fig 20 frame and a label saying where it came from."""
    rh = stellar_half_mass_radius_kpc(snap)
    if not rh:
        return fallback_kpc, "%.3g kpc (no stellar half-mass radius; --extent)" % fallback_kpc
    lo, hi = GALAXY_VIEW_RANGE_KPC
    e = float(np.clip(GALAXY_VIEW_NRH * rh, lo, hi))
    tag = "" if lo < GALAXY_VIEW_NRH * rh < hi else ", clamped to %g-%g kpc" % (lo, hi)
    return e, "%.3g kpc = %g r$_{1/2,\\star}$ (r$_{1/2,\\star}$ = %.3g kpc%s)" % (e, GALAXY_VIEW_NRH, rh, tag)


def galaxy_axis(snap, centre, radius_kpc=None, coherence_min=0.2):
    """The galaxy's own spin axis, for face-on / edge-on views.

    J = sum m (r x v) of the **stars** within ``radius_kpc`` (default: three
    stellar half-mass radii of the target galaxy from Subfind, else 10 kpc).
    If the stars do not rotate coherently -- |sum m r x v| / sum m |r x v| below
    ``coherence_min``, a dispersion-dominated body with no disc -- the cold gas
    (T < 3e4 K) is used instead.  Returns ``(axis, info)``; axis None when
    neither works.
    """
    u, a = snap.units, snap.a
    L = u.length_to_kpc(a)
    if radius_kpc is None:
        rh = stellar_half_mass_radius_kpc(snap)
        radius_kpc = 3.0 * rh if rh else 10.0

    def spin(pt, extra=None):
        d = snap.part(pt, ["Coordinates", "Masses", "Velocities"] + (extra or []))
        if not d.get("count") or "Velocities" not in d:
            return None, 0.0, 0
        rel = io.periodic_delta(d["Coordinates"], centre, snap.boxsize) * L
        sel = np.linalg.norm(rel, axis=1) <= radius_kpc
        if pt == io.PT_GAS and "InternalEnergy" in d:
            T = u.temperature(d["InternalEnergy"], d.get("ElectronAbundance"))
            sel &= T < 3e4
        if sel.sum() < 20:
            return None, 0.0, int(sel.sum())
        m = d["Masses"][sel]
        v = d["Velocities"][sel] - np.average(d["Velocities"][sel], axis=0, weights=m)
        jj = np.cross(rel[sel], v) * m[:, None]
        J = jj.sum(axis=0)
        coh = float(np.linalg.norm(J) / max(np.linalg.norm(jj, axis=1).sum(), 1e-300))
        return (J / np.linalg.norm(J) if np.linalg.norm(J) > 0 else None), coh, int(sel.sum())

    ax_s, coh_s, n_s = spin(io.PT_STAR)
    if ax_s is not None and coh_s >= coherence_min:
        return ax_s, {"from": "stars", "coherence": coh_s, "n": n_s, "radius_kpc": radius_kpc}
    ax_g, coh_g, n_g = spin(io.PT_GAS, ["InternalEnergy", "ElectronAbundance"])
    if ax_g is not None and coh_g >= coherence_min:
        return ax_g, {"from": "cold gas", "coherence": coh_g, "n": n_g, "radius_kpc": radius_kpc,
                      "stars_coherence": coh_s}
    if ax_s is not None:
        return ax_s, {"from": "stars (weak rotation)", "coherence": coh_s, "n": n_s, "radius_kpc": radius_kpc}
    return None, {"from": None, "radius_kpc": radius_kpc}


#: fig 20 young-star row: stars younger than this
YOUNG_MYR = 10.0


def galaxy_view_rows(snap, centre, extent_kpc, R=None, young_myr=YOUNG_MYR, spectra_dir=None, uv_recipe="point"):
    """The rows of fig 20 as ``(label, positions, weights, h, colour-bar label)``
    in the frame rotated by ``R``, plus notes on rows left out."""
    from . import uv

    u, a = snap.units, snap.a
    L = u.length_to_kpc(a)
    R = np.eye(3) if R is None else R
    sig_star = r"$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]"
    rows, notes = [], []          # rows: (label, positions, weights, h, colour-bar label)
    st = snap.part(io.PT_STAR, uv.stars_with_uv_fields())
    if st.get("count"):
        rel = io.periodic_delta(st["Coordinates"], centre, snap.boxsize) * L
        # the frame is a cube of half-width extent (any orientation: within
        # sqrt(3) extent); the margin keeps the k-NN distances right at its edge
        near = np.linalg.norm(rel, axis=1) <= 3.0 * extent_kpc
        if "GFM_StellarFormationTime" in st:
            tform = np.asarray(st["GFM_StellarFormationTime"], dtype=np.float64)
            near &= tform > 0                                   # wind particles are not stars
        rel = rel[near] @ R.T
        mst = st["Masses"][near] * u.mass_to_msun
        h_all = star_smoothing_kpc(rel)
        rows.append(("all stars", rel, mst, h_all, sig_star))

        if "GFM_StellarFormationTime" in st:
            age_myr = (snap.age_gyr - u.age_gyr(tform[near])) * 1e3
            yng = age_myr < young_myr
            if yng.sum() > STAR_KNN:
                rows.append(("stars < %g Myr" % young_myr, rel[yng], mst[yng], star_smoothing_kpc(rel[yng]), sig_star))
            else:
                notes.append("%d stars < %g Myr: young row left out" % (int(yng.sum()), young_myr))

            lnu, why = uv.star_l_nu(snap, {k: np.asarray(v)[near] for k, v in st.items()
                                               if k not in ("count",) and np.ndim(v) > 0 and len(v) == len(near)},
                                        spectra_dir, recipe=uv_recipe)
            if lnu is not None and np.any(lnu > 0):
                rows.append(("intrinsic UV 1500 $\\AA$", rel, lnu, h_all,
                             r"$\Sigma_{\rm UV}$ [erg s$^{-1}$ Hz$^{-1}$ kpc$^{-2}$]"))
                notes += ["UV: " + n.strip() for n in why.split(";") if "get no UV" in n]   # a skipped table
                notes.append("UV: " + ("L_nu at 1500 A" if uv_recipe == "point" else "1490-1510 A average (AREPO)"))
            else:
                notes.append("UV row left out: %s" % (why if lnu is None else "no star emits"))
        else:
            notes.append("no GFM_StellarFormationTime: young and UV rows left out")
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Density"])
    if gas.get("count"):
        rel = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
        hgas = gas_hsml_kpc(gas, u, a)
        near = np.linalg.norm(rel, axis=1) - hgas <= np.sqrt(3.0) * extent_kpc   # kernel reaches the cube
        rows.append(("gas", rel[near] @ R.T, gas["Masses"][near] * u.mass_to_msun, hgas[near],
                     r"$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]"))
    return rows, notes


def plot_galaxy_view(snap, centre, outpath, extent_kpc=None, npix=400, title=None, axis=None, fallback_kpc=20.0,
                     young_myr=YOUNG_MYR, spectra_dir=None, uv_recipe="point"):
    """Fig 20: the galaxy face-on (left) and edge-on (right) in the frame of its
    own spin (:func:`galaxy_axis`).  Rows: all stars (mass), stars younger than
    ``young_myr`` (mass), all stars' intrinsic UV at 1500 A (:mod:`uv`, the
    run's own spectrum tables), gas (mass).  Each element is spread with the
    cubic-spline kernel over its own size; for the young row the k-NN distance is
    taken among the young stars only.  Empty or very faint pixels are white.
    A row that cannot be made (no young stars, no spectrum tables) is left out
    and the reason is given in the title.

    ``extent_kpc`` is the half-width of the frame (the slab is as deep as it is
    wide).  None (default) scales it to the galaxy: :func:`galaxy_view_extent_kpc`,
    3 stellar half-mass radii.  At z ~ 10 that is a few hundred pc, where a
    fixed 20 kpc would show the whole halo."""
    if extent_kpc is None:
        extent_kpc, scale = galaxy_view_extent_kpc(snap, fallback_kpc)
    else:
        scale = "%.3g kpc (set by hand)" % extent_kpc
    info = {"from": "given"}
    if axis is None:
        axis, info = galaxy_axis(snap, centre)
    R = rotation_to_axis(axis) if axis is not None else np.eye(3)
    frame = ("galaxy frame, spin from %s within %.3g kpc" % (info["from"], info.get("radius_kpc", np.nan))
             if axis is not None else "simulation frame (no galaxy spin found)")

    rows, notes = galaxy_view_rows(snap, centre, extent_kpc, R, young_myr, spectra_dir, uv_recipe)
    if not rows:
        return None

    bg, fg = "black", "white"
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(bg)
    cmap.set_under(bg)
    unit, f = ("pc", 1e3) if extent_kpc < 1.0 else ("kpc", 1.0)
    ext = (-f * extent_kpc, f * extent_kpc, -f * extent_kpc, f * extent_kpc)
    bar = scale_bar_length(2.0 * f * extent_kpc)
    nrow = len(rows)
    fig, ax = plt.subplots(nrow, 2, figsize=(8.0, 4.0 * nrow + 0.7), squeeze=False, facecolor=bg,
                           gridspec_kw=dict(wspace=0.015, hspace=0.015, left=0.01, right=0.99, bottom=0.01,
                                            top=1.0 - 0.7 / (4.0 * nrow + 0.7)))
    for i, (name, rel, w, h, cbl) in enumerate(rows):
        imgs = [project_adaptive(rel, w, h, extent_kpc, npix, depth_kpc=extent_kpc, plane=pl, normal=nm)[0]
                for pl, nm in (((0, 1), 2), ((0, 2), 1))]
        if unit == "pc":                                     # surface densities per pc^2 on pc-sized frames
            imgs = [im * 1e-6 for im in imgs]
            cbl = cbl.replace("kpc$^{-2}$", "pc$^{-2}$")
        vmax = max(np.nanpercentile(im[im > 0], 99.9) if (im > 0).any() else 1.0 for im in imgs)
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        for j, (img, vname) in enumerate(zip(imgs, ("face-on", "edge-on"))):
            im = ax[i, j].imshow(np.where(img > 0, img, np.nan), origin="lower", extent=ext, norm=norm, cmap=cmap)
            dark_panel(ax[i, j], "%s, %s" % (name, vname), bar, unit, ext, spin_arrow=(j == 1 and axis is not None),
                       bg=bg, fg=fg)
        inset_colorbar(fig, ax[i, 0], im, cbl, color=fg)
    head = "%s   z = %.2f   (%s)\nhalf-width %s%s" % (title or "", snap.z, frame, scale, "".join("\n" + n for n in notes))
    fig.suptitle(head, y=0.998, fontsize=8, color=fg, va="top")
    return savefig(fig, outpath, facecolor=bg)


def dark_panel(ax, label, bar, unit, extent, spin_arrow=False, bg="black", fg="white"):
    """Map panel without axes: background, label (lower right), scale bar (lower
    left) and, for edge-on views, an arrow along the spin axis (up)."""
    ax.set_facecolor(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    label_panel(ax, label, loc="lower right", color=fg, stroke=True)
    draw_scale_bar(ax, bar, unit, extent, color=fg)
    if spin_arrow:
        ax.annotate("", xy=(0.94, 0.80), xytext=(0.94, 0.62), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=fg, lw=1.2))
        ax.text(0.915, 0.71, r"$\vec J$", transform=ax.transAxes, color=fg, ha="right", va="center", fontsize=9)


def uv_overlay_rgba(uv_img, top, dex, cmap):
    """RGBA image of a UV map for drawing over the gas: colour and opacity both
    rise linearly in log Sigma_UV from ``top - dex`` (transparent) to ``top``
    (opaque).  Returns (rgba, log10 map with empty pixels set below the range)."""
    with np.errstate(divide="ignore"):
        lv = np.where(uv_img > 0, np.log10(np.where(uv_img > 0, uv_img, 1.0)), top - dex - 1.0)
    x = np.clip((lv - (top - dex)) / dex, 0.0, 1.0)
    rgba = cmap(x)
    rgba[..., 3] = x
    return rgba, lv


def plot_gas_uv(snap, centre, outpath, extent_kpc=None, npix=400, title=None, axis=None, fallback_kpc=20.0,
                spectra_dir=None, gas_dex=3.0, uv_dex=4.0, uv_recipe="point"):
    """Fig 21: gas surface density in grey with the stars' intrinsic UV (1500 A)
    on top, face-on and edge-on in the galaxy frame, same frame and kernels as
    fig 20.  The UV colour and opacity rise with log Sigma_UV over the brightest
    ``uv_dex`` decades (fainter UV is transparent, so the gas shows through);
    thin contours mark each decade.  The gas shows ``gas_dex`` decades.
    Returns None (with a printed reason) when the run has no UV."""
    if extent_kpc is None:
        extent_kpc, scale = galaxy_view_extent_kpc(snap, fallback_kpc)
    else:
        scale = "%.3g kpc (set by hand)" % extent_kpc
    info = {"from": "given"}
    if axis is None:
        axis, info = galaxy_axis(snap, centre)
    R = rotation_to_axis(axis) if axis is not None else np.eye(3)
    frame = ("galaxy frame, spin from %s within %.3g kpc" % (info["from"], info.get("radius_kpc", np.nan))
             if axis is not None else "simulation frame (no galaxy spin found)")
    rows, notes = galaxy_view_rows(snap, centre, extent_kpc, R, young_myr=0.0, spectra_dir=spectra_dir,
                                   uv_recipe=uv_recipe)
    byname = {r[0]: r for r in rows}
    gas = byname.get("gas")
    uvr = next((r for r in rows if "UV" in r[0]), None)
    if gas is None or uvr is None:
        print("  [maps] gas + UV overlay skipped: %s" % ("no gas" if gas is None else "; ".join(
            n for n in notes if "UV" in n) or "no UV"))
        return None

    bg, fg = "black", "white"
    unit, f = ("pc", 1e3) if extent_kpc < 1.0 else ("kpc", 1.0)
    per = 1e-6 if unit == "pc" else 1.0
    area = "pc$^{-2}$" if unit == "pc" else "kpc$^{-2}$"
    ext = (-f * extent_kpc, f * extent_kpc, -f * extent_kpc, f * extent_kpc)
    bar = scale_bar_length(2.0 * f * extent_kpc)
    views = (((0, 1), 2, "face-on"), ((0, 2), 1, "edge-on"))
    gimg = [project_adaptive(gas[1], gas[2], gas[3], extent_kpc, npix, depth_kpc=extent_kpc, plane=pl, normal=nm)[0] * per
            for pl, nm, _ in views]
    uimg = [project_adaptive(uvr[1], uvr[2], uvr[3], extent_kpc, npix, depth_kpc=extent_kpc, plane=pl, normal=nm)[0] * per
            for pl, nm, _ in views]
    gtop = max(np.nanpercentile(g[g > 0], 99.9) if (g > 0).any() else 1.0 for g in gimg)
    utop = np.log10(max(np.nanpercentile(u_[u_ > 0], 99.9) if (u_ > 0).any() else 1.0 for u_ in uimg))
    gnorm = LogNorm(vmin=gtop * 10 ** -gas_dex, vmax=gtop)
    grey = plt.get_cmap("gray").copy()
    grey.set_bad(bg)
    grey.set_under(bg)
    warm = plt.get_cmap("inferno")
    levels = np.arange(np.ceil(utop - uv_dex), utop + 1e-9, 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 4.7), squeeze=False, facecolor=bg,
                           gridspec_kw=dict(wspace=0.015, left=0.01, right=0.99, bottom=0.01, top=0.85))
    for j, (g, u_, (_, _, vname)) in enumerate(zip(gimg, uimg, views)):
        a_ = ax[0, j]
        gi = a_.imshow(np.where(g > 0, g, np.nan), origin="lower", extent=ext, norm=gnorm, cmap=grey)
        rgba, lv = uv_overlay_rgba(u_, utop, uv_dex, warm)
        a_.imshow(rgba, origin="lower", extent=ext)
        if len(levels):
            a_.contour(lv, levels=levels, extent=ext, origin="lower", colors=[warm(0.85)], linewidths=0.35, alpha=0.7)
        dark_panel(a_, "gas + UV, %s" % vname, bar, unit, ext, spin_arrow=(j == 1 and axis is not None), bg=bg, fg=fg)
    inset_colorbar(fig, ax[0, 0], gi, r"$\Sigma_{\rm gas}$ [M$_\odot$ %s]" % area, color=fg)
    um = plt.cm.ScalarMappable(norm=LogNorm(vmin=10 ** (utop - uv_dex), vmax=10 ** utop), cmap=warm)
    inset_colorbar(fig, ax[0, 0], um, r"$\Sigma_{\rm UV,1500}$ [erg s$^{-1}$ Hz$^{-1}$ %s]" % area,
                   color=fg, bounds=(0.05, 0.75, 0.45, 0.028))
    head = ("%s   z = %.2f   (%s)\nhalf-width %s;  intrinsic UV, no dust;  UV contours: one per dex%s"
            % (title or "", snap.z, frame, scale, "".join("\n" + n for n in notes if n.startswith("UV: "))))
    fig.suptitle(head, y=0.995, fontsize=8, color=fg, va="top")
    return savefig(fig, outpath, facecolor=bg)


def scale_bar_length(width):
    """A round bar length (1, 2 or 5 x 10^n) of at most a quarter of ``width``."""
    if not width > 0:
        return 1.0
    top = 0.25 * width
    p = 10.0 ** np.floor(np.log10(top))
    return max(m * p for m in (1.0, 2.0, 5.0) if m * p <= top)


def draw_scale_bar(ax, length, unit, extent, color="white"):
    """Horizontal bar of ``length`` (data units) in the lower-left corner, labelled above."""
    x0 = extent[0] + 0.06 * (extent[1] - extent[0])
    y0 = extent[2] + 0.07 * (extent[3] - extent[2])
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.0, solid_capstyle="butt")
    ax.text(x0 + 0.5 * length, y0 + 0.02 * (extent[3] - extent[2]), "%g %s" % (length, unit),
            color=color, ha="center", va="bottom", fontsize=9)


def inset_colorbar(fig, ax, im, label, color="white", bounds=(0.05, 0.89, 0.45, 0.028)):
    """Horizontal colour bar inside ``ax`` (default top left; ``bounds`` in axes
    fraction), label and ticks above it."""
    cax = ax.inset_axes(list(bounds))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(label, color=color, fontsize=8, labelpad=2)
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(colors=color, labelsize=7, which="both")
    cb.outline.set_edgecolor(color)
    return cb


def plot_projection_grid(snap, centre, outpath, extent_kpc=20.0, npix=400, axis=None, title=None, depth_kpc=None):
    """Six-panel view of the gas around the black hole.

    Top row: face-on (looking down the jet axis) column density, temperature,
    radial velocity.  Bottom row: the same, edge-on.  With ``axis=None`` the
    simulation's own xyz frame is used instead.
    """
    u = snap.units
    L = u.length_to_kpc(snap.a)
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "Density", "InternalEnergy", "ElectronAbundance"])
    if gas.get("count", 0) == 0:
        return None

    rel = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
    r = np.linalg.norm(rel, axis=1)
    hgas = gas_hsml_kpc(gas, u, snap.a)
    inner = r <= extent_kpc * 2.0
    near = r - hgas <= extent_kpc * 2.0          # every cell whose kernel reaches the 2 x extent sphere
    vc = np.average(gas["Velocities"][inner], axis=0, weights=gas["Masses"][inner])
    rel, hgas = rel[near], hgas[near]
    m = gas["Masses"][near] * u.mass_to_msun
    T = u.temperature(gas["InternalEnergy"][near], gas["ElectronAbundance"][near] if "ElectronAbundance" in gas else None)

    # peculiar velocity plus the Hubble flow H(a) r (rel is in physical kpc)
    dv = (gas["Velocities"][near] - vc) * u.peculiar_velocity_to_kms(snap.a) + u.hubble_kms_per_kpc(snap.a) * rel
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = np.sum(rel * dv, axis=1) / np.maximum(np.linalg.norm(rel, axis=1), 1e-6)

    if axis is not None:
        R = rotation_to_axis(axis)
        rel = rel @ R.T
        dv = dv @ R.T
        frame = "jet frame"
    else:
        frame = "simulation frame"

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6))

    views = [
        ("face-on (looking down the axis)", (0, 1), 2),
        ("edge-on", (0, 2), 1),
    ]

    for row, (vname, plane, normal) in enumerate(views):
        img, ext = project_adaptive(rel, m, hgas, extent_kpc, npix, depth_kpc, plane, normal)
        a = ax[row, 0]
        finite = img[np.isfinite(img) & (img > 0)]
        vmin = np.percentile(finite, 5) if finite.size else 1.0
        vmax = np.percentile(finite, 99.9) if finite.size else 10.0
        im = a.imshow(img, origin="lower", extent=ext, norm=LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax), cmap="magma")
        fig.colorbar(im, ax=a, label=r"$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]")
        label_panel(a, "column density", color="w", stroke=True)

        img, ext = project_adaptive(rel, m, hgas, extent_kpc, npix, depth_kpc, plane, normal,
                                    values=np.log10(np.maximum(T, 1.0)))
        a = ax[row, 1]
        im = a.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=3.0, vmax=7.5)
        fig.colorbar(im, ax=a, label=r"$\langle\log_{10}T\rangle$ [K]")
        label_panel(a, "temperature", color="w", stroke=True)

        img, ext = project_adaptive(rel, m, hgas, extent_kpc, npix, depth_kpc, plane, normal, values=vr)
        a = ax[row, 2]
        lim = np.nanpercentile(np.abs(img), 98) if np.isfinite(img).any() else 100.0
        im = a.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=a, label=r"$\langle v_r\rangle$ [km s$^{-1}$]")
        label_panel(a, "radial velocity", color="k", stroke=True)

    # Row identity (face-on / edge-on) goes on the leftmost y-axis label
    # instead of prefixing every one of the three panel titles -- at fontsize
    # 9 the full "face-on (looking down the axis): column density" string ran
    # past the right edge of a ~4.5in-wide panel.
    for row, (vname, _, _) in enumerate(views):
        ax[row, 0].set_ylabel("%s\nkpc" % vname, fontsize=8)
    for a in ax.ravel():
        a.set_xlabel("kpc")
        if not a.get_ylabel():
            a.set_ylabel("kpc")
        a.plot(0, 0, "+", color="w", ms=8, mew=1.2)

    fig.suptitle("%s   z = %.2f   (%s)" % (title or "", snap.z, frame), y=0.995)
    fig.tight_layout()
    return savefig(fig, outpath)


def plot_phase_diagram(snap, centre=None, outpath=None, radius_kpc=None, title=None):
    """Density-temperature phase diagram, mass-weighted.

    The equation-of-state branch (a tight power law at high density) is the
    SMUGGLE/GFM effective EoS, not real thermodynamics -- see the note in
    :func:`galaxy.gas_phase_masks`.  A feedback model that works shows a
    populated hot, low-density branch that is *absent* from a no-feedback run.
    """
    u = snap.units
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Density", "InternalEnergy", "ElectronAbundance", "StarFormationRate"])
    if gas.get("count", 0) == 0:
        return None

    sel = np.ones(gas["Masses"].shape, dtype=bool)
    if centre is not None and radius_kpc:
        r = io.radius_from(gas["Coordinates"], centre, snap.boxsize) * u.length_to_kpc(snap.a)
        sel = r <= radius_kpc

    nH = gas["Density"][sel] * u.density_to_nH_cgs(snap.a)
    T = u.temperature(gas["InternalEnergy"][sel], gas["ElectronAbundance"][sel] if "ElectronAbundance" in gas else None)
    m = gas["Masses"][sel] * u.mass_to_msun
    good = (nH > 0) & (T > 0)

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))

    a = ax[0]
    h = a.hexbin(np.log10(nH[good]), np.log10(T[good]), C=m[good], reduce_C_function=np.sum, bins="log", gridsize=70, cmap="magma")
    fig.colorbar(h, ax=a, label=r"$\log_{10}$ mass [M$_\odot$]")
    from .galaxy import NH_DENSE, T_COLD, T_WARM

    a.axhline(np.log10(T_COLD), color="w", ls="--", lw=0.8)
    a.axhline(np.log10(T_WARM), color="w", ls=":", lw=0.8)
    a.axvline(np.log10(NH_DENSE), color="w", ls="-.", lw=0.8)
    a.set_xlabel(r"$\log_{10} n_{\rm H}$ [cm$^{-3}$]")
    a.set_ylabel(r"$\log_{10} T$ [K]")
    label_panel(a, "(a) phase diagram", color="w", stroke=True)

    a = ax[1]
    if "StarFormationRate" in gas:
        sf = gas["StarFormationRate"][sel][good]
        m2 = sf > 0
        if m2.any():
            a.hexbin(np.log10(nH[good][m2]), np.log10(T[good][m2]), C=sf[m2], reduce_C_function=np.sum, bins="log", gridsize=60, cmap="viridis")
            a.set_xlabel(r"$\log_{10} n_{\rm H}$ [cm$^{-3}$]")
            a.set_ylabel(r"$\log_{10} T$ [K]")
            label_panel(a, "(b) star-forming cells", color="w", stroke=True)
        else:
            a.text(0.5, 0.5, "no star-forming cells\nin the aperture", ha="center", va="center", transform=a.transAxes)
    fig.suptitle("%s   z = %.2f" % (title or "", snap.z), y=1.0)
    fig.tight_layout()
    return savefig(fig, outpath)


def softening_kpc(snap, parttypes=(1, 4)):
    """Largest Plummer-equivalent gravitational softening (physical kpc) among
    ``parttypes`` (default DM and stars), from the snapshot's /Parameters:
    eps = min(SofteningComovingType_k * a, SofteningMaxPhysType_k) with
    k = SofteningTypeOfPartType_i.  None if the parameters are not there."""
    p = getattr(snap, "parameters", None) or {}
    eps = []
    for i in parttypes:
        k = p.get("SofteningTypeOfPartType%d" % i)
        if k is None:
            continue
        k = int(k)
        com = p.get("SofteningComovingType%d" % k)
        if com is None:
            continue
        e = float(com) * snap.a
        mx = p.get("SofteningMaxPhysType%d" % k)
        if mx is not None and float(mx) > 0:
            e = min(e, float(mx))
        eps.append(e * snap.units.length_to_kpc(1.0))   # physical code length -> kpc
    return max(eps) if eps else None


def plot_radial_profiles(snap, centre, outpath, rmax_kpc=50.0, title=None):
    """Radial profiles of density, temperature, radial velocity and enclosed mass."""
    from . import galaxy

    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.0))
    bins = np.logspace(-1.5, np.log10(rmax_kpc), 40)

    p = galaxy.radial_profile(snap, centre, io.PT_GAS, "density", bins=bins)
    ax[0, 0].plot(p["r"], p["value"], color=C["gas"], label="all gas")
    for ph, col in (("cold", C["gas_cold"]), ("hot", C["gas_hot"])):
        q = galaxy.radial_profile(snap, centre, io.PT_GAS, "density", bins=bins, phase=ph)
        ax[0, 0].plot(q["r"], q["value"], color=col, ls="--", lw=1.0, label=ph)
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel(r"$n_{\rm H}$ [cm$^{-3}$]")
    ax[0, 0].legend(fontsize=7)
    label_panel(ax[0, 0], "(a) gas density")

    p = galaxy.radial_profile(snap, centre, io.PT_GAS, "temp", bins=bins)
    ax[0, 1].plot(p["r"], p["value"], color=C["gas_hot"])
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel(r"$T$ [K]")
    label_panel(ax[0, 1], "(b) temperature")

    p = galaxy.radial_profile(snap, centre, io.PT_GAS, "vrad", bins=bins)
    ax[1, 0].plot(p["r"], p["value"], color=PALETTE["blue"])
    ax[1, 0].axhline(0, color=PALETTE["grey"], lw=0.8)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_ylabel(r"$v_r$ [km s$^{-1}$]  (+ = outflow)")
    ax[1, 0].set_xlabel("r [kpc]")
    label_panel(ax[1, 0], "(c) radial velocity")

    # A rotation curve is read on a linear radius axis: the flat part and the
    # DM contribution live at 5-50 kpc, and on a log axis half the panel would
    # be the unresolved centre.
    rc = galaxy.rotation_curve(snap, centre, rmax_kpc=rmax_kpc, nbins=120, log=False)
    # The BH is plotted too: it dominates inside its sphere of influence, and
    # seeing where that is makes the inner rotation curve interpretable.
    for k, col in (("total", PALETTE["black"]), ("dm", C["dm"]), ("stars", C["stars"]), ("gas", C["gas"]), ("bh", C["bh"])):
        ax[1, 1].plot(rc["r"], rc[k], color=col, label=k, lw=1.4 if k == "total" else 1.0, ls="-" if k != "bh" else ":")
    ax[1, 1].set_ylabel(r"$v_{\rm circ}$ [km s$^{-1}$]")
    ax[1, 1].set_xlabel("r [kpc]")
    ax[1, 1].legend(fontsize=7)
    label_panel(ax[1, 1], "(d) circular velocity")

    for a in ax.ravel():
        a.set_xlim(bins[0], bins[-1])
    ax[1, 1].set_xlim(0.0, rmax_kpc)
    eps = softening_kpc(snap)
    if eps:
        # inside ~2.8 eps (the spline softening length) gravity is softened:
        # the curve there says nothing about the mass distribution
        ax[1, 1].axvspan(0.0, 2.8 * eps, color=PALETTE["grey"], alpha=0.25, lw=0,
                         label=r"$< 2.8\,\epsilon$ (%.2g kpc), unresolved" % (2.8 * eps))
        ax[1, 1].legend(fontsize=7)
    fig.suptitle("%s   z = %.2f" % (title or "", snap.z), y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)
