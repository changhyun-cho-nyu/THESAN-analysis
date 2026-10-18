"""
Projections, phase diagrams and maps.

The projections here deposit each Voronoi cell's mass into the pixel its
generating point falls in, then optionally smooth.  That is not a proper
Voronoi ray-trace -- it under-resolves large low-density cells and will make
the diffuse CGM look patchier than it is.  For a figure that has to be right in
the diffuse gas, ray-trace the mesh instead.  For seeing whether the feedback
carved a bipolar cavity, this is entirely adequate and about a thousand times
faster.

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
    near = r <= extent_kpc * 2.0
    rel = rel[near]
    m = gas["Masses"][near] * u.mass_to_msun
    T = u.temperature(gas["InternalEnergy"][near], gas["ElectronAbundance"][near] if "ElectronAbundance" in gas else None)

    vc = np.average(gas["Velocities"][near], axis=0, weights=gas["Masses"][near])
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
        img, ext = project(rel, m, extent_kpc, npix, depth_kpc, plane, normal)
        a = ax[row, 0]
        finite = img[np.isfinite(img) & (img > 0)]
        vmin = np.percentile(finite, 5) if finite.size else 1.0
        vmax = np.percentile(finite, 99.9) if finite.size else 10.0
        im = a.imshow(img, origin="lower", extent=ext, norm=LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax), cmap="magma")
        fig.colorbar(im, ax=a, label=r"$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]")
        label_panel(a, "column density", color="w", stroke=True)

        img, ext = project(rel, m, extent_kpc, npix, depth_kpc, plane, normal, values=np.log10(np.maximum(T, 1.0)))
        a = ax[row, 1]
        im = a.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=3.0, vmax=7.5)
        fig.colorbar(im, ax=a, label=r"$\langle\log_{10}T\rangle$ [K]")
        label_panel(a, "temperature", color="w", stroke=True)

        img, ext = project(rel, m, extent_kpc, npix, depth_kpc, plane, normal, values=vr)
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

    rc = galaxy.rotation_curve(snap, centre, rmax_kpc=rmax_kpc)
    # The BH is plotted too: it dominates inside its sphere of influence, and
    # seeing where that is makes the inner rotation curve interpretable.
    for k, col in (("total", PALETTE["black"]), ("dm", C["dm"]), ("stars", C["stars"]), ("gas", C["gas"]), ("bh", C["bh"])):
        ax[1, 1].plot(rc["r"], rc[k], color=col, label=k, lw=1.4 if k == "total" else 1.0, ls="-" if k != "bh" else ":")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_ylabel(r"$v_{\rm circ}$ [km s$^{-1}$]")
    ax[1, 1].set_xlabel("r [kpc]")
    ax[1, 1].legend(fontsize=7)
    label_panel(ax[1, 1], "(d) circular velocity")

    for a in ax.ravel():
        a.set_xlim(bins[0], bins[-1])
    fig.suptitle("%s   z = %.2f" % (title or "", snap.z), y=0.99)
    fig.tight_layout()
    return savefig(fig, outpath)
