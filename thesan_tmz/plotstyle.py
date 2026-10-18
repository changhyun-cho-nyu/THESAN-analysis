"""
Shared matplotlib styling, colour maps and small plotting helpers.

Keeping this in one place means every figure the package produces is
immediately recognisable as coming from the same run, and that changing e.g.
the font size for a paper is a one-line edit rather than a sweep through
fifteen scripts.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Use a non-interactive backend by default so the scripts work over ssh and in
# batch jobs.  Call use_interactive() from a notebook if you want windows.
mpl.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: F401,E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: F401,E402

#: A colour-blind-safe qualitative palette (Okabe & Ito).
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
CYCLE = [PALETTE[k] for k in ("blue", "vermillion", "green", "orange", "purple", "skyblue", "grey", "black")]

#: Consistent colours for physical components, used across every figure.
COMPONENT_COLORS = {
    "bh": PALETTE["black"],
    "stars": PALETTE["orange"],
    "gas": PALETTE["blue"],
    "gas_cold": PALETTE["skyblue"],
    "gas_warm": PALETTE["green"],
    "gas_hot": PALETTE["vermillion"],
    "dm": PALETTE["grey"],
    "sfr": PALETTE["purple"],
    "thermal": PALETTE["vermillion"],
    "kinetic": PALETTE["blue"],
}

RC = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.5,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
    "image.cmap": "magma",
    "figure.facecolor": "white",
    "axes.grid": False,
}


def apply_style():
    """Install the package style.  Call once at the top of a script."""
    mpl.rcParams.update(RC)


def use_interactive():
    """Switch to an interactive backend (for notebooks / a local session)."""
    try:
        mpl.use("MacOSX", force=True)
    except Exception:
        try:
            mpl.use("TkAgg", force=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def add_redshift_axis(ax, units, zticks=(20, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0)):
    """Add a top axis labelled in redshift to a plot whose x is age in Gyr.

    Only ticks that land inside the current x range are drawn, so the same
    default tick list works for a z ~ 20 -> 5 zoom and for a z ~ 6 -> 0 run.
    """
    x0, x1 = ax.get_xlim()
    top = ax.twiny()
    top.set_xlim(x0, x1)
    zt = np.asarray([z for z in zticks], dtype=np.float64)
    at = 1.0 / (1.0 + zt)
    tt = units.age_gyr(at)
    keep = (tt >= min(x0, x1)) & (tt <= max(x0, x1))
    top.set_xticks(tt[keep])
    top.set_xticklabels(["%g" % z for z in zt[keep]])
    top.set_xlabel("redshift")
    top.minorticks_off()
    top.tick_params(direction="in")
    return top


def add_age_axis(ax, units, ageticks=None):
    """Add a top axis labelled in age (Gyr) to a plot whose x is redshift."""
    x0, x1 = ax.get_xlim()
    top = ax.twiny()
    top.set_xlim(x0, x1)
    if ageticks is None:
        ageticks = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2, 3, 5, 8, 13])
    a = _invert_age(units, np.asarray(ageticks, dtype=np.float64))
    z = 1.0 / a - 1.0
    keep = (z >= min(x0, x1)) & (z <= max(x0, x1))
    top.set_xticks(z[keep])
    top.set_xticklabels(["%g" % t for t in np.asarray(ageticks)[keep]])
    top.set_xlabel("age of universe [Gyr]")
    top.minorticks_off()
    return top


def _invert_age(units, t_gyr):
    """Invert the flat-LCDM age(a) analytically."""
    from .units import MPC_IN_CM, SEC_PER_YEAR

    H0 = units.h * 100.0 * 1.0e5 / MPC_IN_CM
    pre = 2.0 / (3.0 * H0 * np.sqrt(units.OmegaLambda))
    t_s = np.asarray(t_gyr, dtype=np.float64) * 1.0e9 * SEC_PER_YEAR
    return (np.sinh(t_s / pre) / np.sqrt(units.OmegaLambda / units.Omega0)) ** (2.0 / 3.0)


def logify(ax, which="y"):
    if "x" in which:
        ax.set_xscale("log")
    if "y" in which:
        ax.set_yscale("log")


def label_panel(ax, text, loc="upper left", stroke=False, **kw):
    """Panel label in axes coordinates.

    ``stroke=True`` outlines the glyphs in the opposite colour, which is what
    makes a label readable on top of an image whose background varies from
    black to white across the panel.
    """
    xy = {"upper left": (0.04, 0.95), "upper right": (0.96, 0.95), "lower left": (0.04, 0.06), "lower right": (0.96, 0.06)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    kw.setdefault("fontsize", 9)
    t = ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va, **kw)
    if stroke:
        import matplotlib.patheffects as pe

        edge = "k" if kw.get("color", "k") in ("w", "white") else "w"
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=edge)])
    return t


def shade_band(ax, x, lo, hi, color, alpha=0.18, label=None, zorder=0):
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, label=label, zorder=zorder)


def savefig(fig, path, verbose=True):
    import os

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print("  wrote %s" % path)
    return path


# ---------------------------------------------------------------------------
# Binned statistics without a scipy dependency in the hot path
# ---------------------------------------------------------------------------


def running_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=5):
    """Percentiles of ``y`` in bins of ``x``.

    Bins with fewer than ``min_count`` points return NaN rather than a noisy
    single-point 'median', which otherwise produces very convincing-looking
    but meaningless wiggles at the ends of a relation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    bins = np.asarray(bins, dtype=np.float64)
    centres = 0.5 * (bins[1:] + bins[:-1])
    out = np.full((len(percentiles), centres.size), np.nan)
    counts = np.zeros(centres.size, dtype=int)

    idx = np.digitize(x, bins) - 1
    for i in range(centres.size):
        m = idx == i
        counts[i] = int(m.sum())
        if counts[i] >= min_count:
            out[:, i] = np.percentile(y[m], percentiles)
    return centres, out, counts


def weighted_histogram2d(x, y, weights, xbins, ybins):
    """2D histogram returning (H, xedges, yedges) with H = 0 where empty."""
    H, xe, ye = np.histogram2d(np.asarray(x), np.asarray(y), bins=[xbins, ybins], weights=np.asarray(weights))
    return H.T, xe, ye
