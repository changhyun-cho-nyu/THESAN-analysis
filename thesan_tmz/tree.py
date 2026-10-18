"""
The THESAN-Zoom merger tree, ``<run>/output/tree.hdf5``.

Layout (as used by the THESAN-Zoom analysis scripts):

    /Redshifts        (N,)   redshift of each entry, high z first
    /Snapshots        (N,)   snapshot number of each entry
    /Subhalo/<field>  (N,..) the main-branch subhalo's Subfind properties
    /Group/<field>    (N,..) its host FoF group's properties

One row per snapshot along the *main branch* of the target halo.  That is what
makes it worth reading: it has already answered "which subhalo is the
progenitor" across every merger, which is the question a snapshot-by-snapshot
walk gets wrong exactly when it matters most.

Use it for the growth history.  Use :mod:`thesan_tmz.catalog` when you need
every galaxy in a snapshot rather than just the main one, and the snapshot
particles when you need what is happening inside one.
"""

from __future__ import annotations

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("thesan_tmz needs h5py: pip install h5py") from exc

from . import arepo_io as io
from .catalog import GRP_ALIASES, SUB_ALIASES, field


class Tree:
    """Main-branch history of one zoom target.

    >>> t = Tree.open('/scratch/g5760/z4/output')
    >>> t.z, t.mstar_msun()
    """

    def __init__(self, path, units=None):
        self.path = path
        self.sub, self.grp = {}, {}
        with h5py.File(path, "r") as f:
            self.z = np.asarray(f["Redshifts"]) if "Redshifts" in f else None
            self.snapnum = np.asarray(f["Snapshots"]) if "Snapshots" in f else None
            if "Subhalo" in f:
                self.sub = {k: np.asarray(f["Subhalo"][k]) for k in f["Subhalo"]}
            if "Group" in f:
                self.grp = {k: np.asarray(f["Group"][k]) for k in f["Group"]}
        if self.z is None and self.snapnum is not None:
            self.z = np.full(len(self.snapnum), np.nan)
        self.a = 1.0 / (1.0 + self.z) if self.z is not None else None
        self.units = units

    @classmethod
    def open(cls, output_dir, units=None):
        """Return the tree for a run, or None if it has none."""
        path = io.tree_file(output_dir)
        if path is None:
            return None
        try:
            return cls(path, units=units)
        except (OSError, KeyError):
            return None

    def __len__(self):
        return 0 if self.z is None else int(np.size(self.z))

    def s(self, key, default=None):
        return field(self.sub, key, SUB_ALIASES, default)

    def g(self, key, default=None):
        return field(self.grp, key, GRP_ALIASES, default)

    # -- physical-unit series ----------------------------------------------

    def _M(self):
        if self.units is not None:
            return self.units.mass_to_msun
        return 1.0e10 / 0.6774  # THESAN code mass unit / h, the standard choice

    def _L(self):
        if self.units is not None:
            return np.asarray([self.units.length_to_kpc(a) for a in self.a])
        return self.a / 0.6774

    def mstar_msun(self, use_inrad=True):
        """Main-branch stellar mass.  ``SubhaloMassInRadType`` when present."""
        mt = self.s("massinradtype") if use_inrad else None
        if mt is None:
            mt = self.s("masstype")
        return None if mt is None else np.asarray(mt)[:, io.PT_STAR] * self._M()

    def mgas_msun(self, use_inrad=True):
        mt = self.s("massinradtype") if use_inrad else None
        if mt is None:
            mt = self.s("masstype")
        return None if mt is None else np.asarray(mt)[:, io.PT_GAS] * self._M()

    def mhalo_msun(self):
        m = self.g("m200c")
        return None if m is None else np.asarray(m) * self._M()

    def mbh_msun(self):
        m = self.s("bhmass")
        return None if m is None else np.asarray(m) * self._M()

    def bh_mdot_msun_per_yr(self):
        m = self.s("bhmdot")
        if m is None:
            return None
        conv = self.units.mdot_to_msun_per_yr if self.units is not None else 1.0
        return np.asarray(m) * conv

    def sfr_msun_per_yr(self, use_inrad=True):
        s = self.s("sfrinrad") if use_inrad else None
        if s is None:
            s = self.s("sfr")
        return None if s is None else np.asarray(s)

    def sigma_kms(self):
        v = self.s("veldisp")
        return None if v is None else np.asarray(v)  # already physical km/s

    def pos_code(self):
        p = self.s("pos")
        return None if p is None else np.asarray(p, dtype=np.float64)

    def at_snapshot(self, num):
        """Row index for snapshot ``num``, or None."""
        if self.snapnum is None:
            return None
        hit = np.where(np.asarray(self.snapnum) == int(num))[0]
        return int(hit[0]) if hit.size else None

    def at_redshift(self, z, tol=0.2):
        """Row index nearest redshift ``z``, or None if nothing is within
        ``tol`` -- the tolerance matters, because the tree does not necessarily
        span the redshift you asked for."""
        if self.z is None:
            return None
        d = np.abs(np.asarray(self.z) - z)
        k = int(np.argmin(d))
        return k if d[k] <= tol else None

    def centre_at(self, num):
        """``SubhaloPos`` (code units) of the main branch at snapshot ``num``."""
        i = self.at_snapshot(num)
        p = self.pos_code()
        return None if (i is None or p is None) else p[i]

    def summary(self):
        """One-line-per-field description of what this tree actually carries."""
        lines = ["tree: %s  (%d entries" % (self.path, len(self))]
        if len(self):
            lines[0] += ", z = %.2f -> %.2f" % (np.nanmax(self.z), np.nanmin(self.z))
        lines[0] += ")"
        lines.append("  Subhalo: %s" % (", ".join(sorted(self.sub)) or "(none)"))
        lines.append("  Group  : %s" % (", ".join(sorted(self.grp)) or "(none)"))
        return "\n".join(lines)
