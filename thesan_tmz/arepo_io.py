"""
Snapshot and group-catalogue I/O for AREPO2 HDF5 output.

Handles every layout AREPO writes:

    <output>/snap_NNN.hdf5                       single file
    <output>/snapdir_NNN/snap_NNN.M.hdf5         split over M files
    <output>/fof_subhalo_tab_NNN.hdf5            single file
    <output>/groups_NNN/fof_subhalo_tab_NNN.M.hdf5

Nothing here knows about black holes specifically -- it is the plumbing that
every other module in the package sits on.

Particle-type numbering in this branch (src/main/allvars.h):

    0 gas   1 halo/DM   2 (unused / bulge)   3 tracers   4 stars   5 black holes
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("thesan_tmz needs h5py: pip install h5py") from exc

from .units import Units

PT_GAS = 0
PT_DM = 1
PT_TRACER = 3
PT_STAR = 4
PT_BH = 5

PT_NAMES = {0: "gas", 1: "dm", 2: "type2", 3: "tracer", 4: "stars", 5: "bh"}


# ---------------------------------------------------------------------------
# Locating files
# ---------------------------------------------------------------------------


def snapshot_files(output_dir, num):
    """Return the sorted list of HDF5 files making up snapshot ``num``.

    Raises FileNotFoundError if the snapshot does not exist, so a caller
    looping over numbers can simply catch it.
    """
    n = "%03d" % int(num)
    candidates = [
        os.path.join(output_dir, "snapdir_%s" % n, "snap_%s.*.hdf5" % n),
        os.path.join(output_dir, "snapdir_%s" % n, "snapshot_%s.*.hdf5" % n),
        os.path.join(output_dir, "snap_%s.hdf5" % n),
        os.path.join(output_dir, "snapshot_%s.hdf5" % n),
    ]
    for pat in candidates:
        hits = sorted(glob.glob(pat), key=_file_chunk_index)
        if hits:
            return hits
    raise FileNotFoundError("no snapshot %s under %s" % (n, output_dir))


def group_files(output_dir, num):
    """Return the sorted list of HDF5 files of the FoF/Subfind catalogue ``num``."""
    n = "%03d" % int(num)
    candidates = [
        os.path.join(output_dir, "groups_%s" % n, "fof_subhalo_tab_%s.*.hdf5" % n),
        os.path.join(output_dir, "groups_%s" % n, "fof_tab_%s.*.hdf5" % n),
        os.path.join(output_dir, "fof_subhalo_tab_%s.hdf5" % n),
        os.path.join(output_dir, "fof_tab_%s.hdf5" % n),
    ]
    for pat in candidates:
        hits = sorted(glob.glob(pat), key=_file_chunk_index)
        if hits:
            return hits
    raise FileNotFoundError("no group catalogue %s under %s" % (n, output_dir))


def _file_chunk_index(path):
    """Sort key that orders snap_012.9.hdf5 before snap_012.10.hdf5."""
    m = re.search(r"\.(\d+)\.hdf5$", path)
    return int(m.group(1)) if m else -1


def available_snapshots(output_dir):
    """All snapshot numbers present under ``output_dir``, sorted ascending."""
    nums = set()
    for pat, rx in (
        ("snapdir_*", r"snapdir_(\d+)$"),
        ("snap_*.hdf5", r"snap_(\d+)\.hdf5$"),
        ("snapshot_*.hdf5", r"snapshot_(\d+)\.hdf5$"),
    ):
        for p in glob.glob(os.path.join(output_dir, pat)):
            m = re.search(rx, p)
            if m:
                nums.add(int(m.group(1)))
    return sorted(nums)


def available_group_catalogs(output_dir):
    nums = set()
    for pat, rx in (
        ("groups_*", r"groups_(\d+)$"),
        ("fof_subhalo_tab_*.hdf5", r"fof_subhalo_tab_(\d+)\.hdf5$"),
    ):
        for p in glob.glob(os.path.join(output_dir, pat)):
            m = re.search(rx, p)
            if m:
                nums.add(int(m.group(1)))
    return sorted(nums)


def tree_file(output_dir):
    """Path to the merger tree, or None.

    THESAN-Zoom writes ``<run>/output/tree.hdf5`` holding the main branch of
    the target halo: ``Redshifts``, ``Snapshots`` and the ``Group``/``Subhalo``
    tables sampled along it.  When it exists it is a far better source for the
    growth history of the central galaxy than re-deriving one snapshot by
    snapshot, because it already resolves which subhalo is the progenitor
    across a merger.
    """
    for cand in (
        os.path.join(output_dir, "tree.hdf5"),
        os.path.join(os.path.dirname(output_dir.rstrip("/")), "tree.hdf5"),
        os.path.join(output_dir, "trees", "tree.hdf5"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def candidates_file(output_dir, num):
    """Path to ``postprocessing/candidates/candidates_NNN.hdf5``, or None.

    Written beside ``output/`` rather than inside it, so this looks one level
    up as well as in place.
    """
    n = "%03d" % int(num)
    run_dir = os.path.dirname(output_dir.rstrip("/"))
    for base in (run_dir, output_dir):
        cand = os.path.join(base, "postprocessing", "candidates", "candidates_%s.hdf5" % n)
        if os.path.isfile(cand):
            return cand
    return None


def read_expansion_list(path):
    """Scale factors from an ``ExpansionList``/``output_*.list`` file.

    Returns ``(a, z)``.  Snapshot number is the row index, which is how the
    THESAN scripts map a target redshift onto a snapshot without opening one.
    """
    a = np.atleast_1d(np.loadtxt(path))
    if a.ndim > 1:
        a = a[:, 0]
    return a, 1.0 / a - 1.0


def snapshot_for_redshift(output_dir, z, expansion_list=None):
    """Snapshot number closest to redshift ``z``.

    Uses the expansion list when one is given (no I/O per snapshot); otherwise
    reads each snapshot's header.  Returns None if nothing is available.
    """
    if expansion_list is not None and os.path.isfile(expansion_list):
        _, zs = read_expansion_list(expansion_list)
        return int(np.argmin(np.abs(zs - z)))
    nums = available_snapshots(output_dir)
    if not nums:
        return None
    zs = []
    for n in nums:
        try:
            zs.append(float(read_header(snapshot_files(output_dir, n)[0]).get("Redshift", np.nan)))
        except (OSError, FileNotFoundError, KeyError):
            zs.append(np.nan)
    zs = np.asarray(zs, dtype=np.float64)
    if np.all(np.isnan(zs)):
        return None
    return int(nums[int(np.nanargmin(np.abs(zs - z)))])


# ---------------------------------------------------------------------------
# Header / Config / Parameters
# ---------------------------------------------------------------------------


def read_header(path):
    """``/Header`` attributes of one HDF5 file, as a plain dict."""
    with h5py.File(path, "r") as f:
        return {k: _decode(v) for k, v in f["Header"].attrs.items()}


def read_config(path):
    """``/Config`` attributes -- the Config.sh options the binary was built with.

    Values are the option's value where one was given (``FOO=3`` -> 3.0) and an
    empty string for a bare ``#define``.  Use :func:`has_option` rather than
    testing truthiness, since a bare option's value is falsy.
    """
    with h5py.File(path, "r") as f:
        if "Config" not in f:
            return {}
        return {k: _decode(v) for k, v in f["Config"].attrs.items()}


def read_parameters(path):
    """``/Parameters`` attributes -- the run's parameter file, as a dict."""
    with h5py.File(path, "r") as f:
        if "Parameters" not in f:
            return {}
        return {k: _decode(v) for k, v in f["Parameters"].attrs.items()}


def has_option(config, name):
    """True if Config.sh option ``name`` was enabled in this build."""
    return name in config


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return [x.decode("utf-8", "replace") for x in v]
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Particle data
# ---------------------------------------------------------------------------


def read_snapshot(output_dir, num, parttype, fields=None, verbose=False):
    """Read particle fields from a (possibly split) snapshot.

    Parameters
    ----------
    parttype : int
        0/1/4/5 for gas/DM/stars/BHs.
    fields : list of str or None
        Dataset names to read.  ``None`` reads every dataset present.  Names
        that are absent are silently skipped -- check the returned keys.

    Returns
    -------
    dict
        ``{field: ndarray}`` concatenated across files, plus the special keys
        ``'header'`` (dict) and ``'count'`` (int).  Masses of a type with a
        non-zero ``MassTable`` entry are synthesised so ``'Masses'`` is always
        available.
    """
    files = snapshot_files(output_dir, num)
    grp = "PartType%d" % parttype
    header = read_header(files[0])

    ntot = int(np.asarray(header["NumPart_Total"])[parttype])
    if "NumPart_Total_HighWord" in header:
        ntot += int(np.asarray(header["NumPart_Total_HighWord"])[parttype]) << 32
    if ntot == 0:
        return {"header": header, "count": 0}

    # Work out which fields exist, from the first file that actually holds
    # particles of this type (a split snapshot can have empty leading chunks).
    present = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if grp in f:
                present = list(f[grp].keys())
                break
    if present is None:
        return {"header": header, "count": 0}

    want = present if fields is None else [x for x in fields if x in present]
    missing = [] if fields is None else [x for x in fields if x not in present]
    if verbose and missing:
        print("  [read_snapshot] not in PartType%d: %s" % (parttype, ", ".join(missing)))

    out, filled = {}, 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            if grp not in f:
                continue
            n_here = int(np.asarray(f["Header"].attrs["NumPart_ThisFile"])[parttype])
            if n_here == 0:
                continue
            for key in want:
                d = f[grp][key]
                if key not in out:
                    shape = (ntot,) + d.shape[1:]
                    out[key] = np.empty(shape, dtype=d.dtype)
                out[key][filled : filled + n_here] = d[:n_here]
            filled += n_here

    # Same truncation guard as the catalogue reader: an incomplete file set
    # would otherwise leave the tail of every array as uninitialised memory.
    if filled < ntot:
        if verbose:
            print("  [read_snapshot] PartType%d: %d of %d particles found -- truncating"
                  % (parttype, filled, ntot))
        for key in list(out):
            out[key] = out[key][:filled]
        ntot = filled

    # MassTable types (usually DM) carry no Masses dataset.
    if "Masses" not in out:
        mt = np.asarray(header.get("MassTable", np.zeros(6)))
        if parttype < len(mt) and mt[parttype] > 0:
            out["Masses"] = np.full(ntot, float(mt[parttype]), dtype=np.float64)

    out["header"] = header
    out["count"] = filled
    return out


def read_group_catalog(output_dir, num, fields=None, table="Group"):
    """Read a FoF (``table='Group'``) or Subfind (``table='Subhalo'``) table.

    Returns ``{field: ndarray}`` plus ``'header'`` and ``'count'``.
    """
    files = group_files(output_dir, num)
    header = read_header(files[0])

    # AREPO2 writes Nsubhalos_* (fof_io.cc:891); the Nsubgroups_* spelling is
    # the legacy Illustris/TNG one, which the code itself still accepts on read
    # (fof_io.cc:923).  Try the current name first, not the legacy one.
    if table == "Group":
        tot_names, this_names = ("Ngroups_Total",), ("Ngroups_ThisFile",)
    else:
        tot_names = ("Nsubhalos_Total", "Nsubgroups_Total")
        this_names = ("Nsubhalos_ThisFile", "Nsubgroups_ThisFile")
    key_tot = next((k for k in tot_names if k in header), tot_names[0])
    key_this = next((k for k in this_names if k in header), this_names[0])

    ntot = int(header.get(key_tot, 0))
    if ntot == 0:
        return {"header": header, "count": 0}

    present = None
    for fn in files:
        with h5py.File(fn, "r") as f:
            if table in f:
                present = list(f[table].keys())
                break
    if present is None:
        return {"header": header, "count": 0}

    want = present if fields is None else [x for x in fields if x in present]

    out, filled = {}, 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            if table not in f:
                continue
            attrs = f["Header"].attrs
            if key_this in attrs:
                n_here = int(np.asarray(attrs[key_this]))
            else:  # a chunk written without the count: trust the dataset
                n_here = int(f[table][want[0]].shape[0]) if want else 0
            if n_here == 0:
                continue
            n_here = min(n_here, ntot - filled)
            if n_here <= 0:
                break
            for key in want:
                d = f[table][key]
                if key not in out:
                    out[key] = np.empty((ntot,) + d.shape[1:], dtype=d.dtype)
                out[key][filled : filled + n_here] = d[:n_here]
            filled += n_here

    # A catalogue can be short of its advertised total if a chunk is missing or
    # still being written.  Trim rather than hand back uninitialised np.empty
    # memory, which looks like real data and is not.
    if filled < ntot:
        print("  [read_group_catalog] %s %03d: %d of %d rows found -- truncating"
              % (table, int(num), filled, ntot))
        for key in list(out):
            out[key] = out[key][:filled]

    out["header"] = header
    out["count"] = filled
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def periodic_delta(x, x0, boxsize):
    """Minimum-image separation ``x - x0`` for a periodic box.

    ``boxsize=None`` (or 0) turns the wrapping off, which is what you want for
    a non-periodic test problem.
    """
    d = np.asarray(x, dtype=np.float64) - np.asarray(x0, dtype=np.float64)
    if boxsize:
        d -= boxsize * np.round(d / boxsize)
    return d


def radius_from(pos, centre, boxsize=None):
    d = periodic_delta(pos, np.asarray(centre, dtype=np.float64), boxsize)
    return np.sqrt(np.sum(d * d, axis=-1))


def spherical_mask(pos, centre, radius, boxsize=None):
    return radius_from(pos, centre, boxsize) <= radius


# ---------------------------------------------------------------------------
# The convenience object most scripts will use
# ---------------------------------------------------------------------------


class Snapshot:
    """One snapshot, with lazy per-type reads and cached unit conversions.

    >>> snap = Snapshot('/scratch/run/output', 42)
    >>> gas = snap.part(0, ['Coordinates', 'Masses', 'Density'])
    >>> print(snap.z, snap.age_gyr)
    """

    def __init__(self, output_dir, num, units=None):
        self.output_dir = output_dir
        self.num = int(num)
        self.files = snapshot_files(output_dir, num)
        self.header = read_header(self.files[0])
        self.config = read_config(self.files[0])
        self.parameters = read_parameters(self.files[0])
        self.units = units or Units.from_snapshot_file(self.files[0])
        self._cache = {}

    # -- scalars ------------------------------------------------------------

    @property
    def a(self):
        """Expansion factor (``Time`` for a comoving run)."""
        return float(self.header.get("Time", 1.0))

    @property
    def z(self):
        return float(self.header.get("Redshift", self.units.redshift(self.a)))

    @property
    def time(self):
        """Raw ``Header/Time``, whatever it means for this run."""
        return float(self.header["Time"])

    @property
    def age_gyr(self):
        return float(self.units.age_gyr(self.a))

    @property
    def boxsize(self):
        return float(self.header.get("BoxSize", 0.0)) or None

    def npart(self, parttype):
        n = int(np.asarray(self.header["NumPart_Total"])[parttype])
        if "NumPart_Total_HighWord" in self.header:
            n += int(np.asarray(self.header["NumPart_Total_HighWord"])[parttype]) << 32
        return n

    # -- data ---------------------------------------------------------------

    def part(self, parttype, fields=None, cache=True):
        key = (parttype, tuple(fields) if fields else None)
        if cache and key in self._cache:
            return self._cache[key]
        d = read_snapshot(self.output_dir, self.num, parttype, fields)
        if cache:
            self._cache[key] = d
        return d

    def groups(self, fields=None):
        return read_group_catalog(self.output_dir, self.num, fields, table="Group")

    def subhalos(self, fields=None):
        return read_group_catalog(self.output_dir, self.num, fields, table="Subhalo")

    def field_names(self, parttype):
        grp = "PartType%d" % parttype
        for fn in self.files:
            with h5py.File(fn, "r") as f:
                if grp in f:
                    return sorted(f[grp].keys())
        return []

    # -- physical-unit shortcuts -------------------------------------------

    def masses_msun(self, parttype):
        d = self.part(parttype, ["Masses"])
        if d["count"] == 0 and "Masses" not in d:
            return np.zeros(0)
        return d["Masses"] * self.units.mass_to_msun

    def positions_kpc(self, parttype, centre=None):
        d = self.part(parttype, ["Coordinates"])
        if "Coordinates" not in d:
            return np.zeros((0, 3))
        p = d["Coordinates"]
        if centre is not None:
            p = periodic_delta(p, centre, self.boxsize)
        return p * self.units.length_to_kpc(self.a)

    def gas_temperature(self):
        d = self.part(0, ["InternalEnergy", "ElectronAbundance"])
        return self.units.temperature(d["InternalEnergy"], d.get("ElectronAbundance"))

    def gas_nH(self):
        d = self.part(0, ["Density"])
        return d["Density"] * self.units.density_to_nH_cgs(self.a)

    def __repr__(self):
        return "Snapshot(%s, num=%d, z=%.3f, a=%.5f)" % (self.output_dir, self.num, self.z, self.a)


def snapshot_series(output_dir, nums=None, units=None):
    """Yield :class:`Snapshot` objects in time order, skipping unreadable ones."""
    nums = available_snapshots(output_dir) if nums is None else list(nums)
    for n in nums:
        try:
            yield Snapshot(output_dir, n, units=units)
        except (OSError, FileNotFoundError, KeyError) as exc:
            print("  [snapshot_series] skipping %d: %s" % (n, exc))
