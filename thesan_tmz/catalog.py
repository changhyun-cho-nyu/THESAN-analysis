"""
Subfind catalogue layer -- the source of every *galaxy-scale* number.

The division of labour in this package, and the reason this module exists:

    groups_NNN/fof_subhalo_tab_NNN.*.hdf5   ->  which galaxies exist, and
                                                their integrated properties
    snapdir_NNN/snap_NNN.*.hdf5             ->  what the gas is doing *inside*
                                                one of them

For a THESAN-Zoom run the FoF **Group** is the whole zoom system -- the main
halo plus every satellite that shares its friends-of-friends envelope -- so
``GroupMassType[:,4]`` is the stellar mass of *all the galaxies together* and
is not the quantity any galaxy-scale plot wants.  An individual galaxy is a
**Subhalo**.  Everything here is therefore subhalo-centric; the Group table is
read only for the two things it is genuinely the authority on, ``Group_M_Crit200``
/ ``Group_R_Crit200`` (the halo the central galaxy lives in) and
``GroupFirstSub`` (which subhalo is that central).

Field names
-----------
AREPO2 (``src/fof/fof_io.cc``) does *not* write the Illustris/TNG-era names
that public analysis scripts use.  The differences that silently produce zeros
if you get them wrong:

    AREPO2 writes                    TNG-era name often assumed
    ------------------------------   ---------------------------------
    SubhaloGroupNr                   SubhaloGrNr
    SubhaloRankInGr                  (no equivalent; 0 == central)
    SubhaloSfrInRad                  SubhaloSFRinRad
    SubhaloHighResStellarFraction    (zoom-specific, no equivalent)
    (nothing)                        SubhaloFlag

Every lookup here goes through :func:`field`, which takes the list of spellings
and returns the first one present, so a catalogue written by either vintage of
the code reads correctly.  Ask for a name that is in neither and you get
``None`` -- never a silent array of zeros.

Units
-----
Taken from ``fof_io.cc`` and ``subfind_properties.cc``:

    SubhaloPos, SubhaloCM, SubhaloHalfmassRad*, SubhaloVmaxRad   comoving, /h
    SubhaloMass, SubhaloMassType, SubhaloMassInRadType           code mass, /h
    SubhaloVel, SubhaloVelDisp, SubhaloVmax                      km/s, physical
    SubhaloSFR, SubhaloSfrInRad, SubhaloSfrInHalfRad             Msun/yr
    SubhaloBHMass                                                code mass, /h
    SubhaloBHMdot                                                code, via
                                                                 Units.mdot_to_msun_per_yr
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io

# ---------------------------------------------------------------------------
# Name aliases.  First entry is what AREPO2 actually writes; the rest are
# spellings found in older catalogues and in public THESAN/TNG data.
# ---------------------------------------------------------------------------

SUB_ALIASES = {
    "groupnr": ("SubhaloGroupNr", "SubhaloGrNr"),
    "rank": ("SubhaloRankInGr", "SubhaloParentRank"),
    "pos": ("SubhaloPos",),
    "cm": ("SubhaloCM",),
    "vel": ("SubhaloVel",),
    "mass": ("SubhaloMass",),
    "masstype": ("SubhaloMassType",),
    "massinradtype": ("SubhaloMassInRadType",),
    "massinhalfradtype": ("SubhaloMassInHalfRadType",),
    "lentype": ("SubhaloLenType",),
    "halfmassradtype": ("SubhaloHalfmassRadType",),
    "halfmassrad": ("SubhaloHalfmassRad",),
    "veldisp": ("SubhaloVelDisp",),
    "vmax": ("SubhaloVmax",),
    "vmaxrad": ("SubhaloVmaxRad",),
    "sfr": ("SubhaloSFR", "SubhaloSfr"),
    "sfrinrad": ("SubhaloSfrInRad", "SubhaloSFRinRad", "SubhaloSFRInRad"),
    "sfrinhalfrad": ("SubhaloSfrInHalfRad", "SubhaloSFRinHalfRad"),
    "bhmass": ("SubhaloBHMass",),
    "bhmdot": ("SubhaloBHMdot",),
    "zgas": ("SubhaloGasMetallicity",),
    "zgassfr": ("SubhaloGasMetallicitySfr",),
    "zstar": ("SubhaloStarMetallicity",),
    "highres": ("SubhaloHighResStellarFraction",),
    "mostbound": ("SubhaloIDMostbound", "SubhaloIDMostBound"),
    "spin": ("SubhaloSpin",),
}

GRP_ALIASES = {
    "pos": ("GroupPos",),
    "vel": ("GroupVel",),
    "mass": ("GroupMass",),
    "masstype": ("GroupMassType",),
    "lentype": ("GroupLenType",),
    "firstsub": ("GroupFirstSub",),
    "nsubs": ("GroupNsubs",),
    "sfr": ("GroupSFR",),
    "bhmass": ("GroupBHMass",),
    "bhmdot": ("GroupBHMdot",),
    "m200c": ("Group_M_Crit200",),
    "r200c": ("Group_R_Crit200",),
    "m200m": ("Group_M_Mean200",),
    "r200m": ("Group_R_Mean200",),
    "mtophat": ("Group_M_TopHat200",),
    "rtophat": ("Group_R_TopHat200",),
}

#: Read by default.  Absent names are skipped by the reader, so this can list
#: fields that only some builds write.
SUBHALO_FIELDS = tuple(n for names in SUB_ALIASES.values() for n in names)
GROUP_FIELDS = tuple(n for names in GRP_ALIASES.values() for n in names)


def field(table, key, aliases=SUB_ALIASES, default=None):
    """First present spelling of ``key`` in ``table``, else ``default``.

    ``key`` may be a short name from the alias table ('masstype') or a literal
    dataset name ('SubhaloMassType').
    """
    names = aliases.get(key, (key,))
    for n in names:
        if n in table:
            return table[n]
    return default


# ---------------------------------------------------------------------------


class Catalog:
    """The FoF/Subfind catalogue of one snapshot, in physical units on demand.

    >>> cat = Catalog('/scratch/run/output', 42, units=snap.units)
    >>> i = cat.target_index()          # the central galaxy of the zoom halo
    >>> cat.properties(i)['mstar_msun']
    """

    def __init__(self, output_dir, num, units=None, a=None, sub_fields=SUBHALO_FIELDS, grp_fields=GROUP_FIELDS):
        self.output_dir = output_dir
        self.num = int(num)
        self.sub = io.read_group_catalog(output_dir, num, list(sub_fields), table="Subhalo")
        self.grp = io.read_group_catalog(output_dir, num, list(grp_fields), table="Group")
        self.header = self.sub.get("header") or self.grp.get("header") or {}
        self.a = float(self.header.get("Time", 1.0)) if a is None else float(a)
        self.z = float(self.header.get("Redshift", 1.0 / self.a - 1.0))
        if units is None:
            from .units import Units

            units = Units.from_snapshot_file(io.group_files(output_dir, num)[0])
        self.units = units

    # -- sizes --------------------------------------------------------------

    @property
    def n_sub(self):
        return int(self.sub.get("count", 0))

    @property
    def n_grp(self):
        return int(self.grp.get("count", 0))

    def s(self, key, default=None):
        """Subhalo field by short name or literal name."""
        return field(self.sub, key, SUB_ALIASES, default)

    def g(self, key, default=None):
        """Group field by short name or literal name."""
        return field(self.grp, key, GRP_ALIASES, default)

    # -- structure ----------------------------------------------------------

    def central_indices(self):
        """Indices of the central subhalo of every group.

        Prefers ``GroupFirstSub`` (the group's own statement of which subhalo is
        its central); falls back to ``SubhaloRankInGr == 0``, which AREPO2
        writes and which means the same thing.  Groups with no subhalo carry
        ``GroupFirstSub == -1`` and are dropped.
        """
        first = self.g("firstsub")
        if first is not None:
            f = np.asarray(first, dtype=np.int64)
            return f[(f >= 0) & (f < self.n_sub)]
        rank = self.s("rank")
        if rank is not None:
            return np.where(np.asarray(rank) == 0)[0]
        return np.zeros(0, dtype=np.int64)

    def is_central(self):
        """Boolean mask over subhaloes: True for the central of its group."""
        m = np.zeros(self.n_sub, dtype=bool)
        idx = self.central_indices()
        if idx.size:
            m[idx] = True
        return m

    def group_of(self):
        """Group index of each subhalo, from ``SubhaloGroupNr``.

        Returns None if the catalogue does not carry it -- callers must handle
        that rather than assume group 0, which is the bug this replaces.
        """
        gn = self.s("groupnr")
        return None if gn is None else np.asarray(gn, dtype=np.int64)

    # -- picking the galaxy the figures are about ---------------------------

    def target_index(self, prefer="central", bh_pos=None, min_star_particles=0, require_highres=None):
        """Index of the galaxy this run is about.

        ``prefer``:
          ``'central'``  the central subhalo of group 0.  For a zoom, group 0 is
                         the target halo and its central is the target galaxy.
                         This is the right default.
          ``'bh'``       the subhalo whose ``SubhaloPos`` is nearest ``bh_pos``
                         (code coordinates), i.e. the galaxy hosting that black
                         hole.  Falls back to ``'massivebh'`` if no position.
          ``'massivebh'`` the subhalo with the largest ``SubhaloBHMass``.
          ``'massive'``  the subhalo with the largest stellar mass.

        Returns None when the catalogue is empty or nothing passes the cuts.
        """
        if self.n_sub == 0:
            return None

        keep = np.ones(self.n_sub, dtype=bool)
        if min_star_particles > 0:
            lt = self.s("lentype")
            if lt is not None:
                keep &= np.asarray(lt)[:, io.PT_STAR] >= min_star_particles
        if require_highres is not None:
            hr = self.s("highres")
            if hr is not None:
                keep &= np.asarray(hr) >= require_highres
        if not keep.any():
            keep = np.ones(self.n_sub, dtype=bool)

        if prefer == "central":
            first = self.g("firstsub")
            if first is not None and self.n_grp > 0:
                i = int(np.asarray(first)[0])
                if 0 <= i < self.n_sub:
                    return i
            rank = self.s("rank")
            if rank is not None:
                hit = np.where(np.asarray(rank) == 0)[0]
                if hit.size:
                    return int(hit[0])
            return 0  # subhaloes are written most-massive-first within group 0

        if prefer == "bh" and bh_pos is not None:
            pos = self.s("pos")
            if pos is not None:
                box = float(self.header.get("BoxSize", 0.0)) or None
                r = io.radius_from(np.asarray(pos), np.asarray(bh_pos, dtype=np.float64), box)
                r = np.where(keep, r, np.inf)
                return int(np.argmin(r))
            prefer = "massivebh"

        if prefer in ("bh", "massivebh"):
            mbh = self.s("bhmass")
            if mbh is not None and np.any(np.asarray(mbh)[keep] > 0):
                return int(np.argmax(np.where(keep, np.asarray(mbh), -np.inf)))
            prefer = "massive"

        mt = self.s("masstype")
        if mt is None:
            return int(np.argmax(np.where(keep, np.asarray(self.s("mass", np.zeros(self.n_sub))), -np.inf)))
        return int(np.argmax(np.where(keep, np.asarray(mt)[:, io.PT_STAR], -np.inf)))

    # -- physical-unit output ----------------------------------------------

    def properties(self, i, use_inrad=True):
        """Every galaxy-scale quantity for subhalo ``i``, in physical units.

        ``use_inrad`` reports stellar/gas mass and SFR inside twice the stellar
        half-mass radius (``*InRadType``) rather than the total bound mass.
        That is the closer analogue of an observed aperture, and the two differ
        by ~0.2 dex for a typical galaxy -- the same size as the offsets people
        argue about in the SMHM relation, so which one is used is recorded in
        the output as ``aperture``.
        """
        if i is None or not (0 <= i < self.n_sub):
            return {}
        u = self.units
        M, L = u.mass_to_msun, u.length_to_kpc(self.a)
        out = {"subhalo_index": int(i), "a": self.a, "z": self.z, "aperture": "inrad" if use_inrad else "total"}

        mt_tot = self.s("masstype")
        mt_rad = self.s("massinradtype")
        mt = mt_rad if (use_inrad and mt_rad is not None) else mt_tot
        if mt is not None:
            mt = np.asarray(mt)
            out["mstar_msun"] = float(mt[i, io.PT_STAR] * M)
            out["mgas_msun"] = float(mt[i, io.PT_GAS] * M)
            out["mdm_msun"] = float(mt[i, io.PT_DM] * M)
        if mt_tot is not None:
            mt_tot = np.asarray(mt_tot)
            out["mstar_total_msun"] = float(mt_tot[i, io.PT_STAR] * M)
            out["mgas_total_msun"] = float(mt_tot[i, io.PT_GAS] * M)
            out["mbound_msun"] = float(mt_tot[i].sum() * M)

        for key, name, conv in (
            ("mass", "msub_msun", M),
            ("bhmass", "bh_mass_msun", M),
            ("veldisp", "sigma_kms", 1.0),
            ("vmax", "vmax_kms", 1.0),
            ("halfmassrad", "rhalf_kpc", L),
            ("vmaxrad", "rvmax_kpc", L),
            ("zgas", "Zgas", 1.0),
            ("zstar", "Zstar", 1.0),
            ("highres", "highres_stellar_fraction", 1.0),
        ):
            v = self.s(key)
            if v is not None:
                out[name] = float(np.asarray(v)[i] * conv)

        sfr = self.s("sfrinrad") if use_inrad else None
        if sfr is None:
            sfr = self.s("sfr")
        if sfr is not None:
            out["sfr_msun_per_yr"] = float(np.asarray(sfr)[i])
        tot_sfr = self.s("sfr")
        if tot_sfr is not None:
            out["sfr_total_msun_per_yr"] = float(np.asarray(tot_sfr)[i])

        mdot = self.s("bhmdot")
        if mdot is not None:
            out["bh_mdot_msun_per_yr"] = float(np.asarray(mdot)[i] * u.mdot_to_msun_per_yr)
            if out.get("bh_mass_msun", 0.0) > 0:
                edd = u.eddington_rate_msun_per_yr(out["bh_mass_msun"])
                out["eddington_ratio"] = out["bh_mdot_msun_per_yr"] / edd if edd > 0 else np.nan

        hr = self.s("halfmassradtype")
        if hr is not None:
            hr = np.asarray(hr)
            out["rhalf_star_kpc"] = float(hr[i, io.PT_STAR] * L)
            out["rhalf_gas_kpc"] = float(hr[i, io.PT_GAS] * L)

        lt = self.s("lentype")
        if lt is not None:
            lt = np.asarray(lt)
            out["nstar"] = int(lt[i, io.PT_STAR])
            out["ngas"] = int(lt[i, io.PT_GAS])
            out["nbh"] = int(lt[i, io.PT_BH]) if lt.shape[1] > io.PT_BH else 0

        pos = self.s("pos")
        if pos is not None:
            out["pos_code"] = np.asarray(pos, dtype=np.float64)[i]
        vel = self.s("vel")
        if vel is not None:
            out["vel_kms"] = np.asarray(vel, dtype=np.float64)[i]  # already physical km/s

        # -- host halo.  The Group is the whole zoom system, so this is the
        #    halo mass of the *system*, correct for the SMHM x-axis only when
        #    ``i`` is that group's central.
        gn = self.group_of()
        gi = int(gn[i]) if (gn is not None and i < gn.size) else (0 if self.n_grp else None)
        if gi is not None and 0 <= gi < self.n_grp:
            out["group_index"] = gi
            out["is_central"] = bool(self.is_central()[i])
            for key, name, conv in (
                ("m200c", "mhalo_msun", M),
                ("r200c", "r200c_kpc", L),
                ("mtophat", "mhalo_tophat_msun", M),
                ("sfr", "group_sfr_msun_per_yr", 1.0),
                ("mass", "mgroup_msun", M),
            ):
                v = self.g(key)
                if v is not None:
                    out[name] = float(np.asarray(v)[gi] * conv)

        # -- derived ------------------------------------------------------
        ms, mg = out.get("mstar_msun", 0.0), out.get("mgas_msun", 0.0)
        if ms > 0:
            out["ssfr_per_gyr"] = out.get("sfr_msun_per_yr", 0.0) / ms * 1.0e9
            if mg + ms > 0:
                out["fgas"] = mg / (mg + ms)
            if out.get("bh_mass_msun", 0.0) > 0:
                out["mbh_over_mstar"] = out["bh_mass_msun"] / ms
            if out.get("mhalo_msun", 0.0) > 0:
                out["mstar_over_mhalo"] = ms / out["mhalo_msun"]
        return out

    # -- the whole population ----------------------------------------------

    def population(self, min_star_particles=10, use_inrad=True, centrals_only=False, min_highres=None):
        """Physical-unit arrays over every subhalo passing the cuts.

        The floor is a *particle count*, not a mass, so it follows the run's
        actual resolution.  ``min_highres`` additionally drops subhaloes whose
        ``SubhaloHighResStellarFraction`` shows low-resolution contamination --
        for a zoom this is the cut that keeps interloper haloes from the padding
        region out of the plot, and there is no substitute for it.
        """
        if self.n_sub == 0:
            return None
        u = self.units
        M, L = u.mass_to_msun, u.length_to_kpc(self.a)

        mt_tot = self.s("masstype")
        mt_rad = self.s("massinradtype")
        if mt_tot is None and mt_rad is None:
            return None
        mt = mt_rad if (use_inrad and mt_rad is not None) else mt_tot
        mt = np.asarray(mt, dtype=np.float64)
        n = mt.shape[0]

        p = {
            "mstar": mt[:, io.PT_STAR] * M,
            "mgas": mt[:, io.PT_GAS] * M,
            "mdm": mt[:, io.PT_DM] * M,
            "z": self.z,
            "aperture": "inrad" if (use_inrad and mt_rad is not None) else "total",
        }
        if mt_tot is not None:
            p["mtot_bound"] = np.asarray(mt_tot, dtype=np.float64).sum(axis=1) * M

        def put(key, name, conv=1.0, col=None):
            v = self.s(key)
            if v is None:
                return
            v = np.asarray(v, dtype=np.float64)
            p[name] = (v[:, col] if col is not None else v) * conv

        put("bhmass", "mbh", M)
        put("bhmdot", "mdot", u.mdot_to_msun_per_yr)
        put("veldisp", "sigma")
        put("vmax", "vmax")
        put("zgas", "Zgas")
        put("zstar", "Zstar")
        put("highres", "highres")
        put("halfmassradtype", "rhalf_star", L, col=io.PT_STAR)

        sfr = self.s("sfrinrad") if use_inrad else None
        p["sfr"] = np.asarray(sfr if sfr is not None else self.s("sfr", np.zeros(n)), dtype=np.float64)

        pos = self.s("pos")
        if pos is not None:
            p["pos_code"] = np.asarray(pos, dtype=np.float64)

        cen = self.is_central()
        p["central"] = cen

        # -- host halo per subhalo, via SubhaloGroupNr ----------------------
        gn = self.group_of()
        m200 = self.g("m200c")
        if gn is not None and m200 is not None and self.n_grp:
            gi = np.clip(gn, 0, self.n_grp - 1)
            p["grnr"] = gn
            p["m200c"] = np.asarray(m200, dtype=np.float64)[gi] * M
            r200 = self.g("r200c")
            if r200 is not None:
                p["r200c"] = np.asarray(r200, dtype=np.float64)[gi] * L
            # A satellite's "halo mass" is its host's, which is not the SMHM
            # x-axis; mask it so a caller cannot plot it by accident.
            p["m200c_central"] = np.where(cen, p["m200c"], np.nan)
        elif gn is None:
            p["grnr"] = None

        # -- cuts -----------------------------------------------------------
        keep = np.ones(n, dtype=bool)
        lt = self.s("lentype")
        if lt is not None and min_star_particles > 0:
            keep &= np.asarray(lt)[:, io.PT_STAR] >= min_star_particles
        else:
            keep &= p["mstar"] > 0
        if min_highres is not None and "highres" in p:
            keep &= p["highres"] >= min_highres
        if centrals_only:
            keep &= cen

        out = {}
        for k, v in p.items():
            out[k] = v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v
        out["count"] = int(keep.sum())
        out["n_total"] = n
        return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def load(snap_or_dir, num=None, units=None):
    """Build a :class:`Catalog` from a :class:`~thesan_tmz.arepo_io.Snapshot`
    or from ``(output_dir, num)``.  Returns None if there is no catalogue."""
    try:
        if num is None:
            snap = snap_or_dir
            return Catalog(snap.output_dir, snap.num, units=units or snap.units, a=snap.a)
        return Catalog(snap_or_dir, num, units=units)
    except (FileNotFoundError, OSError, KeyError):
        return None


def target_galaxy(snap, prefer="central", bh_pos=None, use_inrad=True, units=None):
    """``(catalog, index, properties)`` for the galaxy a snapshot is about.

    Returns ``(None, None, {})`` when the snapshot has no group catalogue --
    early snapshots of a zoom often do not.
    """
    cat = load(snap, units=units)
    if cat is None or cat.n_sub == 0:
        return None, None, {}
    i = cat.target_index(prefer=prefer, bh_pos=bh_pos)
    return cat, i, cat.properties(i, use_inrad=use_inrad)
