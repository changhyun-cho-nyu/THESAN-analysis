"""
Main-branch track of the zoom target from its dark-matter core -- the method of
Rahul Kannan's ``core_track.py`` (``~/Downloads/analyse``, 2026-11), written
for this package with its readers, the same steps and the same parameter values.

Why not "the central of FoF group 0" at every snapshot
--------------------------------------------------------
A FoF group can hold two massive halos that have not merged yet, and SUBFIND
then gives the label "Subhalo 0" to one or the other from snapshot to
snapshot.  A history that takes group 0's central at each snapshot jumps
between them (g0z16: M*(sub) changing 20x between neighbouring snapshots).

Method (from the final snapshot backwards)
------------------------------------------
Anchors are the snapshots with particle data.  At an anchor:

  centre   DM density peak: shrinking sphere on the DM (start 10 ckpc/h,
           factor 0.85, stop below 50 particles), seeded at a position.
  mass     spherical-overdensity M200c of the DM about that centre, M200_DM;
           M200 = M200_DM / f_DM with f_DM = (Om - Ob) / Om.
  T        tracers: all DM within TFRAC x R200 of the centre.
  core     the NCORE innermost DM particles.

Going from anchor s to the previous anchor s':
  1. find T and the core at s' by particle ID.  Seeds = the point the core
     collapses onto + SubhaloPos of every subhalo holding >= 5 % of T.
  2. refine each seed to a DM peak; merge peaks closer than 3 ckpc/h.
  3. for each peak: SO M200 and its share of T (each T particle goes to the
     nearest peak whose R200 contains it).
  4. candidates = peaks with share >= MIN_SHARE, plus the core's peak.
  5. the core's peak continues the lineage; other candidates start new ones
     (at most 4 kept).  At the first anchor where all lineages are separate
     halos (distance > R200_i + R200_j) the most massive wins and its history
     becomes the main branch -- the "mass before the merger" definition.
  6. the subhalo is the one owning most DM within 3 ckpc/h of the peak.

Catalogue-only snapshots between anchors: a subhalo whose SubhaloIDMostbound
is among the particles within max(0.1 R200, 2 ckpc/h) of an anchor's centre
(and has >= 20 % of the anchors' DM) -- the one whose particle sat closest to
the centre; otherwise, among subhalos within max(5 ckpc/h, 0.1 R200) of the
centre interpolated in scale factor, the one with most DM (a tree link breaks
near-ties); otherwise the nearest subhalo with >= 20 % of that DM.

Output: ``<figdir>/core_track.hdf5`` in the layout of a THESAN-Zoom
``tree.hdf5`` (Snapshots, Redshifts, Subhalo/*, Group/*) plus CentrePos,
M200DM, M200, R200, TShare, NCandidates, Method (0 anchor, 3 most-bound ID,
1/2 position + mass, tree-linked / not), InterpOffset, IDMatchRank.  The run's
``output/`` is never written to.

Needs a group-ordered snapshot (``SubhaloOffsetType``) and equal-mass
high-resolution DM (``MassTable[1]``).
"""

from __future__ import annotations

import glob
import os
import shutil

import h5py
import numpy as np

from . import arepo_io as io

# Rahul's values (core_track.py defaults and constants), kept so the two agree
NCORE = 200            # innermost DM particles defining the core
TFRAC = 0.5            # tracers T = DM within TFRAC x R200
MIN_SHARE = 0.15       # minimum share of T for a progenitor candidate
SEED_SHARE = 0.05      # a subhalo holding this share of T seeds a peak
MERGE_DIST = 3.0       # ckpc/h: peaks / lineages closer than this are one
LABEL_RAD = 3.0        # ckpc/h: the subhalo owning most DM within this of the peak
MAX_LINEAGES = 4
PEAK_R0, PEAK_NMIN, SHRINK_FAC = 10.0, 50, 0.85
SO_RMAX0 = 100.0       # ckpc/h: first radius for the SO search
FIRST_HALF = 500.0     # ckpc/h: region half-side at the final snapshot
FILL_TOL, FILL_R, ID_RFRAC, ID_RMIN, NDM_FRAC = 0.5, 5.0, 0.1, 2.0, 0.2

TRACK_FILE = "core_track.hdf5"


def _code_fingerprint():
    """Hash of this module: a track built by other code is rebuilt."""
    import hashlib

    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


def periodic(dx, box):
    return dx - box * np.round(dx / box)


def shrink(x, c0, r0, nmin, fac=SHRINK_FAC):
    """Shrinking-sphere centre of points ``x``."""
    c, r = np.asarray(c0, np.float64), r0
    while True:
        sel = np.sqrt(((x - c) ** 2).sum(axis=1)) < r
        if sel.sum() < nmin:
            return c
        c = x[sel].mean(axis=0)
        r *= fac


class _DM:
    """High-resolution DM (PartType1) IDs and positions of one snapshot."""

    def __init__(self, files, box):
        ids, pos = [], []
        for _, d in io.iter_type(files, io.PT_DM, ("ParticleIDs", "Coordinates")):
            ids.append(d["ParticleIDs"])
            pos.append(d["Coordinates"])
        self.ids = np.concatenate(ids)
        self.pos = np.concatenate(pos)
        self.mdm = 0.0
        for fn in files:
            mt = float(np.asarray(io.read_header(fn)["MassTable"])[io.PT_DM])
            if mt > 0:
                self.mdm = mt
                break
        if not self.mdm > 0:
            raise ValueError("PartType1 has no MassTable mass: the core track needs equal-mass DM")
        self.box = box
        self._order = None
        self.tree = None

    def index_of(self, ids):
        if self._order is None:
            self._order = np.argsort(self.ids, kind="stable")
        p = np.clip(np.searchsorted(self.ids, ids, sorter=self._order), 0, len(self.ids) - 1)
        hit = self.ids[self._order[p]] == ids
        return self._order[p[hit]]

    def set_region(self, centre, half):
        """k-d tree of the DM in a cube of half-side ``half`` about ``centre``."""
        from scipy.spatial import cKDTree

        d = periodic(self.pos - centre.astype(np.float32), self.box)
        sel = np.nonzero(np.all(np.abs(d) < half, axis=1))[0]
        self.rc = centre.astype(np.float64)
        self.ridx = sel
        self.rpos = d[sel].astype(np.float64)
        self.tree = cKDTree(self.rpos)

    def rel(self, p):
        return periodic(np.asarray(p, np.float64) - self.rc, self.box)


class _Cat:
    """The few catalogue fields the track needs, plus file bookkeeping."""

    def __init__(self, output_dir, num):
        from . import catalog

        self.num = num
        self.cat = catalog.Catalog(output_dir, num)
        self.files = io.group_files(output_dir, num)
        self.nsub = self.cat.n_sub
        self.time = float(self.cat.a)
        self.z = float(self.cat.z)

    def sub(self, key):
        return self.cat.s(key)

    def grp(self, key):
        return self.cat.g(key)

    def dm_owner(self, idx):
        """Subhalo owning each global DM index (-1 if none)."""
        if self.sub("offsettype") is None or self.sub("lentype") is None:
            raise ValueError("catalogue %d has no SubhaloOffsetType/LenType: the core track needs group-ordered "
                             "snapshots" % self.num)
        off = np.asarray(self.sub("offsettype"))[:, io.PT_DM].astype(np.int64)
        ln = np.asarray(self.sub("lentype"))[:, io.PT_DM].astype(np.int64)
        srt = np.argsort(off, kind="stable")
        j = np.searchsorted(off[srt], idx, side="right") - 1
        s = srt[np.clip(j, 0, None)]
        return np.where((j >= 0) & (idx < off[s] + ln[s]), s, -1)

    def locate(self, index, table):
        """(file, index within it) of a global Subhalo / Group index."""
        key = "Nsubhalos_ThisFile" if table == "Subhalo" else "Ngroups_ThisFile"
        off = 0
        for i, fn in enumerate(self.files):
            n = int(io.read_header(fn).get(key, 0))
            if index < off + n:
                return i, index - off
            off += n
        raise IndexError("%s %d not in catalogue %d" % (table, index, self.num))


def _rho_thr_dm(units, a):
    """200 rho_crit(z) x f_DM in comoving code units."""
    rho_crit0 = 3.0 * units.Hubble ** 2 / (8.0 * np.pi * units.G)
    e2 = units.Omega0 / a ** 3 + units.OmegaLambda
    return 200.0 * rho_crit0 * e2 * a ** 3 * (units.Omega0 - units.OmegaBaryon) / units.Omega0


def dm_peak(dm, c0, r0=PEAK_R0, nmin=PEAK_NMIN):
    c = np.asarray(c0, np.float64)
    for _ in range(3):
        idx = dm.tree.query_ball_point(c, r0)
        if len(idx) < nmin:
            r0 *= 2
            continue
        c = shrink(dm.rpos[idx], c, r0, nmin)
        break
    return c


def so_mass(dm, c, rho_thr):
    """SO DM mass and radius about ``c`` (region coordinates)."""
    rmax = SO_RMAX0
    while True:
        idx = dm.tree.query_ball_point(c, rmax)
        if len(idx) < 20:
            return 0.0, 0.0
        r = np.sort(np.sqrt(((dm.rpos[idx] - c) ** 2).sum(axis=1)))
        m = dm.mdm * np.arange(1, len(r) + 1)
        above = np.nonzero(m / (4.0 / 3.0 * np.pi * np.maximum(r, 1e-3) ** 3) >= rho_thr)[0]
        if len(above) == 0:
            return 0.0, 0.0
        k = above[-1]
        if r[k] < 0.7 * rmax or rmax > 5000:
            return float(m[k]), float(r[k])
        rmax *= 2


def _halo_state(dm, centre_rel, rho_thr):
    m200, r200 = so_mass(dm, centre_rel, rho_thr)
    r200 = r200 if r200 > 0 else 5.0
    idx_t = np.array(dm.tree.query_ball_point(centre_rel, TFRAC * r200), dtype=np.int64)
    _, idx_c = dm.tree.query(centre_rel, k=min(NCORE, len(dm.rpos)))
    idx_c = np.atleast_1d(idx_c)
    return dict(m200=m200, r200=r200, t_ids=dm.ids[dm.ridx[idx_t]], core_ids=dm.ids[dm.ridx[idx_c]],
                centre=dm.rc + centre_rel)


def _subhalo_at(dm, cat, centre_rel):
    idx = dm.tree.query_ball_point(centre_rel, LABEL_RAD)
    if idx:
        own = cat.dm_owner(dm.ridx[np.array(idx)])
        own = own[own >= 0]
        if len(own):
            return int(np.bincount(own).argmax())
    p = periodic(np.asarray(cat.sub("pos"), dtype=np.float64) - (dm.rc + centre_rel), dm.box)
    return int(np.argmin((p ** 2).sum(axis=1)))


def _step_back(dm, cat, state, rho_thr):
    """Progenitor candidates of the halo in ``state`` at this earlier anchor."""
    box = dm.box
    t_idx = dm.index_of(state["t_ids"])
    c_idx = dm.index_of(state["core_ids"])
    if len(t_idx) == 0 or len(c_idx) == 0:
        return None
    half = max(4 * state["r200"], 50.0) + 100.0
    dm.set_region(state["centre"], half)
    t_rel = periodic(dm.pos[t_idx].astype(np.float64) - dm.rc, box)
    c_rel = periodic(dm.pos[c_idx].astype(np.float64) - dm.rc, box)
    med = np.median(c_rel, axis=0)
    core_seed = shrink(c_rel, c_rel.mean(axis=0) if len(c_rel) < 3 else med,
                       np.sqrt(((c_rel - med) ** 2).sum(axis=1)).max() + 1e-3, max(5, len(c_rel) // 5))
    seeds = [core_seed]
    own = cat.dm_owner(t_idx)
    cnt = np.bincount(own[own >= 0], minlength=max(cat.nsub, 1))
    spos = np.asarray(cat.sub("pos"))
    for s in np.nonzero(cnt >= SEED_SHARE * len(t_idx))[0]:
        p = dm.rel(spos[s])
        if np.all(np.abs(p) < half - 20):
            seeds.append(p)
    peaks = []
    for sd in seeds:
        pk = dm_peak(dm, sd)
        if all(np.linalg.norm(pk - q) > MERGE_DIST for q in peaks):
            peaks.append(pk)
    peaks = np.array(peaks)
    core_peak = int(np.argmin(((peaks - dm_peak(dm, core_seed)) ** 2).sum(axis=1)))
    so = [so_mass(dm, pk, rho_thr) for pk in peaks]
    r200 = np.array([max(r, 1.0) for _, r in so])
    dist = np.sqrt(((t_rel[:, None, :] - peaks[None, :, :]) ** 2).sum(axis=2))
    dist = np.where(dist <= r200[None, :], dist, np.inf)
    near = np.argmin(dist, axis=1)
    inside = np.isfinite(dist[np.arange(len(near)), near])
    share = np.bincount(near[inside], minlength=len(peaks)) / len(t_idx)
    cand = sorted(set(np.nonzero(share >= MIN_SHARE)[0].tolist()) | {core_peak})
    return [dict(peak_rel=peaks[i], share=float(share[i]), is_core=(i == core_peak)) for i in cand]


def _core_id_ranks(files, centre, radius, box):
    """IDs of the gas, DM and stars within ``radius`` of ``centre``, sorted by ID,
    with their distance rank scaled to [0, 1)."""
    ids, dist = [], []
    for t in (io.PT_GAS, io.PT_DM, io.PT_STAR):
        for _, d in io.iter_type(files, t, ("ParticleIDs", "Coordinates")):
            x = periodic(d["Coordinates"] - centre.astype(np.float32), box)
            sel = np.nonzero(np.all(np.abs(x) < radius, axis=1))[0]
            r = np.sqrt((x[sel].astype(np.float64) ** 2).sum(axis=1))
            inside = r < radius
            ids.append(d["ParticleIDs"][sel[inside]])
            dist.append(r[inside])
    ids, dist = np.concatenate(ids), np.concatenate(dist)
    rank = np.empty(len(dist))
    rank[np.argsort(dist, kind="stable")] = np.arange(len(dist)) / max(len(dist), 1)
    o = np.argsort(ids, kind="stable")
    return ids[o], rank[o]


def _rank_of(ids_sorted, rank_sorted, query):
    if len(ids_sorted) == 0:
        return np.full(len(query), np.inf)
    p = np.clip(np.searchsorted(ids_sorted, query), 0, len(ids_sorted) - 1)
    return np.where(ids_sorted[p] == query, rank_sorted[p], np.inf)


def _tree_field(output_dir, kind, num, key):
    """Subhalo/<key> of the ``subhalo_desc`` / ``subhalo_prog`` files of a snapshot, or None."""
    n = "%03d" % num
    files = []
    for pat in (os.path.join(output_dir, "groups_%s" % n, "%s_%s.*.hdf5" % (kind, n)),
                os.path.join(output_dir, "groups_%s" % n, "%s_%s.hdf5" % (kind, n)),
                os.path.join(output_dir, "%s_%s.hdf5" % (kind, n)),
                os.path.join(output_dir, "%s_%s.*.hdf5" % (kind, n))):
        files = sorted(glob.glob(pat), key=io._file_chunk_index)
        if files:
            break
    out = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            if "Subhalo" in f and key in f["Subhalo"]:
                out.append(f["Subhalo"][key][:])
    return np.concatenate(out) if out else None


def _linked(output_dir, s, sub):
    cand = set()
    for key in ("DescSubhaloNr", "FirstDescSubhaloNr"):
        d = _tree_field(output_dir, "subhalo_desc", s, key)
        if d is not None and sub < len(d) and d[sub] >= 0:
            cand.add(int(d[sub]))
    for key in ("ProgSubhaloNr", "FirstProgSubhaloNr"):
        p = _tree_field(output_dir, "subhalo_prog", s + 1, key)
        if p is not None:
            cand |= set(np.nonzero(p == sub)[0].tolist())
    return cand


def track(output_dir, end=None, group=0, verbose=True):
    """Follow FoF ``group``'s central at snapshot ``end`` (default: the last with
    particles) back in time.  Returns a dict of per-snapshot arrays plus
    ``decisions`` (merger decisions, as text) and ``notes``."""
    say = print if verbose else (lambda *a, **k: None)
    anchors = io.available_snapshots(output_dir)
    if end is not None:
        anchors = [s for s in anchors if s <= end]
    if not anchors:
        raise ValueError("no snapshots with particle data")
    end = anchors[-1]
    snap_end = io.Snapshot(output_dir, end)
    units, box = snap_end.units, snap_end.boxsize
    cats = {}

    def cat(s):
        if s not in cats:
            cats[s] = _Cat(output_dir, s)
        return cats[s]

    cend = cat(end)
    if cend.sub("offsettype") is None:
        raise ValueError("catalogue %d has no SubhaloOffsetType: the core track needs group-ordered snapshots" % end)
    sub0 = int(np.asarray(cend.grp("firstsub"))[group])
    if sub0 < 0:
        raise ValueError("group %d has no subhalos at snapshot %d" % (group, end))
    track_, method, info = {}, {}, {}
    dm = _DM(snap_end.files, box)
    dm.set_region(np.asarray(cend.sub("pos"))[sub0].astype(np.float64), FIRST_HALF)
    pk = dm_peak(dm, np.zeros(3))
    state = _halo_state(dm, pk, _rho_thr_dm(units, cend.time))
    track_[end], method[end] = sub0, 0
    info[end] = dict(centre=state["centre"], m200=state["m200"], r200=state["r200"], share=1.0, ncand=1)
    say("  [core track] snapshot %d, group %d, central subhalo %d; %d snapshots with particles"
        % (end, group, sub0, len(anchors)))

    lineages = [dict(state=state, hist={}, born=end)]
    decisions = []

    def resolve(s, why):
        best = max(lineages, key=lambda L: L["hist"][s]["m200"])
        if len(lineages) > 1:
            first = min(min(L["hist"]) for L in lineages)
            decisions.append("snap %d (%s): %s -> main branch for anchors %d..%d" % (
                s, why, " | ".join("%slineage from %d: M200DM %.3e" % ("*" if L is best else " ", L["born"],
                                                                       L["hist"][s]["m200"]) for L in lineages),
                first, max(best["hist"])))
        for k, v in best["hist"].items():
            track_[k], method[k] = v["sub"], 0
            info[k] = v
        best["hist"] = {}
        return [best]

    for s in reversed(anchors[:-1]):
        c = cat(s)
        if c.nsub == 0:
            say("  [core track] snapshot %d: no subhalos, stopping" % s)
            break
        dm = _DM(io.snapshot_files(output_dir, s), box)
        rho = _rho_thr_dm(units, c.time)
        new = []
        for L in lineages:
            prev = L["state"]["centre"]
            res = _step_back(dm, c, L["state"], rho)
            if res is None:
                continue
            for r in res:
                st = _halo_state(dm, r["peak_rel"], rho)
                rec = dict(centre=st["centre"], m200=st["m200"], r200=st["r200"], share=r["share"], ncand=len(res),
                           sub=_subhalo_at(dm, c, r["peak_rel"]),
                           step=float(np.linalg.norm(periodic(st["centre"] - prev, box))))
                if r["is_core"]:
                    L["state"] = st
                    L["hist"][s] = rec
                    new.append(L)
                else:
                    new.append(dict(state=st, hist={s: rec}, born=s))
        lineages = []
        for L in sorted(new, key=lambda L: -L["born"]):
            if all(np.linalg.norm(periodic(L["hist"][s]["centre"] - M["hist"][s]["centre"], box)) > MERGE_DIST
                   for M in lineages):
                lineages.append(L)
        if not lineages:
            say("  [core track] snapshot %d: tracers not found, stopping" % s)
            break
        lineages = sorted(lineages, key=lambda L: -L["hist"][s]["m200"])[:MAX_LINEAGES]
        if len(lineages) > 1:
            if all(np.linalg.norm(periodic(A["hist"][s]["centre"] - B["hist"][s]["centre"], box))
                   > A["hist"][s]["r200"] + B["hist"][s]["r200"]
                   for i, A in enumerate(lineages) for B in lineages[i + 1:]):
                lineages = resolve(s, "progenitors separate")
        else:
            lineages = resolve(s, "single")
        del dm
    if lineages and any(L["hist"] for L in lineages):
        lineages = resolve(min(min(L["hist"]) for L in lineages if L["hist"]), "end of data")
    done = sorted(track_)

    # catalogue-only snapshots between anchors
    offset = {s: -1.0 for s in done}
    idrank = {s: -1.0 for s in done}
    nums = io.available_group_catalogs(output_dir)
    gaps = [(a, b) for a, b in zip(done[:-1], done[1:]) if any(a < s < b for s in nums)]
    cores = {}
    for a, b in gaps:
        for s in (a, b):
            if s not in cores:
                rad = max(ID_RFRAC * info[s]["r200"], ID_RMIN)
                cores[s] = _core_id_ranks(io.snapshot_files(output_dir, s), info[s]["centre"], rad, box)
    for a, b in gaps:
        ca, cb = cat(a), cat(b)
        pa = info[a]["centre"]
        pb = pa + periodic(info[b]["centre"] - pa, box)
        cur = track_[a]
        for s in [x for x in nums if a < x < b]:
            cs = cat(s)
            t = (cs.time - ca.time) / (cb.time - ca.time)
            ptarget = pa + t * (pb - pa)
            r200 = info[a]["r200"] + t * (info[b]["r200"] - info[a]["r200"])
            pos = np.asarray(cs.sub("pos"), dtype=np.float64)
            dist = np.sqrt((periodic(pos - ptarget, box) ** 2).sum(axis=1))
            linked = {x for x in _linked(output_dir, s - 1, cur) if 0 <= x < cs.nsub}
            ndm_ref = NDM_FRAC * min(np.asarray(ca.sub("lentype"))[track_[a]][io.PT_DM],
                                     np.asarray(cb.sub("lentype"))[track_[b]][io.PT_DM])
            ndm = np.asarray(cs.sub("lentype"))[:, io.PT_DM].astype(np.float64)
            mb = cs.sub("mostbound")
            nxt = -1
            if mb is not None:
                rk = np.minimum(_rank_of(*cores[a], np.asarray(mb)), _rank_of(*cores[b], np.asarray(mb)))
                idc = np.nonzero(np.isfinite(rk) & (ndm >= ndm_ref))[0]
                if len(idc):
                    nxt = int(idc[np.argmin(rk[idc])])
                    track_[s], method[s], offset[s], idrank[s] = nxt, 3, float(dist[nxt]), float(rk[nxt])
                    info[s] = dict(centre=pos[nxt], m200=-1.0, r200=-1.0, share=-1.0, ncand=len(idc))
                    cur = nxt
                    continue
            near = np.nonzero(dist < max(FILL_R, 0.1 * r200))[0]
            best = -1
            if len(near):
                score = ndm[near] * np.where(np.isin(near, list(linked)), 1.05, 1.0)
                best = int(near[np.argmax(score)])
            if best >= 0 and ndm[best] >= ndm_ref:
                nxt, m = best, (1 if best in linked else 2)
            else:
                big = np.nonzero(ndm >= ndm_ref)[0]
                nxt, m = int(big[np.argmin(dist[big])]), 2
            track_[s], method[s], offset[s] = nxt, m, float(dist[nxt])
            info[s] = dict(centre=pos[nxt], m200=-1.0, r200=-1.0, share=-1.0, ncand=0)
            cur = nxt

    snaps = np.array(sorted(track_), dtype=np.int32)
    out = dict(snaps=snaps, z=np.array([cat(int(s)).z for s in snaps]),
               sub=np.array([track_[int(s)] for s in snaps], dtype=np.int64),
               grp=np.array([int(np.asarray(cat(int(s)).sub("groupnr"))[track_[int(s)]]) for s in snaps], dtype=np.int64),
               method=np.array([method[int(s)] for s in snaps], dtype=np.int8),
               centre=np.array([info[int(s)]["centre"] for s in snaps]),
               m200dm=np.array([info[int(s)]["m200"] for s in snaps]),
               r200=np.array([info[int(s)]["r200"] for s in snaps]),
               share=np.array([info[int(s)]["share"] for s in snaps]),
               ncand=np.array([info[int(s)]["ncand"] for s in snaps], dtype=np.int32),
               interp_offset=np.array([offset[int(s)] for s in snaps]),
               idrank=np.array([idrank.get(int(s), -1.0) for s in snaps]),
               decisions=decisions, end=end, group=group, cats=cats, box=box,
               fdm=(units.Omega0 - units.OmegaBaryon) / units.Omega0)
    nm = np.bincount(out["method"], minlength=4)
    say("  [core track] %d snapshots (%d-%d): %d anchors, %d most-bound ID match, %d tree link, %d position+mass"
        % (len(snaps), snaps[0], snaps[-1], nm[0], nm[3], nm[1], nm[2]))
    for d in decisions:
        say("  [core track] merger decision: " + d)
    return out


def write(tr, path):
    """Write a track in the ``tree.hdf5`` layout (+ the track's own columns)."""
    snaps = tr["snaps"]
    n = len(snaps)
    ds = {"Subhalo": {}, "Group": {}}
    for i, s in enumerate(snaps):
        c = tr["cats"][int(s)]
        for table, idx in (("Subhalo", int(tr["sub"][i])), ("Group", int(tr["grp"][i]))):
            fi, li = c.locate(idx, table)
            with h5py.File(c.files[fi], "r") as f:
                for key, d in f[table].items():
                    if key not in ds[table]:
                        shape = list(d.shape)
                        shape[0] = n
                        ds[table][key] = np.zeros(shape, dtype=d.dtype)
                    ds[table][key][i] = d[li]
    with h5py.File(path, "w") as ef:
        h = ef.create_group("Header")
        for k, v in dict(NCORE=NCORE, TFRAC=TFRAC, MIN_SHARE=MIN_SHARE, MERGE_DIST=MERGE_DIST, EndSnap=tr["end"],
                         EndGroup=tr["group"], BoxSize=tr["box"]).items():
            h.attrs[k] = v
        h.attrs["TrackMethod"] = "DM core + SO mass + position (thesan_tmz.coretrack, after Rahul's core_track.py)"
        h.attrs["Code"] = _code_fingerprint()
        ef.create_dataset("Snapshots", data=snaps)
        ef.create_dataset("Redshifts", data=tr["z"])
        ef.create_dataset("Method", data=tr["method"])
        ef.create_dataset("CentrePos", data=tr["centre"])
        ef.create_dataset("M200DM", data=tr["m200dm"])
        ef.create_dataset("M200", data=np.where(tr["m200dm"] > 0, tr["m200dm"] / tr["fdm"], -1.0))
        ef.create_dataset("R200", data=tr["r200"])
        ef.create_dataset("TShare", data=tr["share"])
        ef.create_dataset("NCandidates", data=tr["ncand"])
        ef.create_dataset("InterpOffset", data=tr["interp_offset"])
        ef.create_dataset("IDMatchRank", data=tr["idrank"])
        for table in ("Subhalo", "Group"):
            g = ef.create_group(table)
            g.create_dataset("%sID" % table, data=tr["sub"] if table == "Subhalo" else tr["grp"])
            for key, d in ds[table].items():
                g.create_dataset(key, data=d)
        ef.create_dataset("MergerDecisions", data=np.array(tr["decisions"] or [""], dtype="S"))
    return path


def summary_lines(path):
    """MAIN-GALAXY TRACK block for report.txt."""
    with h5py.File(path, "r") as f:
        snaps = np.asarray(f["Snapshots"])
        meth = np.asarray(f["Method"])
        sub = np.asarray(f["Subhalo/SubhaloID"])
        dec = [x.decode() for x in np.asarray(f["MergerDecisions"]) if x]
    nm = np.bincount(meth.astype(int), minlength=4)
    out = ["MAIN-GALAXY TRACK (DM core, after Rahul's core_track.py): %s" % path,
           "  %d snapshots (%d-%d): %d with particles (DM peak + SO M200), %d most-bound ID match, %d tree link, "
           "%d position + mass" % (len(snaps), snaps[0], snaps[-1], nm[0], nm[3], nm[1], nm[2]),
           "  the tracked galaxy is not subhalo 0 at %d of them (where group 0's central is another galaxy)"
           % int(np.sum(sub != 0))]
    out += ["  merger decision: " + d for d in dec]
    return "\n".join(out)


def read(path):
    """{snapshot: (subhalo index, DM-peak centre in code units)} of a track file."""
    with h5py.File(path, "r") as f:
        snaps = np.asarray(f["Snapshots"])
        sub = np.asarray(f["Subhalo/SubhaloID"])
        cen = np.asarray(f["CentrePos"])
    return {int(s): (int(j), c) for s, j, c in zip(snaps, sub, cen)}


def load_or_build(output_dir, figdir, verbose=True):
    """The track of this run, from ``<figdir>/core_track.hdf5`` when it covers the
    run's last particle snapshot and was made by this code, else built and
    written there.  An out-of-date file is kept as
    ``core_track_end<NNN>_<code>.hdf5``, never overwritten.  Returns
    ({snapshot: (subhalo, centre)}, path), or (None, reason)."""
    anchors = io.available_snapshots(output_dir)
    if not anchors:
        return None, "no snapshots with particle data"
    path = os.path.join(figdir, TRACK_FILE)
    if os.path.exists(path):
        try:
            with h5py.File(path, "r") as f:
                end = int(f["Header"].attrs["EndSnap"])
                code = str(f["Header"].attrs.get("Code", ""))
        except (OSError, KeyError) as exc:
            return None, "cannot read %s: %s" % (path, exc)
        if end == anchors[-1] and code == _code_fingerprint():
            return read(path), path
        why = "the run has new snapshots" if end != anchors[-1] else "the tracking code changed"
        kept = os.path.join(figdir, "core_track_end%03d_%s.hdf5" % (end, code or "old"))
        if os.path.exists(kept):
            return None, "%s is out of date (%s) and %s already exists; move one of them" % (path, why, kept)
        shutil.move(path, kept)      # never overwritten: the old track is kept
        if verbose:
            print("  [core track] %s; previous track kept as %s" % (why, kept))
    os.makedirs(figdir, exist_ok=True)
    tr = track(output_dir, verbose=verbose)
    write(tr, path)
    if verbose:
        print("  [core track] wrote %s" % path)
    return read(path), path
