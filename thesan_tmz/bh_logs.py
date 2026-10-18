"""
Parsers for the ASCII logs this branch writes.

Every format below was read off the ``fprintf`` that produces it, not guessed.
Line numbers refer to arepo ``Arepo2-TMZ-newbh`` at ``0b41480f0b``.
The source line is quoted next to each parser so you can re-check it when the
code changes.

    blackholes.txt                    src/blackhole/blackhole.cc:227
    blackhole_details/blackhole_details_<task>.txt
                                      src/blackhole/blackhole_mdot.cc:431   (6 cols, default)
                                      src/blackhole/blackhole_mdot.cc:521   (8 cols, alpha disc)
                                      src/blackhole/blackhole_mdot.cc:637   (7 cols, torque)
    blackhole_bipolar/blackhole_bipolar_<task>.txt
                                      src/blackhole/blackhole_mdot.cc:438
    blackhole_mergers/blackhole_mergers_<task>.txt
                                      src/blackhole/blackhole_mergers.cc:970/1101
    sfr.txt                           src/cooling_sfr/starformation.cc:700
    energy.txt                        src/logs/global.cc:92
    stdout                            src/blackhole/blackhole_bipolar_injection.cc:700/705

A note on the per-task files: AREPO writes one file *per MPI rank*, and a given
black hole migrates between ranks as the domain decomposition changes.  So you
must concatenate every rank's file and then group by ID -- reading only
``blackhole_details_0.txt`` gives you a ragged, misleading subset.  All the
readers here glob the whole directory by default.

Timing caveat: these logs are written on every *active* timestep of each black
hole, not on a global cadence.  Rows are therefore unevenly spaced in time and
denser for BHs on short timebins.  Any time-average must be weighted by dt --
:func:`bin_details_by_time` does this for you.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Column layouts
# ---------------------------------------------------------------------------

#: blackholes.txt -- one row per global timestep, written by rank 0.
BLACKHOLES_COLS = [
    "time",  # All.Time (a for a comoving run)
    "n_bh",  # All.TotNumBHs
    "mass_bh",  # sum of BH_Mass, code units (subgrid mass)
    "mdot",  # sum of BH_Mdot, code units
    "mdot_msun_per_yr",  # the same, converted by the code itself
    "mass_real",  # sum of P[].Mass  (dynamical mass)
    "mdot_eddington",  # sum of Mdot/Mdot_Edd
]

#: blackhole_details, default (Bondi / freefall / ...) -- mdot.cc:431
DETAILS_COLS_DEFAULT = ["id", "time", "bh_mass", "mdot", "rho", "csnd"]

#: blackhole_details with BH_ALPHA_DISK_ACCRETION -- mdot.cc:521.
#: Same six columns as the default, plus Omega and the Toomre Q.
DETAILS_COLS_ALPHA = DETAILS_COLS_DEFAULT + ["omega", "toomre_q"]

#: blackhole_details with BH_TORQUE_LIMITED_ACCRETION -- mdot.cc:637.
#: NOTE: this one does *not* write the sound speed -- f_disk and f_gas take
#: the place of the csnd column, so it is 7 columns, not 8.
DETAILS_COLS_TORQUE = ["id", "time", "bh_mass", "mdot", "rho", "f_disk", "f_gas"]

#: blackhole_bipolar -- mdot.cc:438
BIPOLAR_COLS = [
    "id",
    "time",
    "jx",
    "jy",
    "jz",  # angular momentum of the neighbour gas = jet axis
    "bipolar_sum",  # normalisation of the cone weight sum
    "cold_flag",  # 1 if ColdFraction > BHBipolarColdFraction
    "cold_disk",  # BH_BipolarColdDisk
    "cold_fraction",  # BH_BipolarColdFraction
]

#: blackhole_mergers -- mergers.cc:970
MERGER_COLS = ["task", "time", "id_survivor", "mass_survivor", "id_swallowed", "mass_swallowed"]

#: sfr.txt -- starformation.cc:700
SFR_COLS = [
    "time",
    "total_sm",  # instantaneous stellar mass formed this step (code units)
    "sfr_code",  # total SFR in code units
    "sfr_msun_per_yr",  # the same, converted
    "mass_stars",  # total stellar mass on the fly
    "cum_mass_stars",  # cumulative stellar mass formed
]

#: energy.txt -- global.cc:92.  30 columns.
ENERGY_COLS = (
    ["time", "e_int", "e_pot", "e_kin"]
    + ["%s_type%d" % (k, t) for t in range(6) for k in ("e_int", "e_pot", "e_kin")]
    + ["mass_type%d" % t for t in range(6)]
    + ["e_injected", "e_mag"]
)


# ---------------------------------------------------------------------------
# Generic reading
# ---------------------------------------------------------------------------


def _read_numeric(path_or_paths, ncol_expected=None, id_prefix=None):
    """Read whitespace-separated numbers, tolerating partially written lines.

    AREPO flushes but does not lock these files; a run that is still going, or
    that was killed, routinely leaves a truncated final line.  We therefore
    parse line by line and drop anything that does not have the right number of
    parseable floats, reporting how many were dropped.

    ``id_prefix`` strips a leading tag such as ``BH=`` from the first column.
    """
    paths = [path_or_paths] if isinstance(path_or_paths, str) else list(path_or_paths)
    rows, n_bad = [], 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if id_prefix and line.startswith(id_prefix):
                    line = line[len(id_prefix) :]
                parts = line.split()
                if ncol_expected is not None and len(parts) != ncol_expected:
                    n_bad += 1
                    continue
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    n_bad += 1
    if n_bad:
        print("  [bh_logs] dropped %d malformed line(s) from %d file(s)" % (n_bad, len(paths)))
    if not rows:
        return np.zeros((0, ncol_expected or 0))
    return np.asarray(rows, dtype=np.float64)


def _as_dict(arr, cols):
    return {c: arr[:, i] for i, c in enumerate(cols)} if arr.size else {c: np.zeros(0) for c in cols}


def _sort_by_time(d, key="time"):
    if d[key].size == 0:
        return d
    o = np.argsort(d[key], kind="stable")
    return {k: (v[o] if isinstance(v, np.ndarray) and v.shape[:1] == o.shape else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# blackholes.txt
# ---------------------------------------------------------------------------


def read_blackholes_txt(output_dir):
    """Global BH summary, one row per global step.  See BLACKHOLES_COLS."""
    path = os.path.join(output_dir, "blackholes.txt")
    arr = _read_numeric(path, ncol_expected=len(BLACKHOLES_COLS))
    d = _as_dict(arr, BLACKHOLES_COLS)
    return _sort_by_time(d)


# ---------------------------------------------------------------------------
# blackhole_details
# ---------------------------------------------------------------------------


def detect_details_layout(output_dir):
    """Return ('default'|'alpha'|'torque', columns) by counting columns.

    The three accretion models write 6, 8 and 7 columns respectively, so the
    layout is unambiguous from the file alone -- no need to know which
    Config.sh options were on.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "blackhole_details", "blackhole_details_*.txt")))
    for p in files:
        with open(p, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("BH="):
                    n = len(line.split())
                    if n == 6:
                        return "default", DETAILS_COLS_DEFAULT
                    if n == 8:
                        return "alpha", DETAILS_COLS_ALPHA
                    if n == 7:
                        return "torque", DETAILS_COLS_TORQUE
                    raise ValueError("unexpected blackhole_details column count: %d" % n)
    return "default", DETAILS_COLS_DEFAULT


def read_bh_details(output_dir, layout=None):
    """Concatenate every rank's ``blackhole_details_*.txt``.

    Returns a dict of arrays with the columns for the detected layout, sorted
    by (id, time).  ``layout`` may be forced to 'default' / 'alpha' / 'torque'.
    """
    if layout is None:
        layout, cols = detect_details_layout(output_dir)
    else:
        cols = {"default": DETAILS_COLS_DEFAULT, "alpha": DETAILS_COLS_ALPHA, "torque": DETAILS_COLS_TORQUE}[layout]

    files = sorted(glob.glob(os.path.join(output_dir, "blackhole_details", "blackhole_details_*.txt")))
    arr = _read_numeric(files, ncol_expected=len(cols), id_prefix="BH=")
    d = _as_dict(arr, cols)
    d["_layout"] = layout
    if d["time"].size:
        o = np.lexsort((d["time"], d["id"]))
        for k in cols:
            d[k] = d[k][o]
    return d


def read_bh_bipolar(output_dir):
    """Concatenate every rank's ``blackhole_bipolar_*.txt``.

    Only written when BH_BIPOLAR_FEEDBACK is compiled in.  Adds two derived
    fields: ``j_mag`` (|J| of the neighbour gas) and ``jhat`` (unit axis).
    """
    files = sorted(glob.glob(os.path.join(output_dir, "blackhole_bipolar", "blackhole_bipolar_*.txt")))
    arr = _read_numeric(files, ncol_expected=len(BIPOLAR_COLS), id_prefix="BH=")
    d = _as_dict(arr, BIPOLAR_COLS)
    if d["time"].size:
        o = np.lexsort((d["time"], d["id"]))
        for k in BIPOLAR_COLS:
            d[k] = d[k][o]
        j = np.stack([d["jx"], d["jy"], d["jz"]], axis=1)
        mag = np.linalg.norm(j, axis=1)
        d["j_mag"] = mag
        with np.errstate(invalid="ignore", divide="ignore"):
            d["jhat"] = np.where(mag[:, None] > 0, j / mag[:, None], 0.0)
    else:
        d["j_mag"] = np.zeros(0)
        d["jhat"] = np.zeros((0, 3))
    return d


def read_bh_mergers(output_dir):
    """Concatenate every rank's ``blackhole_mergers_*.txt``.

    Historical note: before the 2026-09-08 fix, the tree-walk branch of
    mergers.cc wrote its columns without separators ("...123450.001..."), so
    those rows are unparseable and are reported as malformed.  Logs written
    after the fix parse cleanly.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "blackhole_mergers", "blackhole_mergers_*.txt")))
    arr = _read_numeric(files, ncol_expected=len(MERGER_COLS))
    return _sort_by_time(_as_dict(arr, MERGER_COLS))


# ---------------------------------------------------------------------------
# Per-BH views
# ---------------------------------------------------------------------------


def unique_bh_ids(details):
    return np.unique(details["id"]).astype(np.int64) if details["id"].size else np.zeros(0, dtype=np.int64)


def select_bh(details, bh_id):
    """Rows of one black hole, in time order."""
    m = details["id"] == bh_id
    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == m.size else v) for k, v in details.items()}


def most_massive_bh_id(details):
    """ID of the black hole that reaches the largest ``BH_Mass``."""
    ids = unique_bh_ids(details)
    if ids.size == 0:
        return None
    peak = [details["bh_mass"][details["id"] == i].max() for i in ids]
    return int(ids[int(np.argmax(peak))])


def bin_details_by_time(details, bins, fields=("mdot",), weight_by_dt=True):
    """Average per-BH detail rows onto a regular time grid.

    Detail rows are written on each BH's own timestep, so a plain mean
    over-weights black holes on short timebins.  With ``weight_by_dt`` the
    weight of each row is the gap to the next row *of the same black hole*,
    which is the right thing for a duty-cycle-style average.

    Returns ``{'time': centres, field: values, 'count': n_rows_per_bin}``.
    """
    bins = np.asarray(bins, dtype=np.float64)
    centres = 0.5 * (bins[1:] + bins[:-1])
    out = {"time": centres, "count": np.zeros(centres.size)}
    if details["time"].size == 0:
        for f in fields:
            out[f] = np.full(centres.size, np.nan)
        return out

    t = details["time"]
    if weight_by_dt:
        w = np.ones_like(t)
        for i in unique_bh_ids(details):
            m = np.where(details["id"] == i)[0]
            tm = t[m]
            dt = np.diff(tm)
            # Last row of each BH inherits the previous gap.
            w[m] = np.append(dt, dt[-1] if dt.size else 1.0)
        w = np.maximum(w, 0.0)
    else:
        w = np.ones_like(t)

    idx = np.digitize(t, bins) - 1
    ok = (idx >= 0) & (idx < centres.size)
    out["count"] = np.bincount(idx[ok], minlength=centres.size).astype(float)
    wsum = np.bincount(idx[ok], weights=w[ok], minlength=centres.size)
    for f in fields:
        v = details[f]
        num = np.bincount(idx[ok], weights=(w * v)[ok], minlength=centres.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f] = np.where(wsum > 0, num / wsum, np.nan)
    return out


# ---------------------------------------------------------------------------
# sfr.txt / energy.txt
# ---------------------------------------------------------------------------


def read_sfr_txt(output_dir):
    path = os.path.join(output_dir, "sfr.txt")
    arr = _read_numeric(path, ncol_expected=len(SFR_COLS))
    return _sort_by_time(_as_dict(arr, SFR_COLS))


def read_energy_txt(output_dir):
    path = os.path.join(output_dir, "energy.txt")
    arr = _read_numeric(path, ncol_expected=len(ENERGY_COLS))
    if arr.size == 0:  # some builds omit the magnetic column
        arr = _read_numeric(path, ncol_expected=len(ENERGY_COLS) - 1)
        if arr.size:
            return _sort_by_time(_as_dict(arr, ENERGY_COLS[:-1]))
    return _sort_by_time(_as_dict(arr, ENERGY_COLS))


def agn_energy_sources(hist, logs, units):
    """Cumulative AGN feedback energy of the whole box, from the best source the
    build provides.  Everything is a box total, like ``blackholes.txt``.

    Where the energy is recorded depends on the build (checked against the C++):

      * LOCAL_INJECTION_BLACK_HOLES + LOCAL_INJECTION_BH_LEDGER -- the per-step
        ledger (local_injection.cc:926) has every channel, both what left the
        reservoirs ("released") and what reached the gas ("deposited", after the
        5e9 K ceiling and the kick velocity cap).  The complete record.
      * BH_BIPOLAR_INJECTION_* without local injection -- the stdout counters
        (cone pass) give the collimated channels; the isotropic channel, if
        BH_THERMALFEEDBACK is on, is in the snapshots' BH_CumEgyInjection_QM.
      * otherwise -- BH_CumEgyInjection_QM (+ _RM) summed over **all** BHs in
        each snapshot (``hist['bh_cum_egy_all_code']``).

    BH_CumEgyInjection_QM is incremented only by the isotropic channel
    (blackhole_mdot.cc:250, under BH_THERMALFEEDBACK); the bipolar channels never
    reach a snapshot field.  energy.txt's ``e_injected`` is All.EgyInjection, a
    mesh-refinement bookkeeping term unrelated to AGN feedback.

    Returns a dict:
      ``released``   ``(t_gyr, e_erg, label)`` or None
      ``deposited``  ``(t_gyr, e_erg, label)`` or None (ledger only)
      ``components`` list of ``(label, t_gyr, e_erg)``, per channel where known
      ``source``     short name of the source used
    """
    out = {"released": None, "deposited": None, "components": [], "source": None}
    E = units.energy_to_erg
    led = (logs or {}).get("ledger") or {}
    inj = (logs or {}).get("injection") or {}

    snap_t = snap_e = None
    if hist is not None and "bh_cum_egy_all_code" in hist and np.isfinite(hist["bh_cum_egy_all_code"]).any():
        ok = np.isfinite(hist["bh_cum_egy_all_code"])
        snap_t, snap_e = hist["age_gyr"][ok], hist["bh_cum_egy_all_code"][ok] * E

    if np.size(led.get("time", [])):
        t = units.age_gyr(led["time"])
        parts = [(name, np.cumsum(np.nan_to_num(led[k])) * E)
                 for name, k in (("isotropic", "drained_iso"), ("collimated thermal", "drained_coll"),
                                 ("kinetic", "drained_kin"))]
        out["components"] = [("%s (ledger)" % n, t, e) for n, e in parts if e[-1] > 0]
        out["released"] = (t, sum(e for _, e in parts), "released (LOCAL_INJECTION BH LEDGER)")
        out["deposited"] = (t, np.cumsum(np.nan_to_num(led["therm_applied"]) + np.nan_to_num(led["kin_applied"])) * E,
                            "deposited in the gas (ledger)")
        out["source"] = "ledger"
        return out

    if np.size(inj.get("time", [])):
        t = units.age_gyr(inj["time"])
        coll = np.cumsum(np.nan_to_num(inj["thermal"]) + np.nan_to_num(inj["kinetic"])) * E
        out["components"].append(("collimated (stdout counters)", t, coll))
        total, label = coll, "released (stdout counters"
        if snap_t is not None and snap_e[-1] > 0:
            iso = np.interp(t, snap_t, snap_e, left=0.0)
            out["components"].append(("isotropic (BH_CumEgyInjection_QM, all BHs)", snap_t, snap_e))
            total, label = coll + iso, label + " + BH_CumEgyInjection_QM"
        out["released"] = (t, total, label + ")")
        out["source"] = "counters"
        return out

    if snap_t is not None:
        out["released"] = (snap_t, snap_e, "released (BH_CumEgyInjection_QM/RM, all BHs)")
        out["components"].append(("isotropic (BH_CumEgyInjection_QM, all BHs)", snap_t, snap_e))
        out["source"] = "snapshots"
    return out


def expected_agn_energy(t_gyr, blackholes, params, config, units):
    """What the parameters predict for the cumulative released energy.

    Integrates Mdot c^2 from ``blackholes.txt`` (box total) and multiplies by
    eps_r times each channel's coupling efficiency, as blackhole_mdot.cc fills
    the reservoirs:

      * isotropic (BH_THERMALFEEDBACK[_ACC]):  eps_f   = BlackHoleFeedbackFactor
      * collimated thermal / kinetic:          eps_th  = BHBipolarInjThermalEfficiency,
                                               eps_kin = BHBipolarInjKineticEfficiency
        With BH_BIPOLAR_INJECTION_MODE_SWITCH each BH-step feeds one of the two,
        so the prediction is a band between them; without, both are fed.

    No defaults: a missing parameter means no prediction (None), never a guessed
    one.  With BH_VARIABLE_RADIATIVE_EFFICIENCY eps_r varies per step and the
    prediction uses BlackHoleRadiativeEfficiency, so it is only indicative.

    Returns ``{'lo': e_erg, 'hi': e_erg, 'label': str, 'note': str}`` on the
    ``t_gyr`` grid, or None.
    """
    from .units import CLIGHT, SEC_PER_YEAR, SOLAR_MASS

    params, cfg = params or {}, set(config or ())
    if not np.size(blackholes.get("time", [])) or "BlackHoleRadiativeEfficiency" not in params:
        return None
    eps_r = float(params["BlackHoleRadiativeEfficiency"])

    lo = hi = 0.0
    parts = []
    if {"BH_THERMALFEEDBACK", "BH_THERMALFEEDBACK_ACC"} & cfg:
        if "BlackHoleFeedbackFactor" not in params:
            return None
        lo = hi = float(params["BlackHoleFeedbackFactor"])
        parts.append("eps_f")
    eff = []
    for opt, key in (("BH_BIPOLAR_INJECTION_THERMAL", "BHBipolarInjThermalEfficiency"),
                     ("BH_BIPOLAR_INJECTION_KINETIC", "BHBipolarInjKineticEfficiency")):
        if opt in cfg:
            if key not in params:
                return None
            eff.append(float(params[key]))
    if eff:
        if "BH_BIPOLAR_INJECTION_MODE_SWITCH" in cfg:
            lo, hi = lo + min(eff), hi + max(eff)
            parts.append("eps_th or eps_kin")
        else:
            lo, hi = lo + sum(eff), hi + sum(eff)
            parts.append("eps_th + eps_kin")
    if not parts:
        return None

    tb = units.age_gyr(blackholes["time"])
    rate = blackholes["mdot_msun_per_yr"]
    acc = np.zeros_like(tb)
    acc[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tb) * 1.0e9)       # Msun accreted
    e_acc = np.interp(t_gyr, tb, acc) * SOLAR_MASS * CLIGHT ** 2                   # erg
    note = "eps_r = %g%s" % (eps_r, " (variable: indicative)" if "BH_VARIABLE_RADIATIVE_EFFICIENCY" in cfg else "")
    return {"lo": lo * eps_r * e_acc, "hi": hi * eps_r * e_acc,
            "label": r"expected: (%s) $\epsilon_r\int\dot M c^2 dt$" % " + ".join(parts), "note": note}


def agn_cumulative_energy(hist, inj, units):
    """Backward-compatible wrapper: ``(t_gyr, e_erg, label)`` of the released energy, or
    ``(None, None, None)``.  Prefer :func:`agn_energy_sources`, which also reads the ledger."""
    r = agn_energy_sources(hist, {"injection": inj}, units)["released"]
    return r if r is not None else (None, None, None)


# ---------------------------------------------------------------------------
# BH_BIPOLAR_INJECTION stdout diagnostics
# ---------------------------------------------------------------------------

#: Matches both the MODE_SWITCH and the plain variant of the per-step summary
#: printed at blackhole_bipolar_injection.cc:700 / :741.
_BIPOLAR_RX = re.compile(
    r"BH_BIPOLAR_INJECTION:\s*injected thermal\s*=\s*(?P<thermal>[-+0-9.eEnaif]+),\s*"
    r"kinetic\s*=\s*(?P<kinetic>[-+0-9.eEnaif]+)\s*\(code units\);\s*"
    r"(?:mode split \(thermal/kinetic\)\s*=\s*(?P<n_thermal>\d+)/(?P<n_kinetic>\d+);\s*)?"
    r"no-axis\s*=\s*(?P<no_axis>\d+),\s*"
    r"cold-disk fallback\s*=\s*(?P<cold_fallback>\d+),\s*"
    r"empty cone\s*=\s*(?P<empty_cone>\d+),\s*"
    r"clipped cells\s*=\s*(?P<clipped>\d+);\s*"
    r"took\s*(?P<seconds>[-+0-9.eE]+)\s*sec"
)

#: AREPO prints "Sync-Point <n>, Time: <t>, Redshift: <z>, ..." before each step,
#: which is how we attach a time to each injection summary.
_SYNC_RX = re.compile(r"Sync-Point\s+(\d+),\s*Time:\s*([-+0-9.eE]+)(?:,\s*Redshift:\s*([-+0-9.eE]+))?")


#: LOCAL_INJECTION_BH_LEDGER, printed once per step by rank 0 after the BH deposit
#: (local_injection.cc:926), MPI-reduced over all BHs:
#:   drained iso/coll/kin   energy taken out of the isotropic, collimated-thermal
#:                          and kinetic reservoirs this step
#:   thermal asked/applied  thermal energy the faces asked for / that reached the
#:                          gas after the 5e9 K ceiling
#:   kinetic asked/applied  kinetic energy asked for / applied after BHBipolarInjMaxVel
#:   |sum dp|/sum|dp|       net momentum of the kicks over their total (0 = balanced)
#: All energies in code units.
LEDGER_COLS = ["drained_iso", "drained_coll", "drained_kin", "therm_asked", "therm_applied",
               "kin_asked", "kin_applied", "dp_ratio"]
_NUM = r"([-+0-9.eEnaif]+)"
_LEDGER_RX = re.compile(
    r"LOCAL_INJECTION BH LEDGER:\s*drained iso\s+%s\s+coll\s+%s\s+kin\s+%s\s*\|\s*"
    r"thermal asked\s+%s\s+applied\s+%s\s*\|\s*kinetic asked\s+%s\s+applied\s+%s\s*\|\s*"
    r"\|sum dp\|/sum\|dp\|\s+%s" % ((_NUM,) * 8)
)


def scan_stdout(logfiles):
    """Read the run's stdout once and return both per-step BH records.

    Returns ``{'injection': ..., 'ledger': ...}``:

      * ``injection`` -- the BH_BIPOLAR_INJECTION counters printed by the cone
        pass (blackhole_bipolar_injection.cc:700/705): keys ``time``, ``sync``,
        ``thermal``, ``kinetic``, ``n_thermal``, ``n_kinetic``, ``no_axis``,
        ``cold_fallback``, ``empty_cone``, ``clipped``, ``seconds``
        (``n_thermal``/``n_kinetic`` NaN without MODE_SWITCH);
      * ``ledger`` -- the LOCAL_INJECTION BH LEDGER lines: ``time``, ``sync`` and
        :data:`LEDGER_COLS`.

    Each row gets the time of the preceding ``Sync-Point``.  ``logfiles`` may be
    one path or a list -- a long run is a restart chain with one stdout per
    job, so pass all of them (what :func:`find_stdout` returns).

    Which of the two a run has depends on its build: the counters only without
    LOCAL_INJECTION_BLACK_HOLES; the ledger only with it and
    LOCAL_INJECTION_BH_LEDGER.  Silence is the dangerous failure mode -- a BH that
    holds energy but never deposits it looks, in every other output, like a
    quiet BH -- so what was and was not found is always printed.
    """
    inj_keys = ["thermal", "kinetic", "n_thermal", "n_kinetic", "no_axis", "cold_fallback", "empty_cone", "clipped", "seconds"]
    inj = {k: [] for k in ["time", "sync"] + inj_keys}
    led = {k: [] for k in ["time", "sync"] + LEDGER_COLS}

    if logfiles is None:
        paths = []
    elif isinstance(logfiles, str):
        paths = [logfiles]
    else:
        paths = list(logfiles)
    paths = [p for p in paths if p and os.path.exists(p)]

    def arrays(rec):
        return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}

    if not paths:
        print("  [bh_logs] no stdout log found -- injection counters and energy ledger unavailable")
        return {"injection": arrays(inj), "ledger": arrays(led)}

    # Count the raw markers separately from the parsed rows.  If a marker is
    # present but nothing parses, the regex is out of date with the code -- a
    # completely different problem from the module never having printed.
    n_marker, n_sync, n_start, n_axis_only, n_led_marker = 0, 0, 0, 0, 0

    for path in paths:
        cur_time, cur_sync = np.nan, -1
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m_sync = _SYNC_RX.search(line)
                if m_sync:
                    n_sync += 1
                    cur_sync = int(m_sync.group(1))
                    cur_time = float(m_sync.group(2))
                    continue
                if "LOCAL_INJECTION BH LEDGER" in line:
                    n_led_marker += 1
                    m = _LEDGER_RX.search(line)
                    if m:
                        led["time"].append(cur_time)
                        led["sync"].append(cur_sync)
                        for k, v in zip(LEDGER_COLS, m.groups()):
                            led[k].append(float(v))
                    continue
                if "BH_BIPOLAR_INJECTION" not in line:
                    continue
                if "Start collimated feedback injection" in line:
                    n_start += 1
                    continue
                if "axis only (deposition handed to LOCAL_INJECTION_BLACK_HOLES)" in line:
                    n_axis_only += 1
                    continue
                if "injected thermal" in line:
                    n_marker += 1
                m = _BIPOLAR_RX.search(line)
                if not m:
                    continue
                g = m.groupdict()
                inj["time"].append(cur_time)
                inj["sync"].append(cur_sync)
                for k in inj_keys:
                    v = g.get(k)
                    inj[k].append(np.nan if v is None else float(v))

    out = {"injection": arrays(inj), "ledger": arrays(led)}
    n_parsed, n_led = out["injection"]["time"].size, out["ledger"]["time"].size

    print("  [bh_logs] stdout: %d file(s), %d Sync-Point lines, %d injection calls, %d counter summaries "
          "(%d parsed), %d ledger lines (%d parsed)"
          % (len(paths), n_sync, n_start, n_marker, n_parsed, n_led_marker, n_led))
    if n_marker and not n_parsed:
        print("  [bh_logs] WARNING: the counter line is present but did not parse -- the format in")
        print("            blackhole_bipolar_injection.cc has changed; update _BIPOLAR_RX.")
    if n_led_marker and not n_led:
        print("  [bh_logs] WARNING: the ledger line is present but did not parse -- the format in")
        print("            local_injection.cc has changed; update _LEDGER_RX.")
    if n_axis_only and not n_marker:
        print("  [bh_logs] LOCAL_INJECTION_BLACK_HOLES build (%d 'axis only' steps): no injection counters exist"
              % n_axis_only)
        print("            by design; %s" % ("the deposit is in the %d ledger lines." % n_led if n_led else
              "and no ledger either -- build with LOCAL_INJECTION_BH_LEDGER to record the deposit."))
    elif not n_marker and not n_led and n_sync:
        print("  [bh_logs] note: the run printed Sync-Points but neither injection counters nor a ledger --")
        print("            expected for isotropic kernel feedback; otherwise check that the stdout of the")
        print("            segment where the BHs were active is among these files.")
    return out


def parse_bipolar_stdout(logfiles):
    """The injection counters only (see :func:`scan_stdout`)."""
    return scan_stdout(logfiles)["injection"]


def find_stdout(run_dir, all_files=True):
    """Find the run's stdout file(s).

    Returns a list (newest last) when ``all_files``, else a single best guess.
    A cluster run is normally a restart chain with one file per job, and the
    injection counters for the epoch you care about may be in any of them, so
    the default is to hand every plausible file to the parser.
    """
    pats = ["*.out", "*.log", "slurm-*.out", "stdout*", "log*", "OUTPUT*"]
    hits = []
    for d in (run_dir, os.path.join(run_dir, "output"), os.path.join(run_dir, "logs")):
        for p in pats:
            hits += glob.glob(os.path.join(d, p))
    hits = sorted({h for h in hits if os.path.isfile(h)})
    if not hits:
        return [] if all_files else None

    # Keep only files that look like AREPO stdout, so a stray note.log or a
    # job script's own log does not get scanned.
    marked = []
    for h in hits:
        try:
            with open(h, "r", errors="replace") as fh:
                head = fh.read(2_000_000)
            if "BH_BIPOLAR_INJECTION" in head or "Sync-Point" in head:
                marked.append(h)
        except OSError:
            pass
    pool = marked or hits
    if not all_files:
        return max(pool, key=os.path.getsize)
    return sorted(pool, key=os.path.getmtime)
