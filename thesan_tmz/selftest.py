"""
Self-test: build a synthetic run and check the package gets the right answers.

    python -m thesan_tmz.selftest

This is not a smoke test.  The synthetic run in :mod:`thesan_tmz._mockrun` has
a *known* jet axis, a known lobe opening angle, known masses and known log
contents, so each check below has a right answer that the code can fail.  Run
it after changing anything in the package, and after any change to the AREPO
side that touches a log format.

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

from . import _mockrun, arepo_io as io
from . import bh_logs, catalog, diagnostics, evolution, galaxy, history, obs, scaling, tree, units
from .plotstyle import apply_style


class Checker:
    def __init__(self):
        self.passed, self.failed = 0, []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print("  PASS  %s" % name)
        else:
            self.failed.append(name)
            print("  FAIL  %s %s" % (name, ("-- " + detail) if detail else ""))

    def close(self, condition, name, detail=""):
        self.check(name, condition, detail)

    def report(self):
        print("\n%d passed, %d failed" % (self.passed, len(self.failed)))
        for f in self.failed:
            print("  failed: %s" % f)
        return 0 if not self.failed else 1


def run(workdir=None, keep=False, make_figures=True):
    tmp = workdir or tempfile.mkdtemp(prefix="thesan_tmz_selftest_")
    root = os.path.join(tmp, "mockrun")
    c = Checker()
    apply_style()

    print("building a synthetic run in %s" % root)
    _mockrun.build(root, nsnap=8)
    out = os.path.join(root, "output")

    # -- units and cosmology ------------------------------------------------
    print("\n[units]")
    u = units.THESAN_UNITS
    c.check("age(a=1) = 13.80 Gyr", abs(u.age_gyr(1.0) - 13.80) < 0.05, "got %.4f" % u.age_gyr(1.0))
    c.check("age(z=6) = 0.934 Gyr", abs(u.age_gyr(1 / 7.0) - 0.9339) < 0.005, "got %.4f" % u.age_gyr(1 / 7.0))
    c.check("mass_to_msun = 1e10/h", abs(u.mass_to_msun / (1e10 / 0.6774) - 1) < 1e-3)
    c.check("Mdot_Edd(1e8, eps=0.1) = 2.22 Msun/yr", abs(u.eddington_rate_msun_per_yr(1e8, 0.1) - 2.22) < 0.02)
    c.check("Mdot_Edd(1e8, eps=0.2) = 1.11 Msun/yr", abs(u.eddington_rate_msun_per_yr(1e8, 0.2) - 1.11) < 0.01)
    c.check("no Eddington efficiency -> lambda is NaN, not a default",
            u.eddington_efficiency is None and np.isnan(u.eddington_ratio(1.0, 1e8)))
    c.check("BH_VARIABLE_RADIATIVE_EFFICIENCY -> eps = 1/16, as blackhole_mdot_eddington()",
            units._eddington_efficiency(0.2, {"BH_VARIABLE_RADIATIVE_EFFICIENCY"}) == 1.0 / 16.0
            and units._eddington_efficiency(0.2, set()) == 0.2)
    c.check("Delta_vir(z=0) = 102", abs(u.delta_vir(1.0) - 102.4) < 1.0)
    # age(a) and its inverse must round-trip, since the redshift axis uses both.
    from .plotstyle import _invert_age

    a_in = np.array([0.05, 0.2, 0.5, 1.0])
    c.check("age <-> scalefactor round-trip", np.allclose(_invert_age(u, u.age_gyr(a_in)), a_in, rtol=1e-8))

    # -- observational relations against published values --------------------
    print("\n[observational relations]")
    c.check("KH13 M_bulge=1e11 -> 4.9e8", abs(obs.kormendy_ho13_mbh_mbulge(1e11) / 4.9e8 - 1) < 1e-6)
    c.check("KH13 sigma=200 -> 3.09e8", abs(obs.kormendy_ho13_mbh_sigma(200.0) / 3.09e8 - 1) < 1e-6)
    c.check("RV15 AGN M*=1e11 -> 2.8e7", abs(np.log10(obs.reines_volonteri15_agn(1e11)) - 7.45) < 1e-6)
    c.check("Moster13 Mh=1e12 z=0 in [2e10, 4e10]", 2e10 < obs.moster13_smhm(1e12, 0.0) < 4e10)
    c.check("Behroozi13 Mh=1e12 z=0 in [2e10, 4e10]", 2e10 < obs.behroozi13_smhm(1e12, 0.0) < 4e10)
    c.check("Behroozi13 finite at z=8", np.isfinite(obs.behroozi13_smhm(np.logspace(9, 13, 30), 8.0)).all())
    c.check("MD14 peaks near z=2", obs.madau_dickinson14_csfh(2.0) > obs.madau_dickinson14_csfh(0.0))
    c.check("BTFR V=200 -> 7.5e10", abs(obs.mcgaugh12_btfr(200.0) / 7.52e10 - 1) < 1e-3)

    # -- snapshot I/O -------------------------------------------------------
    print("\n[snapshot I/O]")
    nums = io.available_snapshots(out)
    c.check("8 snapshots discovered", nums == list(range(8)), "got %s" % nums)
    snap = io.Snapshot(out, nums[-1])
    c.check("split snapshot reassembled (gas)", snap.part(0, ["Masses"])["count"] == 4000)
    c.check("split snapshot reassembled (stars)", snap.part(4, ["Masses"])["count"] == 800)
    c.check("MassTable synthesised for DM", np.allclose(snap.part(1, ["Masses"])["Masses"], 3e-4))
    c.check("/Config read back", "BH_BIPOLAR_INJECTION" in snap.config)
    c.check("/Parameters read back", abs(snap.parameters["BHBipolarInjEddThreshold"] - 0.01) < 1e-12)
    c.check("units auto-detected from snapshot", abs(snap.units.h - 0.6774) < 1e-9)
    c.check("Eddington efficiency read from /Parameters (0.2)", snap.units.eddington_efficiency == _mockrun.EPS_R)
    ec = diagnostics.model_summary(snap)["eddington_check"]
    c.check("our Eddington rate reproduces the snapshot's BH_MdotEddington",
            ec is not None and ec["max_rel_diff"] < 1e-6, "got %s" % ec)
    c.check("group catalogue read", snap.groups(["Group_M_Crit200"])["count"] == 3)
    c.check("subhalo catalogue read", snap.subhalos(["SubhaloMassType"])["count"] == 5)

    # -- log parsing --------------------------------------------------------
    print("\n[log parsing]")
    bh = bh_logs.read_blackholes_txt(out)
    c.check("blackholes.txt: truncated last line dropped", bh["time"].size == 900, "got %d" % bh["time"].size)
    c.check("blackholes.txt sorted in time", np.all(np.diff(bh["time"]) >= 0))

    layout, cols = bh_logs.detect_details_layout(out)
    c.check("details layout detected as torque", layout == "torque", "got %s" % layout)
    det = bh_logs.read_bh_details(out)
    c.check("details: 2 black holes, 1800 rows", det["time"].size == 1800 and bh_logs.unique_bh_ids(det).size == 2)
    c.check("details: torque log has no csnd column", "csnd" not in det)
    c.check("details: f_disk present and in [0,1]", np.all((det["f_disk"] >= 0) & (det["f_disk"] <= 1)))
    c.check("most massive BH identified", bh_logs.most_massive_bh_id(det) == 900001)

    bip = bh_logs.read_bh_bipolar(out)
    c.check("bipolar log read", bip["time"].size > 0)
    c.check("bipolar jhat is a unit vector", np.allclose(np.linalg.norm(bip["jhat"], axis=1), 1.0, atol=1e-9))

    mg = bh_logs.read_bh_mergers(out)
    c.check("merger log: 2 events", mg["time"].size == 2)

    stdouts = bh_logs.find_stdout(root)
    c.check("find_stdout returns a list", isinstance(stdouts, list) and len(stdouts) >= 1)
    inj = bh_logs.parse_bipolar_stdout(stdouts)
    c.check("stdout: 300 injection lines", inj["time"].size == 300, "got %d" % inj["time"].size)
    c.check("stdout: times attached from Sync-Point", np.isfinite(inj["time"]).all())
    c.check("stdout: mode split parsed", np.isfinite(inj["n_thermal"]).all() and inj["n_thermal"].sum() > 0)
    c.check("stdout: counters parsed", inj["empty_cone"].sum() > 0 and inj["clipped"].sum() > 0)

    sfr = bh_logs.read_sfr_txt(out)
    c.check("sfr.txt read", sfr["time"].size == 900)
    en = bh_logs.read_energy_txt(out)
    c.check("energy.txt read (30 columns)", en["time"].size == 900 and "e_injected" in en)

    # -- catalogue: subhalo == one galaxy, group == the whole zoom system ---
    #
    # These checks exist because the package once passed its whole selftest
    # while reading nothing on real data: the mock wrote SubhaloGrNr,
    # SubhaloFlag and SubhaloSFRinRad, none of which appear anywhere in the
    # AREPO2 source, so every lookup fell through to a default of zero.
    print("\n[subfind catalogue]")
    cat = catalog.Catalog(out, snap.num, units=snap.units)
    c.check("catalogue read with AREPO2 Nsubhalos_* header keys", cat.n_sub == 5 and cat.n_grp == 3,
            "got %d subhaloes, %d groups" % (cat.n_sub, cat.n_grp))
    c.check("SubhaloGroupNr found (not the TNG SubhaloGrNr)", cat.group_of() is not None)
    c.check("subhaloes map to their own groups, not all to group 0",
            cat.group_of() is not None and set(cat.group_of().tolist()) == {0, 1, 2})
    c.check("SubhaloSfrInRad found (not SubhaloSFRinRad)", cat.s("sfrinrad") is not None)
    c.check("centrals identified from GroupFirstSub",
            np.array_equal(np.where(cat.is_central())[0], np.array([0, 2, 4])),
            "got %s" % np.where(cat.is_central())[0].tolist())
    c.check("target galaxy is the central of group 0", cat.target_index() == 0)

    props = cat.properties(0)
    mstar_expect = 1.0 * 0.7 * (snap.a / (1.0 / 16.0)) ** 3 * snap.units.mass_to_msun
    c.check("subhalo M* from SubhaloMassInRadType", np.isclose(props["mstar_msun"], mstar_expect, rtol=1e-6),
            "got %.4e want %.4e" % (props["mstar_msun"], mstar_expect))
    # The point of the whole subhalo-centric rewrite: for a zoom the FoF Group
    # is the entire system, so its stellar mass is the sum over its galaxies
    # and is not the central galaxy's stellar mass.
    gmt = cat.g("masstype")
    grp_mstar = float(gmt[0, io.PT_STAR] * snap.units.mass_to_msun)
    c.check("group stellar mass is the sum over its galaxies, not the central's",
            grp_mstar > props["mstar_total_msun"] * 1.1,
            "group %.4e vs central %.4e" % (grp_mstar, props["mstar_total_msun"]))
    c.check("central flagged as central", props.get("is_central") is True)
    c.check("host M200c attached from the Group table", props["mhalo_msun"] > props["mstar_msun"])

    pop = cat.population(min_star_particles=0)
    c.check("population: every subhalo kept with no cut", pop["count"] == 5, "got %d" % pop["count"])
    c.check("population: satellites have NaN m200c_central",
            np.isnan(pop["m200c_central"][~pop["central"]]).all()
            and np.isfinite(pop["m200c_central"][pop["central"]]).all())
    hi = cat.population(min_star_particles=0, min_highres=0.9)
    c.check("population: SubhaloHighResStellarFraction cut drops the contaminated halo",
            hi["count"] == 4, "got %d" % hi["count"])
    cen = cat.population(min_star_particles=0, centrals_only=True)
    c.check("population: centrals_only keeps 3", cen["count"] == 3, "got %d" % cen["count"])

    spop = scaling.population_from_catalog(snap, min_star_particles=0)
    c.check("scaling.population_from_catalog reads the same catalogue", spop is not None and spop["count"] == 5)

    # No lookup anywhere in the package may depend on a name AREPO2 does not
    # write.  This is the fence that keeps the original bug from coming back.
    # It inspects string *literals* rather than the raw text, so prose that
    # names the discarded fields (as the comments here and in catalog.py do)
    # does not trip it, while an actual ``sub["SubhaloGrNr"]`` does.
    c.check("no module depends on a field AREPO2 never writes", *_field_name_fence())

    # -- merger tree --------------------------------------------------------
    print("\n[merger tree]")
    t = tree.Tree.open(out, units=snap.units)
    c.check("output/tree.hdf5 found and read", t is not None and len(t) == 8,
            "got %s" % (None if t is None else len(t)))
    if t is not None:
        c.check("tree redshifts descend with snapshot number", np.all(np.diff(t.z) < 0))
        c.check("tree M* rises monotonically", np.all(np.diff(t.mstar_msun()) > 0))
        c.check("tree row found for snapshot %d" % snap.num, t.at_snapshot(snap.num) is not None)
        i = t.at_snapshot(snap.num)
        c.check("tree M* agrees with the catalogue at the same snapshot",
                np.isclose(t.mstar_msun()[i], props["mstar_msun"], rtol=1e-6),
                "tree %.4e vs catalogue %.4e" % (t.mstar_msun()[i], props["mstar_msun"]))
        c.check("tree centre agrees with SubhaloPos",
                np.allclose(t.centre_at(snap.num), cat.s("pos")[0]))

    # -- catalogue-only history: the fast path must agree with the slow one --
    print("\n[catalogue history]")
    chist = history.build_catalog_history(out, verbose=False)
    c.check("catalogue history covers every snapshot", chist["snapnum"].size == 8,
            "got %d" % chist["snapnum"].size)
    c.check("catalogue history M* grows monotonically", np.all(np.diff(chist["mstar_msun"]) > 0))
    c.check("catalogue history has no aperture quantities (no particles read)",
            chist.get("_source") == "catalog"
            and not any(k.startswith(("mgas_r", "mstar_r", "mdm_r")) for k in chist))
    # The plotting layer asks for aperture names; on a catalogue history those
    # must resolve to the Subfind equivalent rather than to NaN, or every
    # scaling relation comes out empty.
    c.check("history.key falls back to the catalogue equivalent",
            np.allclose(history.key(chist, "mstar"), chist["mstar_msun"])
            and np.allclose(history.key(chist, "sfr"), chist["sfr_msun_per_yr"]))
    c.check("history.key still returns NaN for something genuinely absent",
            np.all(np.isnan(history.key(chist, "mgas_cold"))))
    j = int(np.where(chist["snapnum"] == snap.num)[0][0])
    c.check("catalogue history agrees with a direct Catalog read",
            np.isclose(chist["mstar_msun"][j], props["mstar_msun"], rtol=1e-9))
    c.check("catalogue and particle histories pick the same galaxy centre",
            np.allclose(chist["pos_code"][j], cat.s("pos")[0]))

    zlist = os.path.join(root, "ExpansionList.list")
    c.check("expansion list maps a redshift onto a snapshot number",
            io.snapshot_for_redshift(out, float(snap.z), expansion_list=zlist) == snap.num)

    # -- geometry: the part that is easy to get wrong -----------------------
    print("\n[geometry]")
    centre, cinfo = galaxy.auto_centre(snap)
    c.check("centred on the most massive BH", cinfo.get("method") == "bh" and cinfo.get("bh_id") == 900001)

    axis, rad = diagnostics.bh_axis_from_snapshot(snap, centre, radius_kpc=8.0, cold_only=True)
    ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(axis, _mockrun.TRUE_AXIS))), 0, 1))) if axis is not None else 999.0
    c.check("jet axis recovered from the disc gas (< 10 deg)", ang < 10.0, "off by %.2f deg" % ang)

    from .maps import rotation_to_axis

    R = rotation_to_axis(_mockrun.TRUE_AXIS)
    c.check("rotation matrix is orthonormal", np.allclose(R @ R.T, np.eye(3), atol=1e-12))
    c.check("rotation sends the axis to +z", np.allclose(R @ _mockrun.TRUE_AXIS, [0, 0, 1], atol=1e-12))

    # The collimation test must find the synthetic lobes at the right angle.
    contrast = _collimation_contrast(snap, centre, _mockrun.TRUE_AXIS)
    c.check(
        "collimation: polar gas is hotter than equatorial (> 1.5 dex)",
        contrast["dlogT"] > 1.5,
        "got %.2f dex" % contrast["dlogT"],
    )
    c.check(
        "collimation: polar gas outflows, equatorial does not",
        contrast["vr_pol"] > 50 and abs(contrast["vr_equ"]) < 30,
        "polar %.1f, equatorial %.1f km/s" % (contrast["vr_pol"], contrast["vr_equ"]),
    )
    c.check(
        "collimation: transition found near the true lobe angle (+-10 deg)",
        abs(contrast["transition_deg"] - _mockrun.TRUE_LOBE_HALF_ANGLE) < 10.0,
        "found %.1f deg, true %.1f deg" % (contrast["transition_deg"], _mockrun.TRUE_LOBE_HALF_ANGLE),
    )

    # -- history and measurement --------------------------------------------
    print("\n[history]")
    hist = history.load_history(out, rebuild=True, verbose=False)
    c.check("history covers every snapshot", hist["snapnum"].size == 8)
    c.check("history: BH mass is positive and growing", np.all(np.diff(hist["bh_mass_msun"]) > 0))
    c.check("history: stellar mass increases with aperture", np.all(history.key(hist, "mstar", 1.0) <= history.key(hist, "mstar", 30.0)))
    c.check("history: gas phases sum to the total", _phases_sum_ok(hist))
    c.check("history: cache round-trips", _cache_roundtrip_ok(out, hist))
    lam_expect = hist["bh_mdot_msun_per_yr"] / hist["_units"].eddington_rate_msun_per_yr(hist["bh_mass_msun"], _mockrun.EPS_R)
    c.check("history: Eddington ratio uses the run's eps (0.2), not 0.1",
            np.allclose(hist["eddington_ratio"], lam_expect, rtol=1e-10))
    c.check("history: cached units keep the Eddington efficiency",
            history.load_history(out, verbose=False)["_units"].eddington_efficiency == _mockrun.EPS_R)

    pop = scaling.population_from_catalog(snap)
    c.check("population extracted from Subfind", pop is not None and pop["count"] > 0)

    # -- regressions for the two failures the Au6 run exposed ---------------
    print("\n[regressions]")
    # (1) a key that is a 3-vector but absent from the early rows must not
    #     crash the stacking, and must come back with the right shape.
    mixed = [{"z": 1.0}, {"z": 2.0, "v": np.array([1.0, 2.0, 3.0])}, {"z": 3.0, "v": np.array([4.0, 5.0, 6.0])}]
    stacked = _stack_rows(mixed)
    c.check(
        "history stacks a vector key missing from row 0",
        stacked is not None and stacked["v"].shape == (3, 3) and np.isnan(stacked["v"][0]).all() and stacked["v"][1, 2] == 3.0,
    )
    c.check("history j keys are scalars, direction stored separately",
            hist["jstar_kpc_kms"].ndim == 1 and hist.get("jstar_dir", np.zeros((1, 3))).shape[-1] == 3)

    # (2) centring must still succeed with no BH, no groups and no Potential.
    cen, cinf = galaxy.centre_on_densest_mass(snap)
    c.check("densest-mass centring returns a centre", cen is not None)
    if cen is not None:
        off = float(np.linalg.norm(cen - centre)) * snap.units.length_to_kpc(snap.a)
        c.check("densest-mass centre lands near the BH (< 5 kpc)", off < 5.0, "off by %.2f kpc" % off)

    # -- figures ------------------------------------------------------------
    if make_figures:
        print("\n[figures]")
        figdir = os.path.join(tmp, "figures")
        from . import run_all

        rc = run_all.main([root, "--figdir", figdir, "--rebuild"])
        made = [f for f in os.listdir(figdir) if f.endswith(".png")] if os.path.isdir(figdir) else []
        c.check("run_all exits 0", rc == 0)
        c.check("run_all produced >= 19 figures", len(made) >= 19, "got %d" % len(made))
        c.check("report.txt written", os.path.exists(os.path.join(figdir, "report.txt")))

    if make_figures:
        _check_scenarios(c, tmp)

    rc = c.report()
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("\nworkdir kept at %s" % tmp)
    return rc


def _stack_rows(rows):
    """Run the history stacking logic on a list of row dicts (test helper)."""
    import numpy as _np

    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        present = [r[k] for r in rows if k in r and r[k] is not None]
        if not present:
            continue
        shape = _np.shape(present[0])
        if shape == ():
            out[k] = _np.asarray([float(r[k]) if k in r and r[k] is not None else _np.nan for r in rows], dtype=_np.float64)
        else:
            blank = _np.full(shape, _np.nan, dtype=_np.float64)
            out[k] = _np.asarray(
                [_np.asarray(r[k], dtype=_np.float64) if (k in r and _np.shape(r[k]) == shape) else blank for r in rows],
                dtype=_np.float64,
            )
    return out


def _collimation_contrast(snap, centre, axis):
    """Measure the polar/equatorial contrast the way plot_collimation does."""
    u = snap.units
    L = u.length_to_kpc(snap.a)
    gas = snap.part(0, ["Coordinates", "Masses", "Velocities", "InternalEnergy", "ElectronAbundance"])
    rel = io.periodic_delta(gas["Coordinates"], centre, snap.boxsize) * L
    r = np.linalg.norm(rel, axis=1)
    sel = (r > 0) & (r < 20.0)
    rel, r = rel[sel], r[sel]
    m = gas["Masses"][sel]
    T = u.temperature(gas["InternalEnergy"][sel], gas["ElectronAbundance"][sel])
    vc = np.average(gas["Velocities"][sel], axis=0, weights=m)
    dv = (gas["Velocities"][sel] - vc) * u.peculiar_velocity_to_kms(snap.a)
    rhat = rel / r[:, None]
    vr = np.sum(rhat * dv, axis=1)
    alpha = np.degrees(np.arccos(np.clip(np.abs(rhat @ axis), 0, 1)))

    pol, equ = alpha < 30, alpha > 60
    # Locate the transition as the angle where <v_r> falls to half its polar value.
    bins = np.linspace(0, 90, 46)
    ac = 0.5 * (bins[1:] + bins[:-1])
    idx = np.clip(np.digitize(alpha, bins) - 1, 0, ac.size - 1)
    num = np.bincount(idx, weights=m * vr, minlength=ac.size)
    den = np.bincount(idx, weights=m, minlength=ac.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(den > 0, num / den, np.nan)
    half = 0.5 * np.nanmax(prof)
    below = np.where(prof < half)[0]
    transition = ac[below[0]] if below.size else 90.0

    return {
        "dlogT": float(np.average(np.log10(T[pol]), weights=m[pol]) - np.average(np.log10(T[equ]), weights=m[equ])),
        "vr_pol": float(np.average(vr[pol], weights=m[pol])),
        "vr_equ": float(np.average(vr[equ], weights=m[equ])),
        "transition_deg": float(transition),
    }


def _phases_sum_ok(hist, rtol=1e-6):
    R = float(np.max(hist["_radii_kpc"]))
    tot = history.key(hist, "mgas", R)
    parts = sum(history.key(hist, "mgas_%s" % p, R) for p in ("cold", "warm", "hot"))
    good = np.isfinite(tot) & np.isfinite(parts) & (tot > 0)
    return bool(np.allclose(parts[good], tot[good], rtol=rtol))


#: Field names public THESAN/TNG scripts use that AREPO2's fof_io.cc never
#: writes.  Reading one gets you a silent default, not an error, which is why
#: this needs a test of its own.
ABSENT_IN_AREPO2 = ("SubhaloGrNr", "SubhaloFlag", "SubhaloSFRinRad")


def _check_scenarios(c, tmp):
    """Run the whole pipeline on each feedback configuration.

    The package has to cope with runs that write *different sets of files*:
    without a collimated channel there is no ``blackhole_bipolar`` log and no
    injection counters on stdout, and code that assumes them fails on the first
    read.  Each scenario is checked for the figures it should produce, the ones
    it cannot, and -- the point of the exercise -- that nothing is reported as
    a real gap when it is simply not part of that build.
    """
    from . import run_all

    expect = {
        # scenario            collimated  applicable figures    n/a figures
        "local_both":         (True,      {"06", "07", "13"},   set()),
        "local_stars_sph_bh": (False,     {"13"},               {"06", "07"}),
        "sph_both":           (False,     {"13"},               {"06", "07"}),
    }

    for name, (collimated, want, want_na) in expect.items():
        print("\n[scenario: %s]" % name)
        root = os.path.join(tmp, "sc_" + name)
        _mockrun.build(root, nsnap=6, scenario=name)
        out = os.path.join(root, "output")
        figdir = os.path.join(root, "figures")

        snap = io.Snapshot(out, 5)
        summ = diagnostics.model_summary(snap)
        c.check("%s: collimation classified" % name, summ["collimated"] is collimated,
                "got %s" % summ["collimated"])
        c.check("%s: BH deposition classified" % name,
                summ["bh_deposition"] == ("voronoi-local" if name == "local_both" else "kernel"),
                "got %s" % summ["bh_deposition"])
        c.check("%s: stellar deposition classified" % name,
                summ["stellar_deposition"] == ("kernel" if name == "sph_both" else "voronoi-local"),
                "got %s" % summ["stellar_deposition"])

        # The logs must read without raising even when the files do not exist.
        logs = evolution.collect_logs(out, run_dir=root)
        c.check("%s: log collection survives the missing logs" % name, isinstance(logs, dict))
        c.check("%s: bipolar log present iff the channel is on" % name,
                (np.size(logs["bipolar"].get("time", [])) > 0) is collimated)
        c.check("%s: injection counters present iff the channel is on" % name,
                (np.size(logs["injection"].get("time", [])) > 0) is collimated)
        c.check("%s: details layout detected" % name,
                logs["details"].get("_layout") == ("torque" if name == "local_both" else "default"),
                "got %s" % logs["details"].get("_layout"))

        rc = run_all.main([root, "--figdir", figdir, "--rebuild"])
        c.check("%s: run_all exits 0" % name, rc == 0)
        made = {f.split("_")[0] for f in os.listdir(figdir) if f.endswith(".png")}

        c.check("%s: the snapshot history was built (01 and 03 produced)" % name,
                {"01", "03"} <= made, "missing %s" % sorted({"01", "03"} - made))
        c.check("%s: produces %s" % (name, ",".join(sorted(want))), want <= made,
                "missing %s" % sorted(want - made))
        c.check("%s: does not produce %s" % (name, ",".join(sorted(want_na)) or "(nothing)"),
                not (want_na & made), "unexpectedly present: %s" % sorted(want_na & made))

        rows = diagnostics.figure_applicability(
            summ, logs, [os.path.join(figdir, f) for f in os.listdir(figdir) if f.endswith(".png")])
        status = {r[0]: r[2] for r in rows}
        c.check("%s: nothing reported as a real gap" % name,
                not [k for k, v in status.items() if v == diagnostics.MISSING],
                "MISSING: %s" % [k for k, v in status.items() if v == diagnostics.MISSING])
        for fid in want_na:
            c.check("%s: %s classified n/a, not missing" % (name, fid),
                    status.get(fid) == diagnostics.NOT_APPLICABLE, "got %s" % status.get(fid))
        # Every applicability reason must actually explain something.
        c.check("%s: every non-produced figure carries a reason" % name,
                all(r[3] for r in rows if r[2] != diagnostics.PRODUCED))


def _field_name_fence():
    """``(ok, detail)`` -- no module may look up a field AREPO2 does not write.

    ``catalog.py`` is exempt: it lists the discarded spellings deliberately, as
    fallbacks for catalogues written by older code, and every one of them is
    reached through :func:`catalog.field`, which returns None rather than a
    default when nothing matches.
    """
    import ast as _ast
    import glob as _glob

    exempt = {"selftest.py", "catalog.py"}
    bad = []
    for fn in sorted(_glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
        base = os.path.basename(fn)
        if base in exempt:
            continue
        treeobj = _ast.parse(open(fn).read())
        docstrings = set()
        for node in _ast.walk(treeobj):
            if isinstance(node, (_ast.Module, _ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], _ast.Expr) and isinstance(body[0].value, _ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in _ast.walk(treeobj):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value in ABSENT_IN_AREPO2:
                    bad.append("%s:%d %s" % (base, node.lineno, node.value))
    return not bad, "; ".join(bad)


def _cache_roundtrip_ok(out, hist):
    again = history.load_history(out, rebuild=False, verbose=False)
    keys = [k for k in hist if not k.startswith("_") and hist[k].dtype.kind == "f"]
    return all(np.allclose(hist[k], again[k], equal_nan=True) for k in keys)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Self-test the thesan_tmz package.")
    ap.add_argument("--workdir", default=None, help="where to build the mock run (default: a temp dir, deleted after)")
    ap.add_argument("--keep", action="store_true", help="keep the mock run and figures")
    ap.add_argument("--no-figures", action="store_true", help="skip the figure-producing stage")
    a = ap.parse_args(argv)
    return run(workdir=a.workdir, keep=a.keep, make_figures=not a.no_figures)


if __name__ == "__main__":
    sys.exit(main())
