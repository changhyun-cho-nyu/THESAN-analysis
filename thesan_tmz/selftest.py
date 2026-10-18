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
    # The cone-path build: the one configuration that prints the injection
    # counters the stdout parser is tested on.
    _mockrun.build(root, nsnap=8, scenario="cone_bipolar")
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
    # expected AGN energy: constant Mdot = 1 Msun/yr for 1 Gyr, eps_f = 0.05, eps_r = 0.2
    #   -> 0.05 * 0.2 * 1e9 Msun c^2 = 1.7876e61 erg
    un = units.Units(ComovingIntegrationOn=False)
    bt = {"time": np.array([0.0, 1.0 / un.UnitTime_in_Gyr]), "mdot_msun_per_yr": np.array([1.0, 1.0])}
    ex = bh_logs.expected_agn_energy(np.array([1.0]), bt, {"BlackHoleRadiativeEfficiency": 0.2, "BlackHoleFeedbackFactor": 0.05},
                                     {"BH_THERMALFEEDBACK"}, un)
    c.check("expected AGN energy: 1 Msun/yr x 1 Gyr, eps_f eps_r = 0.01 -> 1.788e61 erg",
            ex is not None and abs(ex["hi"][0] / 1.7876e61 - 1) < 1e-3, "got %s" % (ex and ex["hi"]))
    c.check("expected AGN energy: no prediction without the parameters (no 0.1 default)",
            bh_logs.expected_agn_energy(np.array([1.0]), bt, {}, {"BH_THERMALFEEDBACK"}, un) is None)
    c.check("BH_VARIABLE_RADIATIVE_EFFICIENCY -> eps = 1/16, as blackhole_mdot_eddington()",
            units._eddington_efficiency(0.2, {"BH_VARIABLE_RADIATIVE_EFFICIENCY"}) == 1.0 / 16.0
            and units._eddington_efficiency(0.2, set()) == 0.2)
    c.check("Delta_vir(z=0) = 102", abs(u.delta_vir(1.0) - 102.4) < 1.0)
    c.check("Hubble flow: H0 = 67.74 km/s/Mpc -> 0.06774 km/s per kpc at a = 1",
            abs(u.hubble_kms_per_kpc(1.0) - 0.06774) < 1e-6)
    c.check("Hubble flow is zero for a non-comoving run", units.Units(ComovingIntegrationOn=False).hubble_kms_per_kpc(0.5) == 0.0)
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
    c.check("/Config read back", "BH_BIPOLAR_INJECTION_THERMAL" in snap.config)
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

    c.check("no blackhole_bipolar log for a BH_BIPOLAR_INJECTION build", bh_logs.read_bh_bipolar(out)["time"].size == 0)

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
    # -- step 3: the galaxy's BH is ONE black hole, not a sum ---------------
    cen_m, sum_m, mdyn_ok = [], [], True
    for n_, cnt in zip(hist["snapnum"].astype(int), range(hist["snapnum"].size)):
        s_ = io.Snapshot(out, n_)
        b_ = s_.part(5, ["BH_Mass", "ParticleIDs", "Masses", "Coordinates"])
        cen_m.append(b_["BH_Mass"][b_["ParticleIDs"] == 900001][0] * s_.units.mass_to_msun)
        sum_m.append(b_["BH_Mass"].sum() * s_.units.mass_to_msun)
    c.check("history: M_BH is the central BH (900001), not the sum of both",
            np.allclose(hist["bh_mass_msun"], cen_m, rtol=1e-10) and np.all(hist["bh_id"] == 900001))
    c.check("history: the sum is kept separately as bh_mass_sum_msun", np.allclose(hist["bh_mass_sum_msun"], sum_m, rtol=1e-10))
    s_ = io.Snapshot(out, int(hist["snapnum"][-1]))
    cen_, _ = galaxy.auto_centre(s_)
    m_other = galaxy.measure(s_, cen_, bh_id=900002)
    b_ = s_.part(5, ["BH_Mass", "ParticleIDs"])
    c.check("measure(bh_id=...) follows the requested BH",
            m_other["bh_id"] == 900002 and np.isclose(m_other["bh_mass_msun"],
                                                       b_["BH_Mass"][b_["ParticleIDs"] == 900002][0] * s_.units.mass_to_msun))
    m_ = galaxy.measure(s_, cen_)
    bh_ = s_.part(5, ["Coordinates", "Masses"])
    rb_ = io.radius_from(bh_["Coordinates"], cen_, s_.boxsize) * s_.units.length_to_kpc(s_.a)
    ok_ = all(np.isclose(m_["mtot_r%g" % R_] - m_["mgas_r%g" % R_] - m_["mstar_r%g" % R_] - m_["mdm_r%g" % R_],
                         np.sum(bh_["Masses"][rb_ <= R_]) * s_.units.mass_to_msun, rtol=1e-9, atol=1e-6)
              for R_ in (1.0, 2.0, 5.0, 10.0, 30.0))
    c.check("mtot_rR includes only the BHs inside R", ok_)

    # -- feature A: every black hole ----------------------------------------
    from . import bh_population
    bt = bh_population.bh_table(s_)
    b5 = s_.part(5, ["BH_Mass", "ParticleIDs", "BH_Mdot", "BH_MdotEddington", "Coordinates"])
    c.check("bh_table: one row per BH (2)", bt["count"] == 2 and set(bt["id"]) == {900001, 900002})
    o_ = np.argsort(b5["ParticleIDs"])
    c.check("bh_table: masses are BH_Mass in Msun",
            np.allclose(np.sort(bt["mass_msun"]), np.sort(b5["BH_Mass"] * s_.units.mass_to_msun), rtol=1e-10))
    c.check("bh_table: lambda_Edd is the code's own (BH_Mdot / BH_MdotEddington)",
            bt["_eddington_source"] == "BH_MdotEddington"
            and np.allclose(bt["eddington_ratio"][np.argsort(bt["id"])], (b5["BH_Mdot"] / b5["BH_MdotEddington"])[o_]))
    cat_ = catalog.load(s_)
    k1 = int(np.where(bt["id"] == 900001)[0][0])
    sp = np.asarray(cat_.s("pos"))[cat_.target_index()]
    off_ = io.radius_from(b5["Coordinates"][b5["ParticleIDs"] == 900001][0], sp, s_.boxsize) * s_.units.length_to_kpc(s_.a)
    c.check("bh_table: the central BH is hosted by the target galaxy, as a central, at the right offset",
            bt["host_subhalo"][k1] == cat_.target_index() and bt["host_is_central"][k1]
            and np.isclose(bt["offset_kpc"][k1], off_))
    _check_local_mesh(c, s_)
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
        c.check("run_all wrote the local-mesh page (18)", os.path.exists(os.path.join(figdir, "18_bh_local_mesh.html")))
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


def _check_local_mesh(c, snap):
    """Feature B: each BH's cell in its local Voronoi mesh."""
    from . import local_mesh as lm

    print("\n[local mesh]")
    if lm.missing_dependencies():
        c.check("local mesh: scipy and plotly installed", False, "missing %s" % lm.missing_dependencies())
        return
    meshes = lm.all_bh_meshes(snap)
    c.check("local mesh: a cell for each BH (2)", len(meshes) == 2)
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "InternalEnergy"])
    bh = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs"])
    box = snap.boxsize
    for m in meshes:
        tag = "local mesh BH %d" % m["id"]
        f_ = m["faces"]
        x_bh = bh["Coordinates"][bh["ParticleIDs"] == m["id"]][0]
        # the offset is ~1e-12 of the coordinates, so float64 rounding alone leaves ~1e-4 of it
        c.check("%s: injection point = BH + 1e-12 box (0.6, 0.48, 0.64)" % tag,
                np.allclose(io.periodic_delta(m["src"], x_bh, box), 1e-12 * box * np.array([0.6, 0.48, 0.64]),
                            rtol=1e-3, atol=0))
        c.check("%s: faces close to 4 pi" % tag, abs(m["omega_tot"] / (4 * np.pi) - 1) < 1e-6,
                "got %.8f" % (m["omega_tot"] / (4 * np.pi)))
        # Voronoi vertices are equidistant from the source and the face's neighbour, and no
        # generating point is closer (empty sphere) -- checked against ALL the gas, by brute force.
        d_all = io.periodic_delta(gas["Coordinates"], m["src"], box)
        worst_eq, worst_empty = 0.0, 0.0
        for f in f_:
            for v in f["verts"]:
                rv = np.linalg.norm(v)
                worst_eq = max(worst_eq, abs(np.linalg.norm(v - f["offset"]) - rv) / rv)
                worst_empty = max(worst_empty, (rv - np.min(np.linalg.norm(d_all - v, axis=1))) / rv)
        c.check("%s: vertices on the bisector planes (%.1e)" % (tag, worst_eq), worst_eq < 1e-8)
        c.check("%s: no gas point inside any vertex's sphere (%.1e)" % (tag, worst_empty), worst_empty < 1e-8)
        c.check("%s: the nearest gas cell (the host) is a face neighbour" % tag,
                m["host"] in {f["nb_global"] for f in f_})
        c.check("%s: isotropic shares sum to 1, |sum w| = 0" % tag,
                np.isclose(sum(f["wscalar"] for f in f_), 1.0, rtol=1e-12)
                and np.linalg.norm(np.sum([f["w"] for f in f_], axis=0)) < 1e-10)
        ok_c = all(x is not None and np.isclose(x.sum(), 1.0, rtol=1e-12) for x in (m["coll_therm"], m["coll_kin"]))
        c.check("%s: thermal and kinetic collimated shares each sum to 1" % tag, ok_c)
        if m["axis"] is not None and ok_c:
            cosa = np.array([abs(f["w"] @ m["axis"]) / np.linalg.norm(f["w"]) for f in f_])
            iso = np.array([f["wscalar"] for f in f_])
            c.check("%s: kinetic (20 deg) favours the polar faces more than thermal (30 deg) than isotropic" % tag,
                    cosa @ m["coll_kin"] > cosa @ m["coll_therm"] > cosa @ iso)
    m1 = next((m for m in meshes if m["id"] == 900001), None)
    ang = (np.degrees(np.arccos(min(1.0, abs(float(m1["axis"] @ _mockrun.TRUE_AXIS)))))
           if m1 is not None and m1["axis"] is not None else 999.0)
    c.check("local mesh: central BH's axis from the cold disk, within 3 deg of the true disk axis (%.2f)" % ang,
            m1 is not None and m1["cold_disk"] and ang < 3.0)
    c.check("local mesh: profile exponent gives |cos theta|^N = 1/2",
            np.isclose(np.cos(np.radians(20.0)) ** lm.profile_exponent(20.0), 0.5)
            and lm.profile_exponent(0.0) == 0.0 and lm.profile_exponent(91.0) == 0.0)   # as the C++: cos <= 0 or 1
    page = lm.local_mesh_page(snap, os.path.join(os.path.dirname(snap.output_dir), "lm_test.html"), max_bhs=1)
    c.check("local mesh: page written for the top BH only (--mesh-bhs 1)",
            page is not None and open(page).read().count("<h2>BH ") == 1)


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
        # scenario              collimated  BH deposition    stellar deposition  details layout
        "local_both":          (True,       "voronoi-local", "voronoi-local",    "torque"),
        "cone_bipolar":        (True,       "kernel",        "voronoi-local",    "torque"),
        "legacy_bipolar":      (True,       "kernel",        "kernel",           "default"),
        "local_bh_only":       (False,      "voronoi-local", "kernel",           "default"),
        "local_stars_sph_bh":  (False,      "kernel",        "voronoi-local",    "default"),
        "sph_both":            (False,      "kernel",        "kernel",           "default"),
    }

    for name, (collimated, bh_dep, st_dep, layout) in expect.items():
        print("\n[scenario: %s]" % name)
        root = os.path.join(tmp, "sc_" + name)
        _mockrun.build(root, nsnap=6, scenario=name)
        out = os.path.join(root, "output")
        figdir = os.path.join(root, "figures")
        # What the C++ writes for this Config -- the reference the package is tested against.
        truth = _mockrun.logs_written(_mockrun.SCENARIOS[name]["config"])
        want = {"13", "16", "17"} | ({"06"} if truth["counters"] else set()) | ({"07"} if truth["bipolar_log"] else set())
        want_na = {"06", "07"} - want

        snap = io.Snapshot(out, 5)
        summ = diagnostics.model_summary(snap)
        c.check("%s: collimation classified" % name, summ["collimated"] is collimated,
                "got %s" % summ["collimated"])
        c.check("%s: BH deposition classified" % name, summ["bh_deposition"] == bh_dep,
                "got %s" % summ["bh_deposition"])
        c.check("%s: stellar deposition classified" % name, summ["stellar_deposition"] == st_dep,
                "got %s" % summ["stellar_deposition"])

        # The logs must read without raising even when the files do not exist.
        logs = evolution.collect_logs(out, run_dir=root)
        c.check("%s: log collection survives the missing logs" % name, isinstance(logs, dict))
        c.check("%s: blackhole_bipolar log read iff the build writes it" % name,
                (np.size(logs["bipolar"].get("time", [])) > 0) is truth["bipolar_log"])
        c.check("%s: injection counters read iff the build prints them" % name,
                (np.size(logs["injection"].get("time", [])) > 0) is truth["counters"])
        if truth["bipolar_log"]:
            c.check("%s: bipolar jhat is a unit vector" % name,
                    np.allclose(np.linalg.norm(logs["bipolar"]["jhat"], axis=1), 1.0, atol=1e-9))
        c.check("%s: details layout detected" % name, logs["details"].get("_layout") == layout,
                "got %s" % logs["details"].get("_layout"))

        rc = run_all.main([root, "--figdir", figdir, "--rebuild"])
        c.check("%s: run_all exits 0" % name, rc == 0)

        # -- the AGN energy book (step 5) ------------------------------------
        cfg_ = set(_mockrun.SCENARIOS[name]["config"])
        want_src = ("ledger" if truth["ledger"] else "counters" if truth["counters"]
                    else "snapshots" if "BH_THERMALFEEDBACK" in cfg_ else None)
        if truth["ledger"]:
            c.check("%s: ledger lines parsed (300)" % name, logs["ledger"]["time"].size == 300,
                    "got %d" % logs["ledger"]["time"].size)
        hist_ = history.load_history(out, verbose=False)
        src_ = bh_logs.agn_energy_sources(hist_, logs, snap.units)
        c.check("%s: AGN energy taken from the %s" % (name, want_src), src_["source"] == want_src,
                "got %s" % src_["source"])
        if want_src in ("ledger", "snapshots"):
            rel_ = src_["released"]
            ex_ = bh_logs.expected_agn_energy(rel_[0], logs["blackholes"], summ["params"], summ["config"], snap.units)
            ok_t = rel_[1] > 0
            if np.allclose(ex_["lo"], ex_["hi"]):
                # blackholes.txt is printed with %14e (7 digits in time and Mdot), so the
                # prediction integrated from it is good to ~1e-4, in a real run as here.
                r_ = rel_[1][ok_t] / ex_["hi"][ok_t]
                c.check("%s: released = expected energy (max |ratio-1| %.2g)" % (name, np.max(np.abs(r_ - 1))),
                        np.allclose(r_, 1.0, rtol=1e-3))
            else:
                c.check("%s: released energy inside the mode-switch band" % name,
                        np.all(rel_[1][ok_t] >= ex_["lo"][ok_t] * (1 - 1e-3))
                        and np.all(rel_[1][ok_t] <= ex_["hi"][ok_t] * (1 + 1e-3)))
        if want_src == "ledger":
            # the mock applies 0.999 of the thermal and 0.98 of the kinetic energy it was asked for
            L_ = logs["ledger"]
            dep_expect = np.cumsum(0.999 * (L_["drained_iso"] + L_["drained_coll"]) + 0.98 * L_["drained_kin"])
            dep_ = src_["deposited"]
            ok_d = dep_expect > 0
            c.check("%s: deposited energy = what the ledger says reached the gas" % name,
                    np.allclose(dep_[1][ok_d] / snap.units.energy_to_erg, dep_expect[ok_d], rtol=1e-5)
                    and np.all(dep_[1] <= rel_[1] * (1 + 1e-12)))
        made = {f.split("_")[0] for f in os.listdir(figdir) if f.endswith(".png")}

        c.check("%s: the snapshot history was built (01 and 03 produced)" % name,
                {"01", "03"} <= made, "missing %s" % sorted({"01", "03"} - made))
        c.check("%s: produces %s" % (name, ",".join(sorted(want))), want <= made,
                "missing %s" % sorted(want - made))
        c.check("%s: does not produce %s" % (name, ",".join(sorted(want_na)) or "(nothing)"),
                not (want_na & made), "unexpectedly present: %s" % sorted(want_na & made))

        c.check("%s: 18_bh_local_mesh.html written" % name, os.path.exists(os.path.join(figdir, "18_bh_local_mesh.html")))
        rows = diagnostics.figure_applicability(
            summ, logs, [os.path.join(figdir, f) for f in os.listdir(figdir) if f.endswith((".png", ".html"))])
        status = {r[0]: r[2] for r in rows}
        c.check("%s: nothing reported as a real gap" % name,
                not [k for k, v in status.items() if v == diagnostics.MISSING],
                "MISSING: %s" % [k for k, v in status.items() if v == diagnostics.MISSING])
        for fid in want_na:
            c.check("%s: %s classified n/a, not missing" % (name, fid),
                    status.get(fid) == diagnostics.NOT_APPLICABLE, "got %s" % status.get(fid))
        if truth["axis_only"]:
            reason = {r[0]: r[3] for r in rows}.get("06", "")
            c.check("%s: 06 n/a explains the LOCAL_INJECTION_BLACK_HOLES design" % name,
                    "LOCAL_INJECTION_BLACK_HOLES" in reason and "LEDGER" in reason, "got %r" % reason)
        note18 = {r[0]: r[3] for r in rows}.get("18", "")
        c.check("%s: 18 says when the shares are hypothetical (kernel BH deposit)" % name,
                ("LOCAL_INJECTION_BLACK_HOLES is off" in note18) is ("LOCAL_INJECTION_BLACK_HOLES" not in cfg_),
                "got %r" % note18)
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
