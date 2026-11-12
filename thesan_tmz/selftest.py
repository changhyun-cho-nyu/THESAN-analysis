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
from .plotstyle import apply_style, plt


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
    _check_slr_mesh(c)
    _check_lowres(c, s_, tmp)
    _check_eddington_cap(c, s_.units)
    _check_maps(c, s_)
    _check_uv(c, s_)
    _check_kinematics(c, s_)
    _check_map_snap(c, root, tmp)
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
    # a BH at the edge of a dense region: the cell must equal the one built from all points
    rng = np.random.default_rng(3)
    wrong = 0
    for _ in range(10):
        pts = np.vstack([rng.normal(0, 1, (2000, 3)) - [3, 0, 0], rng.uniform(-60, 60, (300, 3))])
        d_ = pts - np.array([rng.uniform(0, 6), 0, 0])
        r_ = np.linalg.norm(d_, axis=1)
        sc = float(np.sort(r_)[15])
        fa, _ = lm.complete_cell(d_, r_, sc)
        tr = lm.voronoi_cell(d_, sc)
        rv = lambda F: max(float(np.max(np.linalg.norm(f["verts"], axis=1))) for f in F)
        wrong += fa is None or len(fa) != len(tr) or abs(rv(fa) - rv(tr)) > 1e-6 * rv(tr)
    c.check("local mesh: complete cell at the edge of a dense region (10 random BHs)", wrong == 0,
            "%d of 10 differ from the cell built from all points" % wrong)
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


def _check_lowres(c, snap, tmp):
    """Low-res BHs (zoom-edge seeding artefacts) are found and hidden."""
    import shutil

    import h5py

    from . import bh_logs, bh_population, galaxy, lowres
    from . import local_mesh as lm

    print("\n[low-res BHs]")
    # 1. the classifier against brute force, with a BH whose kernel wraps the box edge
    rng = np.random.default_rng(3)
    box = 10.0
    pos = rng.uniform(0, box, (6000, 3))
    mass = rng.uniform(0.5, 1.5, 6000)
    lowzone = io.radius_from(pos, np.array([7.0, 7.0, 7.0]), box) < 1.5
    hr = np.where(lowzone, 0.1, 0.9) * mass
    bhp = np.array([[0.05, 5.0, 5.0], [7.0, 7.0, 7.0], [3.0, 3.0, 3.0]])
    hs = np.array([0.8, 1.0, 0.0])                       # the third has no kernel: nearest cell
    chunks = [(pos[:2500], mass[:2500], hr[:2500]), (pos[2500:], mass[2500:], hr[2500:])]   # two "files"
    f, n = lowres.highres_fraction(bhp, hs, chunks, box)
    exp = []
    for i in range(2):
        r = io.radius_from(pos, bhp[i], box)
        exp.append(hr[r <= hs[i]].sum() / mass[r <= hs[i]].sum())
    j = int(np.argmin(io.radius_from(pos, bhp[2], box)))
    c.check("lowres: f_hr within BH_Hsml = brute force, across files and the periodic edge",
            np.allclose(f[:2], exp, rtol=1e-12) and n[0] == np.sum(io.radius_from(pos, bhp[0], box) <= hs[0]),
            "got %s, expected %s" % (f[:2], exp))
    c.check("lowres: no cell inside BH_Hsml -> the nearest cell decides", n[2] == 0 and np.isclose(f[2], hr[j] / mass[j]))
    c.check("lowres: the BH in the low-res zone is low-res, the others are not",
            list(f < lowres.HIGHRESMASSFAC) == [False, True, False])

    # 2. the mock has no HighResGasMass: nothing can be tested, nothing is hidden
    lr = lowres.classify(snap)
    c.check("lowres: no HighResGasMass -> untestable, all BHs kept",
            not lr["testable"] and not lr["lowres"].any() and bh_population.bh_table(snap)["count"] == 2)

    # 3. the same snapshot with HighResGasMass added: low-res gas around both BHs, then high-res
    out2 = os.path.join(tmp, "lowres_copy", "output")
    if os.path.isdir(os.path.dirname(out2)):
        shutil.rmtree(os.path.dirname(out2))
    os.makedirs(out2)
    for nm in ("snapdir_%03d" % snap.num, "groups_%03d" % snap.num):
        if os.path.isdir(os.path.join(snap.output_dir, nm)):
            shutil.copytree(os.path.join(snap.output_dir, nm), os.path.join(out2, nm))

    def with_hr(frac):
        for fn in io.snapshot_files(out2, snap.num):
            with h5py.File(fn, "r+") as fh:
                g = fh["PartType0"]
                if "HighResGasMass" in g:
                    del g["HighResGasMass"]
                g.create_dataset("HighResGasMass", data=frac * g["Masses"][:])
        return io.Snapshot(out2, snap.num)

    s_low = with_hr(0.2)
    t = bh_population.bh_table(s_low)
    t_all = bh_population.bh_table(s_low, hide_lowres=False)
    c.check("lowres: all gas low-res -> both BHs hidden from bh_table, listed in _hidden_ids",
            t["count"] == 0 and sorted(t["_hidden_ids"]) == [900001, 900002])
    c.check("lowres: hide_lowres=False keeps them, with highres_frac = 0.2",
            t_all["count"] == 2 and np.allclose(t_all["highres_frac"], 0.2))
    if not lm.missing_dependencies():
        c.check("lowres: the 3D page leaves them out", lm.all_bh_meshes(s_low) == [])
    cen, info = galaxy.centre_on_bh(s_low, in_target_galaxy=False)
    c.check("lowres: the centre fallback refuses a low-res BH", cen is None and "low-res" in info.get("reason", ""))
    cen, info = galaxy.centre_on_bh(s_low)
    c.check("lowres: the target galaxy's own BH is still the centre", cen is not None and info["bh_id"] == 900001)
    s_high = with_hr(1.0)
    c.check("lowres: all gas high-res -> nothing hidden",
            bh_population.bh_table(s_high)["count"] == 2 and not lowres.lowres_ids(s_high))
    det = {"id": np.array([1, 2, 1, 3]), "time": np.arange(4.0)}
    mer = {"id_survivor": np.array([1, 3]), "id_swallowed": np.array([2, 4]), "time": np.array([0.5, 0.7])}
    c.check("lowres: drop_bhs removes the rows of hidden BHs (details and both merger IDs)",
            list(bh_logs.drop_bhs(det, {1})["id"]) == [2, 3]
            and list(bh_logs.drop_bhs(mer, {4}, ("id_survivor", "id_swallowed"))["time"]) == [0.5])
    shutil.rmtree(os.path.dirname(out2), ignore_errors=True)


def _check_eddington_cap(c, units):
    """Fig 08 / mode-switch check cap the logged rate only when the build caps."""
    from . import diagnostics as dg

    print("\n[Eddington cap in the details figures]")
    mbh = np.array([1e6, 1e6, 1e6])
    edd = units.eddington_rate_msun_per_yr(mbh)
    raw = edd * np.array([0.1, 3.0, 300.0])
    capped, on, _ = dg._enforced_mdot(raw, mbh, units, {"BlackHoleEddingtonFactor": 1.0}, {"config": {"BLACK_HOLES": 1}})
    c.check("Eddington cap: default build clips at BlackHoleEddingtonFactor x Edd",
            on and np.allclose(capped / edd, [0.1, 1.0, 1.0]))
    capped2, _, _ = dg._enforced_mdot(raw, mbh, units, {"BlackHoleEddingtonFactor": 2.0}, {"config": {}})
    c.check("Eddington cap: factor 2 clips at 2 x Edd", np.allclose(capped2 / edd, [0.1, 2.0, 2.0]))
    free, on3, note = dg._enforced_mdot(raw, mbh, units, {"BlackHoleEddingtonFactor": 1.0},
                                        {"config": {"BH_NO_EDDINGTON_LIMIT": 1}})
    c.check("Eddington cap: BH_NO_EDDINGTON_LIMIT keeps the raw rate (super-Eddington visible) and says why",
            not on3 and np.array_equal(free, raw) and "BH_NO_EDDINGTON_LIMIT" in note)
    det = {"time": np.array([0.10, 0.11, 0.12]), "id": np.array([1, 1, 1]), "bh_mass": mbh / units.mass_to_msun,
           "mdot": raw / units.mdot_to_msun_per_yr}
    prm = {"BlackHoleEddingtonFactor": 1.0, "BHBipolarInjEddThreshold": 2.0}
    r_cap = dg.check_mode_switch(det, None, units, prm, summary={"config": {}})
    r_free = dg.check_mode_switch(det, None, units, prm, summary={"config": {"BH_NO_EDDINGTON_LIMIT": 1}})
    c.check("mode-switch check: lambda >= 2 never happens with the cap, does without it",
            "predicted thermal fraction    : 0.000" in r_cap and "predicted thermal fraction    : 0.000" not in r_free
            and "no Eddington cap" in r_free)


def _check_uv(c, snap):
    """Per-star intrinsic 1500 A luminosity against the mock's analytic spectra."""
    import h5py

    from . import uv

    print("\n[uv]")
    st = snap.part(io.PT_STAR, uv.stars_with_uv_fields())
    lam, note = uv.star_l_nu(snap, st)
    c.check("uv: spectrum tables found from the run's /Parameters", lam is not None, note)
    if lam is None:
        return
    # expected, from the formula, with the mock's L_nu = amp E:
    #   point (default): L_nu at E_1500 = hc / 1500 A, i.e. amp E_1500
    #   arepo: integral of L_nu dnu over 1490-1510 A = amp (eV/h) (E2^2 - E1^2)/2, over 20 A -> L_lambda,
    #          then L_nu = L_lambda lambda^2 / c
    hc_ev_a = 6.62607015e-27 * 2.99792458e10 / 1.602176634e-12 * 1e8
    e1, e2, e15 = hc_ev_a / 1510.0, hc_ev_a / 1490.0, hc_ev_a / 1500.0
    to_lnu = 1e8 * 1.5e-5 ** 2 / 2.99792458e10
    per_amp_point = e15
    per_amp_arepo = 1.602176634e-12 / 6.62607015e-27 * (e2 ** 2 - e1 ** 2) / 2.0 / 20.0 * to_lnu
    u = snap.units
    age = snap.age_gyr - u.age_gyr(np.asarray(st["GFM_StellarFormationTime"], dtype=np.float64))
    p3 = np.asarray(st["IsPOPIII"], dtype=bool)
    amp = np.where(p3, _mockrun.sed_lnu(age, pop3=True, e_ev=np.ones(1))[:, 0],
                   _mockrun.sed_lnu(age, np.asarray(st["GFM_Metallicity"]), e_ev=np.ones(1))[:, 0])
    m_init = np.asarray(st["GFM_InitialMass"], dtype=np.float64) * u.mass_to_msun
    want = m_init * amp * per_amp_point
    c.check("uv (point, default): L_nu(1500) per star = initial mass x table L_nu at exactly 1500 A, Pop II and III",
            p3.any() and (~p3).any() and np.allclose(lam, want, rtol=1e-9),
            "max rel. diff %.2e, %d Pop III" % (np.max(np.abs(lam / want - 1)), int(p3.sum())))
    lam_a, _ = uv.star_l_nu(snap, st, recipe="arepo")
    c.check("uv (arepo): L_nu = (1490-1510 A average of L_lambda) lambda^2/c, x initial mass, Pop II and III",
            lam_a is not None and np.allclose(lam_a, m_init * amp * per_amp_arepo, rtol=1e-9))
    fake = {k: np.asarray(v)[:3].copy() for k, v in st.items() if np.ndim(v) > 0}
    fake["GFM_StellarFormationTime"][:] = [-0.5, 0.0, snap.a]
    l3, _ = uv.star_l_nu(snap, fake)
    tab_young = _mockrun.sed_lnu(_mockrun.SED_AGES_GYR[0], fake["GFM_Metallicity"][2:3], e_ev=np.ones(1))[0, 0]
    c.check("uv: wind particles emit nothing; a star born now gets the youngest table entry (clamped, as AREPO)",
            l3[0] == 0 and l3[1] == 0 and np.isclose(l3[2], fake["GFM_InitialMass"][2] * u.mass_to_msun * tab_young
                                                      * per_amp_point, rtol=1e-9))
    c.check("uv: L_nu = L_lambda lambda^2/c at 1500 A, and M_AB = 51.60 - 2.5 log L_nu (1e28 erg/s/Hz -> -18.40)",
            np.isclose(uv.l_nu(1.0), 1e8 * 1.5e-5 ** 2 / 2.99792458e10) and abs(float(uv.muv(1e28)) + 18.40) < 0.01,
            "M_UV %.3f" % float(uv.muv(1e28)))
    from . import galaxy, maps

    cen, _ = galaxy.auto_centre(snap)
    rows, notes = maps.galaxy_view_rows(snap, cen, 1.0)
    rows_y, _ = maps.galaxy_view_rows(snap, cen, 1.0, young_myr=1e5)
    names = [r[0] for r in rows]
    young = [r for r in rows_y if r[0].startswith("stars <")]
    allst = rows_y[0]
    c.check("fig 20 rows: all stars, UV, gas; the mock's few young stars are left out with a note",
            names[0] == "all stars" and any("UV" in n for n in names) and names[-1] == "gas"
            and not any(n.startswith("stars <") for n in names) and any("young row left out" in n for n in notes),
            "%s | %s" % (names, notes))
    c.check("fig 20 young row: with every star 'young' it holds all of them, with its own k-NN h (= the all-star h)",
            len(young) == 1 and len(young[0][2]) == len(allst[2]) and np.allclose(young[0][3], allst[3]))
    uvrow = [r for r in rows if "UV" in r[0]][0]
    c.check("fig 20 UV row weights = L_nu of the same stars (no winds: mock has none)",
            len(uvrow[2]) == len(rows[0][2]) and np.all(uvrow[2] > 0))
    bars = [maps.scale_bar_length(w) for w in (2.12, 600.0, 40.0, 0.3)]
    c.check("fig 20 scale bar: a round 1/2/5 x 10^n length, at most a quarter of the frame",
            bars == [0.5, 100.0, 10.0, 0.05], str(bars))
    # fig 21: UV opacity and colour rise linearly in log Sigma_UV over the shown decades
    test_img = np.array([[0.0, 1e2, 1e4, 1e6, 1e7]])
    rgba, lv = maps.uv_overlay_rgba(test_img, 6.0, 4.0, plt.get_cmap("inferno"))
    c.check("fig 21 UV overlay: transparent at and below 4 dex under the top, half-opaque 2 dex under it, "
            "opaque at and above the top; empty pixels transparent",
            np.allclose(rgba[0, :, 3], [0.0, 0.0, 0.5, 1.0, 1.0]), str(rgba[0, :, 3]))
    with tempfile.TemporaryDirectory() as td:
        out21 = maps.plot_gas_uv(snap, cen, os.path.join(td, "21_gas_uv.png"), title="selftest")
        c.check("fig 21 is drawn when the run has spectrum tables", out21 is not None and os.path.isfile(out21))
        saved = dict(snap.parameters)
        try:
            # Pop II table missing: skipped, Pop II stars dark, Pop III stars still lit
            snap.parameters["SpectrumTablePath"] = "/nonexistent/table.hdf5"
            l_p2gone, why_p2 = uv.star_l_nu(snap, st)
            # both missing: no UV, no fig 21
            snap.parameters["PopIIISpectrumTablePath"] = "/nonexistent/pop3.hdf5"
            l0, why = uv.star_l_nu(snap, st)
            none21 = maps.plot_gas_uv(snap, cen, os.path.join(td, "21_none.png"))
            # the run names no table: nothing without --spectra-dir, the two default files with it
            del snap.parameters["SpectrumTablePath"], snap.parameters["PopIIISpectrumTablePath"]
            l_noparam, why_noparam = uv.star_l_nu(snap, st)
            for src, name in zip((saved["SpectrumTablePath"], saved["PopIIISpectrumTablePath"]), uv.DEFAULT_TABLES):
                shutil.copy(src, os.path.join(td, name))
            l_def, why_def = uv.star_l_nu(snap, st, spectra_dir=td)
            os.remove(os.path.join(td, uv.DEFAULT_TABLES[1]))
            l_def2, why_def2 = uv.star_l_nu(snap, st, spectra_dir=td)
        finally:
            snap.parameters.clear()
            snap.parameters.update(saved)
    c.check("uv: a missing Pop II table is skipped: Pop II stars get no UV, Pop III stars keep theirs, and it says so",
            l_p2gone is not None and np.all(l_p2gone[~p3] == 0) and np.allclose(l_p2gone[p3], want[p3], rtol=1e-9)
            and "Pop II table not found" in why_p2, why_p2)
    c.check("uv: both tables missing gives no UV and says why", l0 is None and why.count("not found") == 2, why)
    c.check("uv: a run naming no table has no UV without --spectra-dir", l_noparam is None and "not given" in why_noparam,
            why_noparam)
    c.check("uv: a run naming no table uses the two default files in --spectra-dir (same answer as the run's own tables)",
            l_def is not None and np.allclose(l_def, want, rtol=1e-9) and why_def.count("default") == 2, why_def)
    c.check("uv: a default file missing from --spectra-dir is skipped (Pop III dark, Pop II kept)",
            l_def2 is not None and np.all(l_def2[p3] == 0) and np.allclose(l_def2[~p3], want[~p3], rtol=1e-9), why_def2)
    c.check("fig 21 is not drawn without spectrum tables", none21 is None)
    rows21 = diagnostics.figure_applicability({}, {}, [], no_uv=why)
    st21 = [r for r in rows21 if r[0] == "21"]
    c.check("fig 21 is 'not applicable' (not 'missing') without spectrum tables, with the reason",
            len(st21) == 1 and st21[0][2] == "n/a" and "table not found" in st21[0][3], str(st21))


def _check_map_snap(c, root, tmp):
    """run_all --map-snap: the maps and the kinematics use the chosen snapshot; a missing one skips them."""
    import contextlib
    import io as _io

    from . import kinematics as kin, maps, run_all

    print("\n[--map-snap]")
    seen = {}
    orig = (maps.plot_galaxy_view, maps.plot_phase_diagram, kin.analyse)

    def rec(name, fn):
        def wrapped(snap, *a, **kw):
            seen[name] = snap.num
            return fn(snap, *a, **kw)
        return wrapped

    maps.plot_galaxy_view = rec("galaxy_view", orig[0])
    maps.plot_phase_diagram = rec("phase_diagram", orig[1])
    kin.analyse = rec("kinematics", orig[2])
    buf = _io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_all.main([root, "--figdir", os.path.join(tmp, "fig_mapsnap"), "--map-snap", "3", "--mesh-bhs", "0"])
            seen_good = dict(seen)
            seen.clear()
            run_all.main([root, "--figdir", os.path.join(tmp, "fig_mapsnap_bad"), "--map-snap", "999", "--mesh-bhs", "0"])
    finally:
        maps.plot_galaxy_view, maps.plot_phase_diagram, kin.analyse = orig
    log = buf.getvalue()
    c.check("--map-snap 3: figs 10, 20 and the kinematics are made from snapshot 3, not the last",
            seen_good == {"galaxy_view": 3, "phase_diagram": 3, "kinematics": 3}, str(seen_good))
    c.check("--map-snap 999 (no such snapshot): maps skipped with a message, listed as skipped, not missing",
            not seen and "no such snapshot" in log and "22 kinematics          skipped" in log
            and "missing" in log and " 0 missing" in log, str(seen))


def _check_kinematics(c, snap):
    """Fig 22: spin angles and v/sigma on a synthetic rotating disc; bound-star slice and its order check."""
    from . import galaxy, kinematics as kin

    print("\n[kinematics]")
    rng = np.random.default_rng(11)

    def disc(n, axis, vrot, sig):
        e3 = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        e1 = np.cross([0.0, 0.0, 1.0] if abs(e3[2]) < 0.9 else [1.0, 0.0, 0.0], e3)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        R = rng.uniform(0.05, 2.0, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        pos = np.outer(R * np.cos(ph), e1) + np.outer(R * np.sin(ph), e2) + np.outer(rng.normal(0, 0.05, n), e3)
        vel = vrot * (np.outer(-np.sin(ph), e1) + np.outer(np.cos(ph), e2)) + rng.normal(0, sig, (n, 3))
        return pos, vel

    ax_s = np.array([0.2, -0.3, 0.93])
    tilt = np.radians(40.0)
    ax_g = np.array([np.sin(tilt), 0.0, np.cos(tilt)])                 # 40 deg from z
    rot = np.cross([0.0, 0.0, 1.0], ax_s / np.linalg.norm(ax_s))
    th = np.arccos(ax_s[2] / np.linalg.norm(ax_s))
    k = rot / np.linalg.norm(rot)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rm = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K           # takes z to ax_s
    ax_g = Rm @ ax_g                                                    # gas axis: 40 deg from the stars'
    ps, vs = disc(20000, ax_s, 150.0, 50.0)
    pg, vg = disc(20000, ax_g, 120.0, 0.0)
    bulk = np.array([300.0, -100.0, 50.0])
    stars = dict(pos=ps, vel=vs + bulk, mass=np.ones(len(ps)), age_myr=np.where(np.arange(len(ps)) % 2, 5.0, 500.0))
    gas = dict(pos=pg, vel=vg + bulk, mass=np.ones(len(pg)), sfr=np.ones(len(pg)))
    tab = kin.compute(stars, gas, rhalf_kpc=1.0, radii_rhalf=(0.5, 1.0, 3.0))
    big = tab["spheres"][-1]
    c.check("kinematics: angle(L_star, L_gas) = the 40 deg tilt; star spin axis found; bulk velocity removed",
            abs(big["ang_star_gas"] - 40.0) < 1.0 and kin.angle_deg(tab["axis"], ax_s) < 1.0
            and np.allclose(tab["vframe"], bulk, atol=3.0),
            "%.2f deg, axis %.2f deg off, vframe %s" % (big["ang_star_gas"], kin.angle_deg(tab["axis"], ax_s),
                                                         np.round(tab["vframe"], 1)))
    c.check("kinematics: v/sigma of a disc with v_rot 150, sigma 50 km/s: v ~ 150, sigma_z ~ sigma_1D ~ 50 (2 %)",
            abs(big["all_v"] - 150) < 3 and abs(big["all_sigz"] - 50) < 1.0 and abs(big["all_sig1d"] - 50) < 1.0
            and abs(big["young_v"] - 150) < 4 and big["young_n"] == big["all_n"] // 2,
            "v %.1f sig_z %.1f sig_1D %.1f" % (big["all_v"], big["all_sigz"], big["all_sig1d"]))
    c.check("kinematics: shells cover the spheres (star counts add up)",
            sum(x["all_n"] for x in tab["shells"]) == big["all_n"])

    # bound-star slice: a catalogue whose offset/length cover the mock's stars and whose r_1/2 is right
    # is accepted; a wrong r_1/2 (snapshot not group-ordered) is refused
    st = snap.part(io.PT_STAR, ["Coordinates", "Masses"])
    nst = int(st["count"])
    L = snap.units.length_to_kpc(snap.a)
    m = np.asarray(st["Masses"], dtype=float)
    cm = np.average(np.asarray(st["Coordinates"], dtype=float), axis=0, weights=m)
    r = np.linalg.norm(io.periodic_delta(st["Coordinates"], cm, snap.boxsize), axis=1)
    o = np.argsort(r)
    rh = r[o][np.searchsorted(np.cumsum(m[o]), 0.5 * m.sum())]

    class FakeCat:
        def __init__(self, rh_code):
            lt = np.zeros((1, 6), dtype=np.int64)
            lt[0, 4] = nst
            hr = np.zeros((1, 6))
            hr[0, 4] = rh_code
            self.d = {"offsettype": np.zeros((1, 6), dtype=np.int64), "lentype": lt, "halfmassradtype": hr,
                      "pos": cm[None, :]}

        def s(self, k, default=None):
            return self.d.get(k, default)

        def target_index(self):
            return 0

    good, why_g = kin.bound_stars(snap, FakeCat(rh), 0)
    bad, why_b = kin.bound_stars(snap, FakeCat(3.0 * rh), 0)
    c.check("kinematics: bound-star slice accepted when half its mass lies within the catalogue r_1/2; "
            "refused (not group-ordered) when not",
            good is not None and np.isclose(good["rhalf_kpc"], rh * L) and bad is None and "group-ordered" in why_b,
            "%s | %s" % (why_g, why_b))
    with tempfile.TemporaryDirectory() as td:
        out = kin.plot(tab, os.path.join(td, "22_kinematics.png"), title="selftest", z=4.0)
        c.check("kinematics: fig 22 is drawn", out is not None and os.path.isfile(out))
    rows = diagnostics.figure_applicability({}, {}, [], no_kinematics="catalogue has no SubhaloOffsetType")
    r22 = [x for x in rows if x[0] == "22"]
    c.check("fig 22 is 'not applicable' with the reason when there is no bound-star slice",
            len(r22) == 1 and r22[0][2] == "n/a" and "SubhaloOffsetType" in r22[0][3], str(r22))
    lines = kin.report_lines(tab, "selftest")
    c.check("kinematics: report block lists every sphere and shell", lines.count("shell ") == 3 and lines.count("r < ") == 3)


def _check_maps(c, snap):
    """Adaptive projection, galaxy spin axis, softening."""
    from . import galaxy, maps

    print("\n[maps]")
    if not maps._HAVE_SCIPY:
        c.check("maps: scipy installed (adaptive projection)", False)
        return
    rng = np.random.default_rng(5)
    pos = rng.uniform(-5, 5, (3000, 3))
    w = rng.uniform(0.5, 1.5, 3000)
    sig = rng.uniform(0.01, 0.8, 3000)
    img, ext = maps.project_adaptive(pos, w, sig, 20.0, npix=200)
    px = 40.0 / 200
    c.check("maps: adaptive projection conserves mass (elements well inside the frame)",
            np.isclose(img.sum() * px ** 2, w.sum(), rtol=1e-6), "%g vs %g" % (img.sum() * px ** 2, w.sum()))
    # one element of h = 3 kpc at a pixel centre: nothing beyond h (compact support), and its projected
    # second moment <R^2> = 2/3 <r^2> of the 3D cubic spline, <r^2> = 0.225 h^2 (integrated here, independently)
    q = np.linspace(0.0, 1.0, 100001)
    W = np.where(q < 0.5, 1 - 6 * q ** 2 + 6 * q ** 3, 2 * (1 - q) ** 3)
    r2_3d = np.trapezoid(q ** 4 * W, q) / np.trapezoid(q ** 2 * W, q)
    one, _ = maps.project_adaptive(np.array([[0.5 * px, 0.5 * px, 0.0]]), [1.0], [3.0], 20.0, npix=200)
    yy, xx = np.mgrid[0:200, 0:200]
    rr = np.hypot(xx - 100, yy - 100) * px
    R2 = np.sum(one * rr ** 2) / np.sum(one)
    c.check("maps: cubic spline: nothing beyond h, projected <R^2> = 2/3 <r^2> of the 3D kernel",
            one[rr > 3.0 + px].sum() < 1e-10 * one.sum() and np.isclose(R2, 2.0 / 3.0 * r2_3d * 9.0 + px ** 2 / 6.0, rtol=0.01),
            "<R^2> %.4f vs %.4f kpc^2 (+ pixel px^2/6)" % (R2, 2.0 / 3.0 * r2_3d * 9.0 + px ** 2 / 6.0))
    vals, _ = maps.project_adaptive(pos, w, sig, 20.0, npix=200, values=np.full(3000, 7.0))
    c.check("maps: a constant field stays constant in the weighted mean", np.allclose(vals[np.isfinite(vals)], 7.0))
    cen, _ = galaxy.auto_centre(snap)
    ax_, info = maps.galaxy_axis(snap, cen)
    ang = np.degrees(np.arccos(min(1.0, abs(float(ax_ @ _mockrun.TRUE_AXIS))))) if ax_ is not None else 180.0
    c.check("maps: galaxy spin = the mock disc axis (stars do not rotate, so from the cold gas)",
            ang < 3.0 and info["from"] == "cold gas", "%.2f deg from %s" % (ang, info["from"]))
    rh = maps.stellar_half_mass_radius_kpc(snap)
    e20, why = maps.galaxy_view_extent_kpc(snap)
    lo, hi = maps.GALAXY_VIEW_RANGE_KPC
    c.check("maps: fig 20 half-width = 3 stellar half-mass radii (clamped), not the fixed --extent",
            rh is not None and np.isclose(e20, np.clip(3.0 * rh, lo, hi)), "r_half %s -> %s" % (rh, why))
    snap.parameters.update(SofteningTypeOfPartType1=1, SofteningTypeOfPartType4=2, SofteningComovingType1=1.0,
                           SofteningMaxPhysType1=0.5, SofteningComovingType2=0.5, SofteningMaxPhysType2=0.1)
    e = maps.softening_kpc(snap)
    exp_ = max(min(1.0 * snap.a, 0.5), min(0.5 * snap.a, 0.1)) * snap.units.length_to_kpc(1.0)
    c.check("maps: softening = max over DM/stars of min(comoving x a, max physical)", np.isclose(e, exp_))
    for k in ("SofteningTypeOfPartType1", "SofteningTypeOfPartType4", "SofteningComovingType1", "SofteningMaxPhysType1",
              "SofteningComovingType2", "SofteningMaxPhysType2"):
        snap.parameters.pop(k, None)


def _check_slr_mesh(c):
    """SLR_BH refinement diagnostic: the rules mirrored from the C++, without a snapshot."""
    from . import slr_mesh as sm
    from .units import GAMMA, GAMMA_MINUS1, Units

    print("\n[SLR mesh]")
    # R_B at the formula's reference point: 1e7 Msun, c_s = 30 km/s -> 50 pc (kpc/Msun/km/s units, h = 1)
    u = Units(UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e33, UnitVelocity_in_cm_per_s=1e5,
              HubbleParam=1.0, ComovingIntegrationOn=False)
    rb = sm.code_bondi_radius(1e7, 30.0 ** 2 / (GAMMA * GAMMA_MINUS1), u)
    c.check("slr_mesh: R_B = 50 pc at 1e7 Msun and 30 km/s (blackhole_refinement.cc:55-58)",
            np.isclose(rb, 0.05, rtol=1e-6), "got %g kpc" % rb)
    s = dict(fixed=True, force=False, radius=1.0, rmin=0.01, rmax=0.1, min_mass=10.0, lower_c=2.0, target_mass=None)
    R, rmin, rmax = sm.bh_region(s, 0.5, 0.05)
    c.check("slr_mesh: fixed region and ramp", (R, rmin, rmax) == (1.0, 0.01, 0.1)
            and np.isclose(sm.target_radius(0.5, R, rmin, rmax), 0.055))
    s2 = dict(s, fixed=False, radius_hsml=4.0, rmax_hsml=0.1, rmin_rb=0.5)
    c.check("slr_mesh: r_min = r_max/1e4 when r_min > r_max (:237-241)",
            sm.bh_region(s2, 1.0, 10.0)[1] == 0.1 / 1e4)
    # cells: [on target, too big, too big at floor, too small, below floor/4, outside]
    tg = np.array([0.05, 0.05, 0.05, 0.05, 0.05, np.inf])
    rc = np.array([0.04, 0.08, 0.08, 0.01, 0.04, 0.3])
    ms = np.array([100.0, 100.0, 5.0, 100.0, 1.0, 100.0])
    st = sm.classify(rc, ms, tg, s, target_mass=1000.0)
    c.check("slr_mesh: refine/derefine classes as criterion_(de)refinement.cc",
            list(st) == [sm.ON_TARGET, sm.TOO_BIG, sm.AT_FLOOR, sm.TOO_SMALL, sm.BELOW_FLOOR, sm.OUTSIDE],
            "got %s" % list(st))
    # the slice is the Voronoi tessellation: every pixel's label is its nearest point
    try:
        rng = np.random.default_rng(7)
        d = rng.uniform(-1, 1, (400, 3))
        ext, lab, _ = sm.voronoi_slice(d, np.full(400, 0.05), 0.8, n=60)
        g = np.linspace(-0.8, 0.8, 60)
        X, Y = np.meshgrid(g, g)
        q = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        brute = np.argmin(((q[:, None, :] - d[None, :, :]) ** 2).sum(-1), axis=1).reshape(60, 60)
        c.check("slr_mesh: slice labels = nearest generating point (the Voronoi cells)", np.array_equal(lab, brute))
    except ImportError:
        c.check("slr_mesh: scipy installed (for the slice)", False)


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
        from . import kinematics as kin_

        kst_, kwhy_ = kin_.bound_stars(snap)
        rows = diagnostics.figure_applicability(
            summ, logs, [os.path.join(figdir, f) for f in os.listdir(figdir) if f.endswith((".png", ".html"))],
            no_kinematics=None if kst_ is not None else kwhy_)
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
