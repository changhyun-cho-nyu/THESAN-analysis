"""
Synthetic AREPO2 run generator, used by :mod:`thesan_tmz.selftest`.

Builds a small but structurally faithful run directory: split-file HDF5
snapshots with /Header, /Parameters and /Config groups, FoF+Subfind
catalogues, all the ASCII logs (including deliberately truncated final lines,
which a live run always has), and a stdout file with Sync-Point markers and
BH_BIPOLAR_INJECTION summaries.

The gas is laid out as a cold rotating disc plus two hot, fast polar lobes
around a known axis, so the collimation diagnostic has a signal with a known
right answer -- that is what makes this useful as a test rather than a smoke
screen.  Nothing here is physically self-consistent; it exists to exercise
every code path.
"""

import os
import shutil

import h5py
import numpy as np

#: The axis the synthetic bipolar structure is built around.  The collimation
#: test must recover this to within a few degrees.
TRUE_AXIS = np.array([0.3, 0.2, 0.93])
TRUE_AXIS = TRUE_AXIS / np.linalg.norm(TRUE_AXIS)

#: Half-angle of the synthetic hot polar lobes, in degrees (cos = 0.7).
TRUE_LOBE_HALF_ANGLE = float(np.degrees(np.arccos(0.7)))


#: The feedback configurations this package has to cope with.  They differ in
#: which *logs exist at all*, which is the thing that breaks analysis code.
#: Which logs a configuration writes is decided by :func:`logs_written`, which
#: encodes the C++ rules; the scenarios only list the Config options.
#: ``BH_BIPOLAR_INJECTION`` itself never appears: the code derives it from the
#: _THERMAL/_KINETIC switches (allvars.h), so a real /Config never lists it.
SCENARIOS = {
    # The production configuration: local injection for stars and BHs, both
    # collimated channels with the Eddington mode switch, and the energy ledger.
    "local_both": dict(
        config=("BLACK_HOLES", "BH_TORQUE_LIMITED_ACCRETION",
                "BH_BIPOLAR_INJECTION_THERMAL", "BH_BIPOLAR_INJECTION_KINETIC",
                "BH_BIPOLAR_INJECTION_MODE_SWITCH", "LOCAL_INJECTION",
                "LOCAL_INJECTION_BLACK_HOLES", "LOCAL_INJECTION_BH_LEDGER",
                "VORONOI", "COOLING", "GFM"),
        details="torque", bipolar_params=True,
    ),
    # The same collimated channels deposited by the kernel/cone pass: the only
    # build that prints the per-step injection counters.
    "cone_bipolar": dict(
        config=("BLACK_HOLES", "BH_TORQUE_LIMITED_ACCRETION",
                "BH_BIPOLAR_INJECTION_THERMAL", "BH_BIPOLAR_INJECTION_KINETIC",
                "BH_BIPOLAR_INJECTION_MODE_SWITCH", "LOCAL_INJECTION",
                "VORONOI", "COOLING", "GFM"),
        details="torque", bipolar_params=True,
    ),
    # Curtis's legacy BH_BIPOLAR_FEEDBACK: the only writer of blackhole_bipolar.
    "legacy_bipolar": dict(
        config=("BLACK_HOLES", "BH_BONDI_DEFAULT", "BH_THERMALFEEDBACK", "BH_BIPOLAR_FEEDBACK",
                "VORONOI", "COOLING", "GFM"),
        details="default", bipolar_params=False,
    ),
    # LOCAL_INJECTION_BLACK_HOLES on its own (stars keep kernel feedback),
    # isotropic thermal, Bondi, with the ledger.
    "local_bh_only": dict(
        config=("BLACK_HOLES", "BH_BONDI_DEFAULT", "BH_THERMALFEEDBACK",
                "LOCAL_INJECTION_BLACK_HOLES", "LOCAL_INJECTION_BH_LEDGER",
                "VORONOI", "COOLING", "GFM"),
        details="default", bipolar_params=False,
    ),
    # LOCAL_INJECTION for the stars, kernel-weighted deposition for the black
    # holes, isotropic.
    "local_stars_sph_bh": dict(
        config=("BLACK_HOLES", "BH_BONDI_DEFAULT", "BH_THERMALFEEDBACK", "LOCAL_INJECTION",
                "VORONOI", "COOLING", "GFM"),
        details="default", bipolar_params=False,
    ),
    # Kernel-weighted deposition for both.  The baseline model.
    "sph_both": dict(
        config=("BLACK_HOLES", "BH_BONDI_DEFAULT", "BH_THERMALFEEDBACK", "VORONOI", "COOLING", "GFM"),
        details="default", bipolar_params=False,
    ),
}


def logs_written(config):
    """Which black-hole logs the C++ writes for a given set of Config options.

    Read off arepo ``Arepo2-TMZ-newbh`` at ``0b41480f0b``:

      * ``blackhole_bipolar/``   only BH_BIPOLAR_FEEDBACK (blackhole_mdot.cc:438)
      * "injected thermal = ..." counters on stdout: the cone pass of
        BH_BIPOLAR_INJECTION_THERMAL/KINETIC (blackhole_bipolar_injection.cc:700/705);
        with LOCAL_INJECTION_BLACK_HOLES that pass stops after the axis and
        prints "axis only ..." instead (blackhole_bipolar_injection.cc:551)
      * "LOCAL_INJECTION BH LEDGER: ..." on stdout: LOCAL_INJECTION_BH_LEDGER
        with LOCAL_INJECTION_BLACK_HOLES (local_injection.cc:926)
    """
    cfg = set(config)
    channels = bool({"BH_BIPOLAR_INJECTION_THERMAL", "BH_BIPOLAR_INJECTION_KINETIC"} & cfg)
    li_bh = "LOCAL_INJECTION_BLACK_HOLES" in cfg
    return dict(
        bipolar_log="BH_BIPOLAR_FEEDBACK" in cfg,
        counters=channels and not li_bh,
        axis_only=channels and li_bh,
        ledger=li_bh and "LOCAL_INJECTION_BH_LEDGER" in cfg,
    )


#: Radiative efficiency of the synthetic run: 0.2 like the production runs, and
#: deliberately not 0.1, so an analysis that silently assumes 0.1 fails a check.
EPS_R = 0.2


def _mdot_edd_code(bh_mass_code, h, UL, UV):
    """BH_MdotEddington as blackhole_mdot_eddington() writes it, in code units."""
    G, mp, c, sigma_T = 6.6738e-8, 1.67262178e-24, 2.99792458e10, 6.65245873e-25
    return 4 * np.pi * G * c * mp / (EPS_R * c * c * sigma_T) * bh_mass_code * (UL / UV) / h


def build(root, nsnap=12, seed=42, scenario="local_both"):
    """Create a synthetic run under ``root``.  Returns the run directory.

    ``scenario`` selects one of :data:`SCENARIOS` -- which Config options the
    snapshots claim, which black-hole logs exist, and which accretion layout
    ``blackhole_details`` is written in.
    """
    cfg = SCENARIOS[scenario]
    logs = logs_written(cfg["config"])
    rng = np.random.default_rng(seed)
    ROOT = root
    OUT = os.path.join(ROOT, "output")
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(OUT)
    logdirs = ["blackhole_details", "blackhole_mergers"]
    if logs["bipolar_log"]:
        logdirs.append("blackhole_bipolar")
    for d in logdirs:
        os.makedirs(os.path.join(OUT, d))
    # THESAN units
    UL, UM, UV = 3.085678e21, 1.989e43, 1.0e5
    h, Om, OL, Ob = 0.6774, 0.3089, 0.6911, 0.0486
    BOX = 5000.0  # ckpc/h
    M2MSUN = UM / 1.989e33 / h  # 1.476e10

    NSNAP = nsnap
    avals = np.linspace(1.0 / 16.0, 1.0 / 5.0, NSNAP)  # z = 15 -> 4

    # The fine time grid of the ASCII logs, and the energy the BHs accreted on it.
    # Every AGN energy the mock writes (ledger, BH_CumEgyInjection_QM) is
    # efficiency x eps_r x this, exactly as blackhole_mdot.cc fills the
    # reservoirs, so the energy book of the self-test closes to round-off.
    from .units import CLIGHT, SOLAR_MASS, Units

    nt = 900
    tt = np.linspace(avals[0], avals[-1], nt)
    grow = (tt / tt[0]) ** 3
    burst = 1.0 + 3.0 * np.abs(np.sin(np.linspace(0, 25, nt)))
    mdot_msun_yr = 3e-6 * grow * burst * 10.227          # column 5 of blackholes.txt
    u_mock = Units(UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV, HubbleParam=h,
                   Omega0=Om, OmegaLambda=OL, OmegaBaryon=Ob)
    age = u_mock.age_gyr(tt)
    acc_msun = np.concatenate([[0.0], np.cumsum(0.5 * (mdot_msun_yr[1:] + mdot_msun_yr[:-1]) * np.diff(age) * 1e9)])
    e_acc_code = acc_msun * SOLAR_MASS * CLIGHT ** 2 / u_mock.energy_to_erg   # integral of Mdot c^2 dt, code energy
    eps_f, eps_th, eps_kin = 0.05, 0.05, 0.02

    CEN = np.array([2500.0, 2500.0, 2500.0])
    AXIS = TRUE_AXIS


    def write_snapshot(num, a, nfiles=2):
        z = 1.0 / a - 1.0
        growth = (a / avals[0]) ** 3
        ngas, ndm, nstar, nbh = 4000, 3000, 800, 2

        # gas: a flattened disc plus a hot halo, so the collimation test has signal
        ndisc = int(0.7 * ngas)
        nhalo = ngas - ndisc
        rd = rng.exponential(1.0, ndisc)
        ph = rng.uniform(0, 2 * np.pi, ndisc)
        zd = rng.normal(0, 0.15, ndisc)
        disc = np.stack([rd * np.cos(ph), rd * np.sin(ph), zd], axis=1)
        halo = rng.normal(0, 6.0, (nhalo, 3))
        gpos_local = np.vstack([disc, halo])

        # rotate the disc so its axis is AXIS
        tmp = np.array([1.0, 0, 0])
        e1 = np.cross(tmp, AXIS); e1 /= np.linalg.norm(e1)
        e2 = np.cross(AXIS, e1)
        R = np.vstack([e1, e2, AXIS])
        gpos_local = gpos_local @ R

        gpos = CEN + gpos_local / (a / h)  # comoving code units from physical kpc
        r = np.linalg.norm(gpos_local, axis=1)
        cosang = np.abs(gpos_local @ AXIS / np.maximum(r, 1e-6))

        # temperature: hot bipolar lobes, cold disc
        Tphys = np.where(cosang > 0.7, 10 ** rng.normal(6.3, 0.3, ngas), 10 ** rng.normal(3.8, 0.4, ngas))
        u_code = Tphys * 1.380649e-16 / (0.6 * 1.6726e-24) / (2.0 / 3.0) / UV ** 2
        xe = np.where(Tphys > 1e5, 1.16, 0.02)

        dens = 1e-2 * growth / (1.0 + r ** 2)

        # The disc must actually rotate about AXIS, otherwise the net angular
        # momentum is pure noise and nothing downstream can recover the axis --
        # which is exactly the situation the code's own "no axis" counter exists
        # to report, so the mock has to get this right to test the good path.
        rperp = gpos_local - np.outer(gpos_local @ AXIS, AXIS)
        rp = np.linalg.norm(rperp, axis=1)
        phihat = np.cross(AXIS, rperp / np.maximum(rp[:, None], 1e-9))
        vrot = 150.0 * rp / np.sqrt(rp ** 2 + 1.0)  # rises then flattens at ~150 km/s
        vel = rng.normal(0, 15, (ngas, 3)) + phihat * vrot[:, None]

        # outflow along the axis
        outflow = cosang > 0.7
        vel[outflow] += (gpos_local[outflow] / np.maximum(r[outflow, None], 1e-6)) * 300.0

        fields = {
            0: dict(
                Coordinates=gpos,
                Masses=np.full(ngas, 1e-5),
                Velocities=vel,
                Density=dens,
                InternalEnergy=u_code,
                ElectronAbundance=xe,
                StarFormationRate=np.where((Tphys < 1e4) & (r < 3), rng.exponential(0.02, ngas), 0.0),
                GFM_Metallicity=np.full(ngas, 0.002 * growth),
                ParticleIDs=np.arange(1, ngas + 1, dtype=np.uint64),
            ),
            1: dict(
                Coordinates=CEN + rng.normal(0, 30, (ndm, 3)),
                Velocities=rng.normal(0, 80, (ndm, 3)),
                Potential=-rng.exponential(1e4, ndm),
                ParticleIDs=np.arange(10_000, 10_000 + ndm, dtype=np.uint64),
            ),
            4: dict(
                Coordinates=CEN + (rng.normal(0, 1.2, (nstar, 3)) @ R) / (a / h),
                Masses=np.full(nstar, 8e-6 * growth),
                Velocities=rng.normal(0, 60, (nstar, 3)),
                GFM_StellarFormationTime=rng.uniform(avals[0] * 0.8, a, nstar),
                GFM_InitialMass=np.full(nstar, 1e-5),
                GFM_Metallicity=np.full(nstar, 0.001 * growth),
                Potential=-rng.exponential(2e4, nstar),
                ParticleIDs=np.arange(50_000, 50_000 + nstar, dtype=np.uint64),
            ),
            5: dict(
                Coordinates=np.array([CEN, CEN + np.array([0.4, 0.1, 0.0])]),
                Masses=np.array([2e-4, 3e-5]) * growth,
                BH_Mass=np.array([1.5e-4, 2e-5]) * growth,
                BH_Mdot=np.array([3e-6, 4e-7]) * growth,
                BH_MdotBondi=np.array([5e-6, 6e-7]) * growth,
                BH_MdotEddington=_mdot_edd_code(np.array([1.5e-4, 2e-5]) * growth, h, UL, UV),
                BH_Hsml=np.array([1.2, 1.0]),
                BH_Density=np.array([0.02, 0.01]),
                BH_U=np.array([50.0, 40.0]),
                Velocities=rng.normal(0, 10, (2, 3)),
                ParticleIDs=np.array([900001, 900002], dtype=np.uint64),
                **({"BH_CumEgyInjection_QM": eps_f * EPS_R * np.interp(a, tt, e_acc_code) * np.array([0.8, 0.2])}
                   if "BH_THERMALFEEDBACK" in cfg["config"] else {}),
            ),
        }

        npart_total = np.array([ngas, ndm, 0, 0, nstar, nbh], dtype=np.int64)
        snapdir = os.path.join(OUT, "snapdir_%03d" % num)
        os.makedirs(snapdir, exist_ok=True)

        # split across nfiles
        splits = {pt: np.array_split(np.arange(npart_total[pt]), nfiles) for pt in fields}
        for k in range(nfiles):
            fn = os.path.join(snapdir, "snap_%03d.%d.hdf5" % (num, k))
            with h5py.File(fn, "w") as f:
                hd = f.create_group("Header")
                this = np.array([len(splits[pt][k]) if pt in splits else 0 for pt in range(6)], dtype=np.int64)
                hd.attrs["NumPart_ThisFile"] = this
                hd.attrs["NumPart_Total"] = npart_total.astype(np.uint32)
                hd.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
                hd.attrs["MassTable"] = np.array([0, 3e-4, 0, 0, 0, 0], dtype=np.float64)
                hd.attrs["Time"] = a
                hd.attrs["Redshift"] = z
                hd.attrs["BoxSize"] = BOX
                hd.attrs["NumFilesPerSnapshot"] = nfiles
                hd.attrs["Omega0"] = Om
                hd.attrs["OmegaLambda"] = OL
                hd.attrs["HubbleParam"] = h
                hd.attrs["ComovingIntegrationOn"] = 1

                pg = f.create_group("Parameters")
                for kk, vv in dict(
                    UnitLength_in_cm=UL, UnitMass_in_g=UM, UnitVelocity_in_cm_per_s=UV,
                    HubbleParam=h, Omega0=Om, OmegaLambda=OL, OmegaBaryon=Ob, ComovingIntegrationOn=1, BoxSize=BOX,
                    BlackHoleRadiativeEfficiency=EPS_R, BlackHoleFeedbackFactor=eps_f, BlackHoleAccretionFactor=1.0,
                    BlackHoleEddingtonFactor=1.0,
                ).items():
                    pg.attrs[kk] = vv
                if cfg["bipolar_params"]:
                    for kk, vv in dict(
                        BHBipolarInjEddThreshold=0.01, BHBipolarInjThermalTheta=30.0,
                        BHBipolarInjKineticTheta=20.0, BHBipolarInjThermalEfficiency=eps_th,
                        BHBipolarInjKineticEfficiency=eps_kin, BHBipolarInjColdFraction=0.2,
                        BHBipolarInjColdTemp=1e5, BHBipolarInjMaxVel=3000.0,
                    ).items():
                        pg.attrs[kk] = vv

                cg = f.create_group("Config")
                for opt in cfg["config"]:
                    cg.attrs[opt] = ""

                for pt, data in fields.items():
                    idx = splits[pt][k]
                    if idx.size == 0:
                        continue
                    g = f.create_group("PartType%d" % pt)
                    for name, arr in data.items():
                        g.create_dataset(name, data=np.asarray(arr)[idx])

        # ---- group catalogue ----
        gdir = os.path.join(OUT, "groups_%03d" % num)
        os.makedirs(gdir, exist_ok=True)
        ngroups, nsubs = 3, 5
        with h5py.File(os.path.join(gdir, "fof_subhalo_tab_%03d.0.hdf5" % num), "w") as f:
            hd = f.create_group("Header")
            hd.attrs["Ngroups_Total"] = ngroups
            hd.attrs["Ngroups_ThisFile"] = ngroups
            # AREPO2 spelling (fof_io.cc:948).  The mock used to write the
            # legacy Nsubgroups_* names, which meant the readers were only ever
            # tested against a catalogue no current run produces.
            hd.attrs["Nsubhalos_Total"] = nsubs
            hd.attrs["Nsubhalos_ThisFile"] = nsubs
            hd.attrs["Time"] = a
            hd.attrs["Redshift"] = z
            hd.attrs["BoxSize"] = BOX
            hd.attrs["HubbleParam"] = h
            hd.attrs["Omega0"] = Om
            hd.attrs["OmegaLambda"] = OL
            hd.attrs["NumFiles"] = 1
            G = f.create_group("Group")
            G.create_dataset("GroupPos", data=np.tile(CEN, (ngroups, 1)) + rng.normal(0, 5, (ngroups, 3)))
            G.create_dataset("Group_M_Crit200", data=np.array([80.0, 8.0, 1.0]) * growth)
            G.create_dataset("Group_R_Crit200", data=np.array([60.0, 25.0, 12.0]))
            G.create_dataset("GroupSFR", data=np.array([3.0, 0.3, 0.02]) * growth)
            G.create_dataset("GroupFirstSub", data=np.array([0, 2, 4], dtype=np.int32))
            G.create_dataset("GroupNsubs", data=np.array([2, 2, 1], dtype=np.int32))
            S = f.create_group("Subhalo")
            mt = np.zeros((nsubs, 6))
            mt[:, 0] = np.array([2.0, 0.5, 0.4, 0.1, 0.05]) * growth
            mt[:, 1] = np.array([70.0, 5.0, 6.0, 1.0, 0.6]) * growth
            mt[:, 4] = np.array([1.0, 0.2, 0.15, 0.03, 0.01]) * growth
            mt[:, 5] = np.array([2e-4, 1e-5, 3e-5, 0.0, 0.0]) * growth
            # The Group holds every subhalo of the system, so its mass is the
            # sum over them -- several galaxies, not one.  Writing it that way
            # is what makes the group-vs-subhalo distinction testable.
            grpnr = np.array([0, 0, 1, 1, 2])
            gmt = np.array([mt[grpnr == k].sum(axis=0) for k in range(ngroups)])
            G.create_dataset("GroupMassType", data=gmt)
            G.create_dataset("GroupMass", data=gmt.sum(axis=1))
            S.create_dataset("SubhaloMassType", data=mt)
            S.create_dataset("SubhaloMassInRadType", data=mt * 0.7)
            S.create_dataset("SubhaloLenType", data=(mt * 5000).astype(np.int32))
            S.create_dataset("SubhaloBHMass", data=mt[:, 5])
            S.create_dataset("SubhaloBHMdot", data=mt[:, 5] * 1e-2)
            S.create_dataset("SubhaloSFR", data=np.array([3.0, 0.3, 0.2, 0.01, 0.0]) * growth)
            S.create_dataset("SubhaloSfrInRad", data=np.array([2.5, 0.25, 0.15, 0.008, 0.0]) * growth)
            S.create_dataset("SubhaloVelDisp", data=np.array([120.0, 45.0, 50.0, 20.0, 15.0]))
            S.create_dataset("SubhaloVmax", data=np.array([200.0, 70.0, 80.0, 30.0, 22.0]))
            rh = np.zeros((nsubs, 6)); rh[:, 4] = np.array([1.2, 0.6, 0.7, 0.3, 0.2])
            S.create_dataset("SubhaloHalfmassRadType", data=rh)
            S.create_dataset("SubhaloGasMetallicity", data=np.full(nsubs, 0.002) * growth)
            S.create_dataset("SubhaloStarMetallicity", data=np.full(nsubs, 0.0015) * growth)
            spos = np.tile(CEN, (nsubs, 1))
            spos[1:] += np.array([[6.0, 0, 0], [0, 40.0, 0], [0, 42.0, 3.0], [0, 0, 90.0]])
            S.create_dataset("SubhaloPos", data=spos)
            S.create_dataset("SubhaloVel", data=rng.normal(0, 60, (nsubs, 3)))
            # Real AREPO2 names (fof_io.cc:186,189,293).  SubhaloGrNr,
            # SubhaloFlag and SubhaloSFRinRad appear nowhere in the AREPO2
            # source -- writing them here is what let the readers fall back to
            # zeros on real data while the selftest passed.
            S.create_dataset("SubhaloGroupNr", data=grpnr.astype(np.int64))
            S.create_dataset("SubhaloRankInGr", data=np.array([0, 1, 0, 1, 0], dtype=np.int32))
            S.create_dataset("SubhaloHighResStellarFraction", data=np.array([1.0, 1.0, 0.99, 0.98, 0.2]))


    for i, a in enumerate(avals):
        write_snapshot(i, a)

    # ---- merger tree (THESAN-Zoom writes output/tree.hdf5) ----
    # One row per snapshot along the main branch, which is what the real one
    # holds; the values match the central subhalo written above so the tree and
    # the catalogues can be checked against each other.
    with h5py.File(os.path.join(OUT, "tree.hdf5"), "w") as f:
        gr = (avals / avals[0]) ** 3
        f.create_dataset("Redshifts", data=1.0 / avals - 1.0)
        f.create_dataset("Snapshots", data=np.arange(NSNAP, dtype=np.int32))
        T = f.create_group("Subhalo")
        mt = np.zeros((NSNAP, 6))
        mt[:, 0] = 2.0 * gr
        mt[:, 1] = 70.0 * gr
        mt[:, 4] = 1.0 * gr
        mt[:, 5] = 2e-4 * gr
        T.create_dataset("SubhaloMassType", data=mt)
        T.create_dataset("SubhaloMassInRadType", data=mt * 0.7)
        T.create_dataset("SubhaloBHMass", data=mt[:, 5])
        T.create_dataset("SubhaloSFR", data=3.0 * gr)
        T.create_dataset("SubhaloSfrInRad", data=2.5 * gr)
        T.create_dataset("SubhaloVelDisp", data=np.full(NSNAP, 120.0))
        T.create_dataset("SubhaloPos", data=np.tile(CEN, (NSNAP, 1)))
        Gt = f.create_group("Group")
        Gt.create_dataset("Group_M_Crit200", data=80.0 * gr)
        Gt.create_dataset("Group_R_Crit200", data=np.full(NSNAP, 60.0))

    # ---- expansion list, as the THESAN scripts use to map z -> snapshot ----
    np.savetxt(os.path.join(root, "ExpansionList.list"), avals)

    # ------------------------------------------------------------------ logs
    with open(os.path.join(OUT, "blackholes.txt"), "w") as f:
        for t, g, b in zip(tt, grow, burst):
            mass = 1.7e-4 * g
            mdot = 3e-6 * g * b
            f.write("%14e %6d  %14e %14e %14e %14e %14e\n"
                    % (t, 2, mass, mdot, mdot * 10.227, mass * 1.05, mdot / (2e-5)))

    with open(os.path.join(OUT, "sfr.txt"), "w") as f:
        cum = 0.0
        for t, g, b in zip(tt, grow, burst):
            sfr = 2.0 * g * b
            cum += sfr * 1e-6
            f.write("%14e %14e %14e %14e %14e %14e\n" % (t, sfr * 1e-7, sfr / 10.227, sfr, cum * 1.1, cum))

    with open(os.path.join(OUT, "energy.txt"), "w") as f:
        for i, (t, g) in enumerate(zip(tt, grow)):
            vals = [t, 1e3 * g, -5e4 * g, 2e3 * g] + [1.0 * g] * 18 + [10.0 * g] * 6 + [1e2 * g, 0.0]
            f.write(" ".join("%g" % v for v in vals) + "\n")

    # blackhole_details: torque layout (7 cols) or the default Bondi one (6)
    for rank in range(2):
        with open(os.path.join(OUT, "blackhole_details", "blackhole_details_%d.txt" % rank), "w") as f:
            for j, (t, g, b) in enumerate(zip(tt, grow, burst)):
                if j % 2 != rank:
                    continue
                for bid, sc in ((900001, 1.0), (900002, 0.1)):
                    if cfg["details"] == "torque":
                        f.write("BH=%d %g %g %g %g %g %g\n"
                                % (bid, t, 1.5e-4 * g * sc, 3e-6 * g * b * sc, 0.02 * g,
                                   0.3 + 0.3 * np.sin(j * 0.05), 0.5 + 0.2 * np.cos(j * 0.03)))
                    else:
                        f.write("BH=%d %g %g %g %g %g\n"
                                % (bid, t, 1.5e-4 * g * sc, 3e-6 * g * b * sc, 0.02 * g, 12.0 + 3.0 * np.sin(j * 0.05)))

    # blackhole_bipolar: 9 cols, axis wandering slowly.  Only BH_BIPOLAR_FEEDBACK
    # writes it (see logs_written).
    if logs["bipolar_log"]:
        for rank in range(2):
            with open(os.path.join(OUT, "blackhole_bipolar", "blackhole_bipolar_%d.txt" % rank), "w") as f:
                for j, t in enumerate(tt):
                    if j % 2 != rank:
                        continue
                    ang = 0.15 * np.sin(j * 0.02) + rng.normal(0, 0.03)
                    n = AXIS + np.array([ang, -ang * 0.5, 0.0])
                    n = n / np.linalg.norm(n) * (1e-3 * (1 + j))
                    cf = 0.3 + 0.25 * np.sin(j * 0.04)
                    f.write("BH=%d %g %g %g %g %g %d %d %g\n"
                            % (900001, t, n[0], n[1], n[2], 0.8, int(cf > 0.2), 1, cf))

    with open(os.path.join(OUT, "blackhole_mergers", "blackhole_mergers_0.txt"), "w") as f:
        f.write("0 %g 900001 1.2e-4 900003 5e-6\n" % tt[nt // 3])
        f.write("1 %g 900001 1.6e-4 900004 8e-6\n" % tt[2 * nt // 3])

    # stdout with Sync-Points and injection summaries
    with open(os.path.join(ROOT, "run.out"), "w") as f:
        f.write("AREPO2 starting\n")
        for j, (t, g, b) in enumerate(zip(tt, grow, burst)):
            z = 1.0 / t - 1.0
            f.write("Sync-Point %d, Time: %g, Redshift: %g, Systemstep: 1e-4, Dloga: 1e-4\n" % (j, t, z))
            if logs["counters"] and j % 3 == 0:
                th = 1e-3 * g * b if b > 2 else 0.0
                kin = 0.0 if b > 2 else 4e-4 * g
                nth, nk = (2, 0) if b > 2 else (0, 2)
                f.write("BH_BIPOLAR_INJECTION: injected thermal = %g, kinetic = %g (code units); "
                        "mode split (thermal/kinetic) = %d/%d; no-axis = %d, cold-disk fallback = %d, "
                        "empty cone = %d, clipped cells = %d; took %g sec\n"
                        % (th, kin, nth, nk, 1 if j % 97 == 0 else 0, 1 if j % 31 == 0 else 0,
                           1 if j % 17 == 0 else 0, int(rng.integers(0, 5)), 0.001 + 0.002 * rng.random()))
            if (logs["axis_only"] or logs["ledger"]) and j % 3 == 0:
                # what a LOCAL_INJECTION_BLACK_HOLES step prints instead
                if logs["axis_only"]:
                    f.write("BH_BIPOLAR_INJECTION: Start collimated feedback injection\n")
                    f.write("BH_BIPOLAR_INJECTION: axis only (deposition handed to LOCAL_INJECTION_BLACK_HOLES)\n")
                f.write("LOCAL_INJECTION BLACK HOLE FEEDBACK: Evaluating 2 black holes\n")
                f.write("LOCAL_INJECTION BLACK HOLE FEEDBACK: inject energy\n")
                if logs["ledger"]:
                    de = e_acc_code[j] - e_acc_code[max(j - 3, 0)] if j else 0.0   # accreted since the last line
                    iso = eps_f * EPS_R * de if "BH_THERMALFEEDBACK" in cfg["config"] else 0.0
                    # with the mode switch a step feeds one collimated channel: thermal when bursting
                    coll = (eps_th * EPS_R * de if b > 2 else 0.0) if logs["axis_only"] else 0.0
                    kin = (0.0 if b > 2 else eps_kin * EPS_R * de) if logs["axis_only"] else 0.0
                    f.write("LOCAL_INJECTION BH LEDGER: drained iso %g coll %g kin %g | thermal asked %g applied %g | "
                            "kinetic asked %g applied %g | |sum dp|/sum|dp| %g\n"
                            % (iso, coll, kin, iso + coll, 0.999 * (iso + coll), kin, 0.98 * kin, 1e-3 if kin else 0.0))
        # a deliberately truncated final line, like a killed run
        if logs["counters"]:
            f.write("BH_BIPOLAR_INJECTION: injected therm")
        elif logs["ledger"]:
            f.write("LOCAL_INJECTION BH LEDGER: drained iso 1.2e")

    # a truncated last line in blackholes.txt too
    with open(os.path.join(OUT, "blackholes.txt"), "a") as f:
        f.write("1.9e-01 2  1.2e-04 3.1e")

    return ROOT

