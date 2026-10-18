"""
Galaxy-scale measurements from a snapshot: centring, apertures, gas phases,
kinematics.

This is the layer between raw particle arrays and the plots, and it is
deliberately only half the story:

  * **Integrated galaxy properties come from the Subfind catalogue**
    (:mod:`thesan_tmz.catalog`) -- stellar mass, SFR, half-mass radius,
    velocity dispersion, BH mass.  Subfind has already decided which particles
    belong to which galaxy, and for a THESAN-Zoom run that decision is the
    whole problem: the FoF *Group* is the entire zoom system, several galaxies
    at once, so a Group-level number is never a galaxy-level number.  An
    individual galaxy is a *Subhalo*.
  * **Spatially resolved quantities come from the snapshot particles** -- the
    profiles, phase diagram, projections and the jet-collimation test.  Those
    are questions about the gas *inside* a galaxy, which no catalogue answers.

:func:`measure` does both and labels which is which: catalogue-derived entries
carry a ``sub_`` prefix, particle-derived ones an ``_r<radius>`` suffix.  Where
the two overlap (stellar mass, SFR) that is on purpose -- an aperture sum that
disagrees badly with the catalogue means the centre is wrong, and that check is
worth having.
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from . import catalog as cat_mod
from .units import GRAVITY, KPC_IN_CM, PROTONMASS, SOLAR_MASS

# Gas phase boundaries in K.  These are the conventional ISM/CGM cuts; change
# them here and every figure in the package follows.
T_COLD = 1.0e4
T_WARM = 1.0e5
#: A cold gas cell only counts as "dense/ISM" above this hydrogen density.
NH_DENSE = 0.1  # cm^-3


# ---------------------------------------------------------------------------
# Centring
# ---------------------------------------------------------------------------


def centre_on_bh(snap, bh_id=None, cat=None, in_target_galaxy=True):
    """Centre on a black hole: the most massive one, or a specified ID.

    ``in_target_galaxy`` first asks the Subfind catalogue which galaxy this run
    is about and then takes the most massive black hole *inside that galaxy*.
    Without it, "most massive black hole in the snapshot" is a statement about
    the whole zoom volume, and in a zoom the volume contains other galaxies:
    once a satellite's black hole overtakes the central's -- which happens
    during a merger, exactly when the figures matter -- every plot silently
    jumps to a different object.  The cut is generous (five stellar half-mass
    radii) because a black hole that has been kicked or is still sinking is
    still the black hole of that galaxy.

    Returns ``(centre_code, info)``.  ``info`` carries the ID and BH_Mass so
    the caller can record which object the figure is about.
    """
    d = snap.part(io.PT_BH, ["Coordinates", "BH_Mass", "ParticleIDs", "Masses", "Velocities"])
    if d.get("count", 0) == 0 or "Coordinates" not in d:
        return None, {"reason": "no black holes in snapshot"}

    mass = np.asarray(d.get("BH_Mass", d.get("Masses")), dtype=np.float64)
    allowed = np.ones(mass.size, dtype=bool)
    host = {}
    if bh_id is None and in_target_galaxy:
        try:
            if cat is None:
                cat = cat_mod.load(snap)
            if cat is not None and cat.n_sub:
                j = cat.target_index()
                pos = cat.s("pos")
                if j is not None and pos is not None:
                    L = snap.units.length_to_kpc(snap.a)
                    props = cat.properties(j)
                    rlim = 5.0 * props.get("rhalf_star_kpc", 0.0)
                    if not (rlim > 0):
                        rlim = max(5.0, props.get("r200c_kpc", 0.0) * 0.1)
                    r = io.radius_from(d["Coordinates"], np.asarray(pos)[j], snap.boxsize) * L
                    near = r <= rlim
                    if near.any():
                        allowed = near
                        # Only claim this galaxy when the black hole is really
                        # in it.  If nothing was inside the cut we fall back to
                        # the whole snapshot, and then the chosen black hole may
                        # belong to some other galaxy -- leave subhalo_index
                        # unset so measure() resolves it from the centre.
                        host = {"host_subhalo_index": j, "host_radius_kpc": float(rlim),
                                "subhalo_index": j}
                # Hand the catalogue on either way, so measure() does not open
                # every chunk of it a second time.
                host["_catalog"] = cat
        except (FileNotFoundError, OSError, KeyError, IndexError):
            pass

    if bh_id is None:
        k = int(np.argmax(np.where(allowed, mass, -np.inf)))
    else:
        hit = np.where(d["ParticleIDs"] == bh_id)[0]
        if hit.size == 0:
            return None, {"reason": "BH id %s not in snapshot" % bh_id}
        k = int(hit[0])

    info = {
        "bh_id": int(d["ParticleIDs"][k]),
        "bh_mass_msun": float(mass[k] * snap.units.mass_to_msun),
        "bh_index": k,
        "n_bh_considered": int(allowed.sum()),
        "n_bh_snapshot": int(mass.size),
    }
    info.update(host)
    if "Velocities" in d:
        info["bh_vel"] = d["Velocities"][k]
    return np.array(d["Coordinates"][k], dtype=np.float64), info


def centre_on_group(snap, group_index=0):
    """Centre on a FoF group's ``GroupPos`` (the potential minimum)."""
    g = snap.groups(["GroupPos", "Group_M_Crit200", "Group_R_Crit200", "GroupMass"])
    if g.get("count", 0) == 0:
        return None, {"reason": "empty group catalogue"}
    if group_index >= g["count"]:
        return None, {"reason": "group %d beyond catalogue (%d groups)" % (group_index, g["count"])}
    info = {"group_index": group_index}
    if "Group_M_Crit200" in g:
        info["M200c_msun"] = float(g["Group_M_Crit200"][group_index] * snap.units.mass_to_msun)
    if "Group_R_Crit200" in g:
        info["R200c_kpc"] = float(g["Group_R_Crit200"][group_index] * snap.units.length_to_kpc(snap.a))
    return np.array(g["GroupPos"][group_index], dtype=np.float64), info


def centre_on_subhalo(snap, prefer="central", bh_id=None, cat=None, group_index=0):
    """Centre on an individual galaxy's ``SubhaloPos``.

    This is the right default for a zoom.  ``GroupPos`` is the potential
    minimum of the whole FoF system -- which is usually, but not always, inside
    the central galaxy, and during a major merger is emphatically not.
    ``SubhaloPos`` is the potential minimum of one galaxy, chosen by Subfind
    with the particle binding information this module does not have.

    ``prefer`` is passed to :meth:`Catalog.target_index`.  With ``bh_id`` set,
    the black hole is located in the snapshot first and the galaxy hosting it
    is the one selected, so the figures follow that black hole rather than
    whichever galaxy happens to be most massive.
    """
    if cat is None:
        cat = cat_mod.load(snap)
    if cat is None or cat.n_sub == 0:
        return None, {"reason": "no Subfind catalogue for snapshot %d" % snap.num}

    bh_pos = None
    if bh_id is not None or prefer == "bh":
        c, _ = centre_on_bh(snap, bh_id=bh_id)
        if c is not None:
            bh_pos, prefer = c, "bh"

    i = cat.target_index(prefer=prefer, bh_pos=bh_pos)
    if i is None:
        return None, {"reason": "no subhalo passed selection"}
    pos = cat.s("pos")
    if pos is None:
        return None, {"reason": "catalogue has no SubhaloPos"}

    props = cat.properties(i)
    info = {"subhalo_index": i, "selection": prefer}
    for k in ("mstar_msun", "sfr_msun_per_yr", "bh_mass_msun", "mhalo_msun", "r200c_kpc",
              "rhalf_star_kpc", "sigma_kms", "is_central", "group_index",
              "highres_stellar_fraction", "nstar"):
        if k in props:
            info[k] = props[k]
    info["_catalog"] = cat
    info["_properties"] = props
    return np.asarray(pos, dtype=np.float64)[i], info


def centre_on_potential_minimum(snap, parttypes=(1, 4)):
    """Centre on the globally deepest ``Potential`` among the given types."""
    best, best_pos = np.inf, None
    for pt in parttypes:
        d = snap.part(pt, ["Coordinates", "Potential"])
        if d.get("count", 0) == 0 or "Potential" not in d:
            continue
        k = int(np.argmin(d["Potential"]))
        if d["Potential"][k] < best:
            best = float(d["Potential"][k])
            best_pos = np.array(d["Coordinates"][k], dtype=np.float64)
    if best_pos is None:
        return None, {"reason": "no Potential field in snapshot"}
    return best_pos, {"potential_min": best}


def shrinking_sphere(pos, mass, centre0, r0, shrink=0.9, min_particles=50, max_iter=200, boxsize=None):
    """Classic shrinking-sphere centre of mass (Power et al. 2003).

    Robust against the centre wandering during a merger, which the plain
    centre of mass of a fixed aperture is not.
    """
    c = np.array(centre0, dtype=np.float64)
    r = float(r0)
    for _ in range(max_iter):
        d = io.radius_from(pos, c, boxsize)
        m = d <= r
        n = int(m.sum())
        if n < min_particles:
            break
        w = mass[m]
        rel = io.periodic_delta(pos[m], c, boxsize)
        c = c + np.sum(rel * w[:, None], axis=0) / np.sum(w)
        r *= shrink
        if r <= 0:
            break
    return c


def centre_on_densest_mass(snap, parttype=None, refine=True):
    """Shrinking-sphere centre of the high-resolution mass.  The last resort.

    Used when there is no black hole yet, no group catalogue, and the snapshot
    carries no ``Potential`` field -- the situation at high redshift in a zoom
    run written without OUTPUTPOTENTIAL, where every other method fails.

    Starts from the densest region of the chosen particle type (found on a
    coarse grid, so it is not thrown by a single outlier) and refines with a
    shrinking sphere (Power et al. 2003).  For a zoom, PartType1 is the
    high-resolution dark matter and converges on the main progenitor; gas is
    the fallback for a non-cosmological or gas-only setup.

    Caveat worth stating in a caption: with no halo finder this can lock onto a
    massive satellite rather than the main progenitor during a major merger.
    Prefer 'group' whenever a catalogue exists.
    """
    order = [parttype] if parttype is not None else [io.PT_DM, io.PT_STAR, io.PT_GAS]
    for pt in order:
        d = snap.part(pt, ["Coordinates", "Masses"])
        if d.get("count", 0) == 0 or "Coordinates" not in d:
            continue
        pos = d["Coordinates"]
        mass = d.get("Masses")
        if mass is None:
            mass = np.ones(pos.shape[0])

        # Coarse 3D histogram to find the densest cell, so the starting guess
        # is robust; then shrink onto it.
        nb = 32
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        idx = np.clip(((pos - lo) / span * nb).astype(int), 0, nb - 1)
        flat = (idx[:, 0] * nb + idx[:, 1]) * nb + idx[:, 2]
        H = np.bincount(flat, weights=mass, minlength=nb ** 3)
        k = int(np.argmax(H))
        cell = np.array([k // (nb * nb), (k // nb) % nb, k % nb], dtype=np.float64)
        guess = lo + (cell + 0.5) / nb * span

        if not refine:
            return guess, {"centre_parttype": pt, "refined": False}

        r0 = float(np.max(span)) * 0.25
        c = shrinking_sphere(pos, mass, guess, r0, boxsize=snap.boxsize)
        return c, {"centre_parttype": pt, "refined": True, "start_radius_kpc": r0 * snap.units.length_to_kpc(snap.a)}
    return None, {"reason": "no particles with coordinates"}


def auto_centre(snap, prefer=("bh", "subhalo", "group", "potmin", "densest"), bh_id=None, group_index=0):
    """Try centring strategies in order and return the first that works.

    Returns ``(centre_code, info)`` where ``info['method']`` records which one
    was used -- always print this, since comparing figures centred different
    ways is a classic way to manufacture a spurious result.

    On failure ``info['reason']`` names *each* method and why it declined,
    rather than a single unhelpful "all methods failed".
    """
    tried = []
    for method in prefer:
        if method in ("subhalo", "subhalo_bh"):
            try:
                c, info = centre_on_subhalo(
                    snap, prefer="bh" if (method == "subhalo_bh" or bh_id is not None) else "central",
                    bh_id=bh_id, group_index=group_index,
                )
            except (FileNotFoundError, OSError) as exc:
                c, info = None, {"reason": str(exc)}
        elif method == "bh":
            c, info = centre_on_bh(snap, bh_id=bh_id)
        elif method == "group":
            try:
                c, info = centre_on_group(snap, group_index=group_index)
            except FileNotFoundError as exc:
                c, info = None, {"reason": str(exc)}
        elif method == "potmin":
            c, info = centre_on_potential_minimum(snap)
        elif method == "densest":
            c, info = centre_on_densest_mass(snap)
        else:
            continue
        if c is not None:
            info["method"] = method
            return c, info
        tried.append("%s: %s" % (method, info.get("reason", "failed")))
    return None, {"method": None, "reason": "; ".join(tried) or "no methods tried"}


# ---------------------------------------------------------------------------
# Gas phases
# ---------------------------------------------------------------------------


def gas_phase_masks(T, nH=None, sfr=None):
    """Split gas into cold / warm / hot (and optionally dense, star-forming).

    Returns a dict of boolean arrays.  Note that with the SMUGGLE/GFM effective
    equation of state, star-forming cells sit on the EoS and their *stored*
    temperature is a mass-weighted fiction of the two-phase subgrid model, not
    a real one -- so ``sf`` is reported separately and is deliberately allowed
    to overlap the temperature bins.
    """
    T = np.asarray(T, dtype=np.float64)
    masks = {
        "cold": T < T_COLD,
        "warm": (T >= T_COLD) & (T < T_WARM),
        "hot": T >= T_WARM,
    }
    if nH is not None:
        masks["dense"] = (np.asarray(nH) > NH_DENSE) & masks["cold"]
    if sfr is not None:
        masks["sf"] = np.asarray(sfr) > 0
    return masks


# ---------------------------------------------------------------------------
# Aperture measurements
# ---------------------------------------------------------------------------


def measure(snap, centre, radii_kpc=(1.0, 2.0, 5.0, 10.0, 30.0), vel_centre=None, group_index=0,
            cat=None, subhalo_index=None, use_inrad=True):
    """Measure a standard set of galaxy properties in spherical apertures.

    Parameters
    ----------
    centre : (3,) array
        Centre in *code* coordinates.
    radii_kpc : sequence
        Physical aperture radii.
    vel_centre : (3,) array or None
        Velocity to subtract before computing kinematics.  If None it is the
        mass-weighted mean velocity of star + gas inside the smallest aperture,
        which is the sensible galaxy rest frame.

    Returns
    -------
    dict
        Scalars keyed ``<quantity>_r<radius>``, e.g. ``mstar_r5``, plus
        aperture-independent entries (``bh_mass_msun``, ``mhalo_msun``, ...).
        Everything is in physical units: Msun, Msun/yr, kpc, km/s, K.
    """
    u = snap.units
    a = snap.a
    box = snap.boxsize
    L = u.length_to_kpc(a)
    M = u.mass_to_msun

    out = {"num": snap.num, "a": a, "z": snap.z, "age_gyr": snap.age_gyr, "centre_code": np.asarray(centre, dtype=np.float64)}

    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Density", "InternalEnergy", "ElectronAbundance", "StarFormationRate", "Velocities", "GFM_Metallicity"])
    star = snap.part(io.PT_STAR, ["Coordinates", "Masses", "Velocities", "GFM_StellarFormationTime", "GFM_Metallicity", "GFM_InitialMass"])
    dm = snap.part(io.PT_DM, ["Coordinates", "Masses"])
    bh = snap.part(io.PT_BH, ["Coordinates", "Masses", "BH_Mass", "BH_Mdot", "BH_Hsml", "ParticleIDs", "BH_Density", "BH_U",
                               "BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"])

    # -- black hole ---------------------------------------------------------
    if bh.get("count", 0) and "Coordinates" in bh:
        rbh = io.radius_from(bh["Coordinates"], centre, box) * L
        near = rbh < max(radii_kpc)
        mbh = bh.get("BH_Mass", bh.get("Masses"))
        out["n_bh_in_aperture"] = int(near.sum())
        out["bh_mass_msun"] = float(np.sum(mbh[near]) * M) if near.any() else 0.0
        out["bh_mass_max_msun"] = float(np.max(mbh) * M)
        if "BH_Mdot" in bh:
            out["bh_mdot_msun_per_yr"] = float(np.sum(bh["BH_Mdot"][near]) * u.mdot_to_msun_per_yr) if near.any() else 0.0
            out["bh_mdot_max_msun_per_yr"] = float(np.max(bh["BH_Mdot"]) * u.mdot_to_msun_per_yr)
        if "BH_Hsml" in bh and near.any():
            out["bh_hsml_kpc"] = float(np.mean(bh["BH_Hsml"][near]) * L)
        if out.get("bh_mass_msun", 0) > 0 and "bh_mdot_msun_per_yr" in out:
            out["eddington_ratio"] = float(u.eddington_ratio(out["bh_mdot_msun_per_yr"], out["bh_mass_msun"]))
        # BH_CumEgyInjection_QM/RM (snap_io.cc: BCEQ/BCER) are the code's own
        # running totals of thermal/kinetic feedback energy actually coupled to
        # the gas by BH_THERMALFEEDBACK -- unlike energy.txt's e_injected
        # (All.EgyInjection, a mesh refinement/derefinement bookkeeping term
        # with no connection to AGN feedback), this is the real quantity.  Not
        # incremented by BH_BIPOLAR_INJECTION*, which keeps its own budget only
        # in the per-step stdout counters (see evolution.plot_energy_budget).
        cum_th = bh.get("BH_CumEgyInjection_QM")
        cum_kin = bh.get("BH_CumEgyInjection_RM")
        if (cum_th is not None or cum_kin is not None) and near.any():
            z = np.zeros(mbh.shape, dtype=np.float64)
            out["bh_cum_egy_injected_code"] = float(np.sum(((cum_th if cum_th is not None else z) +
                                                              (cum_kin if cum_kin is not None else z))[near]))
    else:
        out["n_bh_in_aperture"] = 0
        out["bh_mass_msun"] = 0.0

    # -- galaxy and halo, from the Subfind catalogue ------------------------
    #
    # These are the *galaxy-scale* numbers and they come from Subfind, not from
    # summing particles: Subfind has already decided which particles are bound
    # to this galaxy rather than to a neighbour or to the surrounding halo.
    # They are stored with a ``sub_`` prefix so a plot can never confuse a
    # catalogue stellar mass with an aperture sum -- and ``mhalo_msun`` is the
    # host FoF group's M200c, which is the whole zoom system and is only the
    # right SMHM x-axis when this galaxy is that group's central (recorded as
    # ``sub_is_central``).
    try:
        if cat is None:
            cat = cat_mod.load(snap)
        if cat is not None and cat.n_sub:
            if subhalo_index is None:
                pos = cat.s("pos")
                if pos is not None:
                    r = io.radius_from(np.asarray(pos, dtype=np.float64), centre, box) * L
                    subhalo_index = int(np.argmin(r))
                    out["sub_centre_offset_kpc"] = float(r[subhalo_index])
                else:
                    subhalo_index = cat.target_index()
            props = cat.properties(subhalo_index, use_inrad=use_inrad)
            for k, v in props.items():
                if k in ("a", "z", "aperture"):
                    continue
                out["sub_%s" % k] = v
            # Halo mass and radius are what the Group table is genuinely the
            # authority on, so they keep their unprefixed names.
            for k in ("mhalo_msun", "r200c_kpc", "group_sfr_msun_per_yr"):
                if k in props:
                    out[k] = props[k]
            out["n_subhalos"] = cat.n_sub
            out["n_groups"] = cat.n_grp
    except (FileNotFoundError, OSError, KeyError, IndexError) as exc:
        out["sub_error"] = str(exc)

    # -- velocity frame -----------------------------------------------------
    if vel_centre is None:
        vel_centre = _galaxy_rest_frame(snap, centre, min(radii_kpc))
    out["vel_centre_code"] = vel_centre

    # -- per-aperture -------------------------------------------------------
    T = u.temperature(gas["InternalEnergy"], gas.get("ElectronAbundance")) if gas.get("count", 0) else np.zeros(0)
    nH = gas["Density"] * u.density_to_nH_cgs(a) if gas.get("count", 0) else np.zeros(0)
    phases = gas_phase_masks(T, nH, gas.get("StarFormationRate")) if gas.get("count", 0) else {}

    r_gas = io.radius_from(gas["Coordinates"], centre, box) * L if gas.get("count", 0) else np.zeros(0)
    r_star = io.radius_from(star["Coordinates"], centre, box) * L if star.get("count", 0) else np.zeros(0)
    r_dm = io.radius_from(dm["Coordinates"], centre, box) * L if dm.get("count", 0) else np.zeros(0)

    for R in radii_kpc:
        tag = "r%g" % R
        mg = r_gas <= R
        ms = r_star <= R
        md = r_dm <= R

        out["mgas_%s" % tag] = float(np.sum(gas["Masses"][mg]) * M) if mg.any() else 0.0
        out["mstar_%s" % tag] = float(np.sum(star["Masses"][ms]) * M) if ms.any() else 0.0
        out["mdm_%s" % tag] = float(np.sum(dm["Masses"][md]) * M) if md.any() else 0.0
        out["ngas_%s" % tag] = int(mg.sum())
        out["nstar_%s" % tag] = int(ms.sum())

        for ph in ("cold", "warm", "hot", "dense", "sf"):
            if ph in phases:
                sel = mg & phases[ph]
                out["mgas_%s_%s" % (ph, tag)] = float(np.sum(gas["Masses"][sel]) * M) if sel.any() else 0.0

        if "StarFormationRate" in gas and mg.any():
            out["sfr_%s" % tag] = float(np.sum(gas["StarFormationRate"][mg]))  # AREPO writes Msun/yr

        if mg.any():
            w = gas["Masses"][mg]
            out["nH_mean_%s" % tag] = float(np.sum(nH[mg] * w) / np.sum(w))
            out["T_mean_%s" % tag] = float(np.sum(T[mg] * w) / np.sum(w))
            vol = 4.0 / 3.0 * np.pi * R ** 3  # kpc^3
            out["rho_gas_%s" % tag] = out["mgas_%s" % tag] / vol  # Msun / kpc^3
            if "GFM_Metallicity" in gas:
                out["Zgas_%s" % tag] = float(np.sum(gas["GFM_Metallicity"][mg] * w) / np.sum(w))
        if ms.any() and "GFM_Metallicity" in star:
            w = star["Masses"][ms]
            out["Zstar_%s" % tag] = float(np.sum(star["GFM_Metallicity"][ms] * w) / np.sum(w))

        # Total enclosed mass and circular velocity: the cleanest dynamical
        # summary, and the thing to compare against a rotation curve.
        mtot = out["mgas_%s" % tag] + out["mstar_%s" % tag] + out["mdm_%s" % tag] + out.get("bh_mass_msun", 0.0)
        out["mtot_%s" % tag] = mtot
        out["vcirc_%s" % tag] = _vcirc_kms(mtot, R)

    # -- kinematics and shape inside the fiducial (largest) aperture --------
    Rk = max(radii_kpc)
    if star.get("count", 0):
        ms = r_star <= Rk
        if ms.sum() > 10:
            out["sigma_star_kms"] = _velocity_dispersion(star["Velocities"][ms], star["Masses"][ms], vel_centre, u, a)
            out["rhalf_star_kpc"] = _half_mass_radius(r_star[ms], star["Masses"][ms])
            out["jstar_kpc_kms"], out["jstar_dir"] = _specific_angular_momentum(
                io.periodic_delta(star["Coordinates"][ms], centre, box) * L,
                star["Velocities"][ms],
                star["Masses"][ms],
                vel_centre,
                u,
                a,
            )
    if gas.get("count", 0):
        mg = r_gas <= Rk
        if mg.sum() > 10:
            out["jgas_kpc_kms"], out["jgas_dir"] = _specific_angular_momentum(
                io.periodic_delta(gas["Coordinates"][mg], centre, box) * L,
                gas["Velocities"][mg],
                gas["Masses"][mg],
                vel_centre,
                u,
                a,
            )
            cold = mg & phases["cold"]
            if cold.sum() > 10:
                out["jgas_cold_kpc_kms"], out["jgas_cold_dir"] = _specific_angular_momentum(
                    io.periodic_delta(gas["Coordinates"][cold], centre, box) * L,
                    gas["Velocities"][cold],
                    gas["Masses"][cold],
                    vel_centre,
                    u,
                    a,
                )

    # -- derived ratios -----------------------------------------------------
    Rf = max(radii_kpc)
    tagf = "r%g" % Rf
    ms_, mg_ = out.get("mstar_%s" % tagf, 0.0), out.get("mgas_%s" % tagf, 0.0)
    if ms_ > 0:
        out["fgas"] = mg_ / (mg_ + ms_)
        out["ssfr_per_gyr"] = out.get("sfr_%s" % tagf, 0.0) / ms_ * 1.0e9
        if out.get("bh_mass_msun", 0) > 0:
            out["mbh_over_mstar"] = out["bh_mass_msun"] / ms_
    if out.get("mhalo_msun", 0) > 0 and ms_ > 0:
        out["mstar_over_mhalo"] = ms_ / out["mhalo_msun"]

    return out


def _galaxy_rest_frame(snap, centre, radius_kpc):
    """Mass-weighted mean velocity of stars+gas inside ``radius_kpc``."""
    L = snap.units.length_to_kpc(snap.a)
    box = snap.boxsize
    num, den = np.zeros(3), 0.0
    for pt in (io.PT_STAR, io.PT_GAS):
        d = snap.part(pt, ["Coordinates", "Masses", "Velocities"])
        if d.get("count", 0) == 0 or "Velocities" not in d:
            continue
        r = io.radius_from(d["Coordinates"], centre, box) * L
        m = r <= radius_kpc
        if not m.any():
            continue
        w = d["Masses"][m]
        num += np.sum(d["Velocities"][m] * w[:, None], axis=0)
        den += float(np.sum(w))
    return num / den if den > 0 else np.zeros(3)


def _velocity_dispersion(vel, mass, vel_centre, units, a):
    """Mass-weighted 1D velocity dispersion in km/s (physical peculiar)."""
    f = units.peculiar_velocity_to_kms(a)
    dv = (np.asarray(vel) - np.asarray(vel_centre)) * f
    w = np.asarray(mass, dtype=np.float64)
    var = np.sum(w[:, None] * dv ** 2, axis=0) / np.sum(w)
    return float(np.sqrt(np.sum(var) / 3.0))


def _half_mass_radius(r, mass):
    o = np.argsort(r)
    c = np.cumsum(mass[o])
    return float(np.interp(0.5 * c[-1], c, r[o]))


def _specific_angular_momentum(rel_kpc, vel, mass, vel_centre, units, a):
    """Mass-weighted specific angular momentum of a set of particles.

    Returns ``(magnitude, unit_direction)`` -- the magnitude in kpc km/s and the
    direction as a unit 3-vector.  These are stored under separate keys because
    mixing a scalar and a vector under one name is how the history builder used
    to break: a snapshot with no stars has no j at all, and the missing value is
    a scalar NaN that cannot be stacked with 3-vectors from later snapshots.

    The direction is worth keeping in its own right -- it is the disc axis, and
    its drift over time is the cleanest measure of how much the notional jet
    axis wanders.
    """
    f = units.peculiar_velocity_to_kms(a)
    dv = (np.asarray(vel) - np.asarray(vel_centre)) * f
    w = np.asarray(mass, dtype=np.float64)
    J = np.sum(w[:, None] * np.cross(rel_kpc, dv), axis=0)
    j = J / np.sum(w)
    mag = float(np.linalg.norm(j))
    return mag, (j / mag if mag > 0 else np.zeros(3))


def _vcirc_kms(M_msun, R_kpc):
    """sqrt(GM/R) in km/s for M in Msun and R in kpc."""
    if R_kpc <= 0 or M_msun <= 0:
        return 0.0
    v_cgs = np.sqrt(GRAVITY * M_msun * SOLAR_MASS / (R_kpc * KPC_IN_CM))
    return float(v_cgs / 1.0e5)


# ---------------------------------------------------------------------------
# Radial profiles
# ---------------------------------------------------------------------------


def radial_profile(snap, centre, parttype, quantity="mass", bins=None, phase=None, log=True):
    """Spherically averaged radial profile.

    ``quantity``:
        'mass'     -> enclosed mass (Msun) and shell density (Msun/kpc^3)
        'density'  -> mass-weighted mean cell Density (as nH, cm^-3)
        'temp'     -> mass-weighted mean temperature (K)
        'sfr'      -> summed SFR per shell (Msun/yr) and surface-ish density
        'vrad'     -> mass-weighted radial velocity (km/s), positive = outflow
        'metal'    -> mass-weighted GFM_Metallicity

    Returns ``{'r': centres_kpc, 'value': ..., 'count': ...}`` plus extras
    depending on the quantity.
    """
    u, a, box = snap.units, snap.a, snap.boxsize
    L = u.length_to_kpc(a)
    M = u.mass_to_msun

    fields = ["Coordinates", "Masses"]
    if quantity in ("density", "temp", "sfr", "metal"):
        fields += ["Density", "InternalEnergy", "ElectronAbundance", "StarFormationRate", "GFM_Metallicity"]
    if quantity == "vrad":
        fields += ["Velocities"]
    if phase is not None:
        fields += ["Density", "InternalEnergy", "ElectronAbundance", "StarFormationRate"]

    d = snap.part(parttype, sorted(set(fields)))
    if d.get("count", 0) == 0:
        return {"r": np.zeros(0), "value": np.zeros(0), "count": np.zeros(0)}

    r = io.radius_from(d["Coordinates"], centre, box) * L
    w = d["Masses"] * M

    if phase is not None and "InternalEnergy" in d:
        T = u.temperature(d["InternalEnergy"], d.get("ElectronAbundance"))
        nH = d["Density"] * u.density_to_nH_cgs(a) if "Density" in d else None
        pm = gas_phase_masks(T, nH, d.get("StarFormationRate"))
        sel = pm[phase]
        r, w = r[sel], w[sel]
        d = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape else v) for k, v in d.items()}

    if bins is None:
        rmax = np.percentile(r, 99.5) if r.size else 10.0
        bins = np.logspace(np.log10(max(rmax * 1e-3, 1e-3)), np.log10(max(rmax, 1e-2)), 40) if log else np.linspace(0, rmax, 40)
    bins = np.asarray(bins, dtype=np.float64)
    centres = np.sqrt(bins[1:] * bins[:-1]) if log else 0.5 * (bins[1:] + bins[:-1])

    idx = np.digitize(r, bins) - 1
    ok = (idx >= 0) & (idx < centres.size)
    count = np.bincount(idx[ok], minlength=centres.size).astype(float)
    mshell = np.bincount(idx[ok], weights=w[ok], minlength=centres.size)
    vshell = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    res = {"r": centres, "count": count, "bins": bins, "mass_shell": mshell, "menc": np.cumsum(mshell)}

    def wmean(vals):
        num = np.bincount(idx[ok], weights=(w * vals)[ok], minlength=centres.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(mshell > 0, num / mshell, np.nan)

    if quantity == "mass":
        res["value"] = mshell / vshell  # Msun / kpc^3
        res["label"] = r"$\rho$ [M$_\odot$ kpc$^{-3}$]"
    elif quantity == "density":
        res["value"] = wmean(d["Density"] * u.density_to_nH_cgs(a))
        res["label"] = r"$n_{\rm H}$ [cm$^{-3}$]"
    elif quantity == "temp":
        res["value"] = wmean(u.temperature(d["InternalEnergy"], d.get("ElectronAbundance")))
        res["label"] = r"$T$ [K]"
    elif quantity == "sfr":
        s = np.bincount(idx[ok], weights=d["StarFormationRate"][ok], minlength=centres.size)
        res["value"] = s
        res["label"] = r"SFR per shell [M$_\odot$ yr$^{-1}$]"
    elif quantity == "metal":
        res["value"] = wmean(d["GFM_Metallicity"])
        res["label"] = r"$Z$ (mass fraction)"
    elif quantity == "vrad":
        rel = io.periodic_delta(d["Coordinates"], centre, box) * L
        rr = np.linalg.norm(rel, axis=1)
        vc = _galaxy_rest_frame(snap, centre, max(centres) if centres.size else 10.0)
        dv = (d["Velocities"] - vc) * u.peculiar_velocity_to_kms(a)
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = np.sum(rel * dv, axis=1) / np.where(rr > 0, rr, np.nan)
        res["value"] = wmean(np.nan_to_num(vr))
        res["label"] = r"$v_r$ [km s$^{-1}$]"
    else:
        raise ValueError("unknown quantity %r" % quantity)

    return res


def rotation_curve(snap, centre, rmax_kpc=30.0, nbins=40, log=True):
    """Enclosed-mass circular velocity curve, split by component.

    This is the spherical v_c = sqrt(GM(<r)/r), not a tilted-ring fit -- fine
    for a diagnostic, not comparable to an observed HI rotation curve without
    caveats.
    """
    bins = np.logspace(-1.5, np.log10(rmax_kpc), nbins) if log else np.linspace(0.05, rmax_kpc, nbins)
    curves = {"r": bins}
    total = np.zeros(bins.size)
    for pt, name in ((io.PT_GAS, "gas"), (io.PT_DM, "dm"), (io.PT_STAR, "stars"), (io.PT_BH, "bh")):
        d = snap.part(pt, ["Coordinates", "Masses"])
        if d.get("count", 0) == 0 or "Masses" not in d:
            curves[name] = np.zeros(bins.size)
            continue
        r = io.radius_from(d["Coordinates"], centre, snap.boxsize) * snap.units.length_to_kpc(snap.a)
        m = d["Masses"] * snap.units.mass_to_msun
        o = np.argsort(r)
        menc = np.interp(bins, r[o], np.cumsum(m[o]), left=0.0)
        total += menc
        curves[name] = np.array([_vcirc_kms(mm, rr) for mm, rr in zip(menc, bins)])
    curves["total"] = np.array([_vcirc_kms(mm, rr) for mm, rr in zip(total, bins)])
    curves["menc_total"] = total
    return curves
