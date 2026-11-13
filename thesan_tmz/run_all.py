"""
Driver: produce the whole figure set for one run.

    python -m thesan_tmz.run_all /path/to/run --figdir figures

``/path/to/run`` may be either the run directory (containing ``output/``) or
the output directory itself.  Everything that can be produced from what is
present is produced; anything that needs a missing file is skipped with a
message rather than aborting, so a run that is still going still gives you
figures.

Exit code is 0 if at least one figure was made, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

from . import arepo_io as io
from . import bh_logs, bh_population, catalog, contamination, density, diagnostics, metallicity, evolution, galaxy, history, kinematics, local_mesh, lowres, maps, scaling, slr_mesh, tree as tree_mod, uv
from .plotstyle import apply_style


def resolve_dirs(path):
    """Accept either a run directory or an output directory."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(os.path.join(path, "output")):
        return path, os.path.join(path, "output")
    return os.path.dirname(path.rstrip("/")), path


def _try(label, fn, *args, **kw):
    """Run one figure-producing step, reporting rather than propagating errors."""
    try:
        r = fn(*args, **kw)
        return [r] if r else []
    except Exception as exc:
        print("  [run_all] %s FAILED: %s: %s" % (label, type(exc).__name__, exc))
        if os.environ.get("THESAN_TMZ_TRACEBACK"):
            traceback.print_exc()
        return []


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyse an AREPO2 THESAN-TMZ-newbh run.")
    ap.add_argument("run", help="run directory (containing output/) or the output directory")
    ap.add_argument("--figdir", default=None, help="where to write figures (default: <run>/figures)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached snapshot history")
    ap.add_argument("--snapshots", default=None, help="subset of snapshot numbers, e.g. '0-40' or '0,10,20'")
    ap.add_argument("--radii", default="1,2,5,10,30", help="aperture radii in physical kpc")
    ap.add_argument("--bh-id", type=int, default=None, help="centre on this BH ID instead of the most massive")
    ap.add_argument("--centre", default="bh,subhalo,group,potmin,densest", help="centring strategies to try, in order")
    ap.add_argument("--stdout", default=None, help="stdout log(s) for the injection counters; comma-separated for a restart chain")
    ap.add_argument("--extent", type=float, default=20.0, help="half-size of the projection maps, kpc")
    ap.add_argument("--galaxy-extent", type=float, default=None,
                    help="half-size of figs 20-21 (galaxy view, gas + UV), kpc; default 3 stellar half-mass radii")
    ap.add_argument("--uv-recipe", choices=uv.RECIPES, default="point",
                    help="stellar UV in figs 20, 21: 'point' = L_nu at exactly 1500 A (default); "
                         "'arepo' = 1490-1510 A average, AREPO's MRT_FOF_1500 (what Subhalo_Luminosity1500 sums)")
    ap.add_argument("--spectra-dir", default=None,
                    help="directory holding the spectrum tables (file names from SpectrumTablePath / "
                         "PopIIISpectrumTablePath, or the two THESAN-Zoom defaults); default %s when it exists"
                         % uv.DEFAULT_SPECTRA_DIR)
    ap.add_argument("--map-snap", type=int, default=None,
                    help="make the maps, kinematics, density and metallicity (figs 10-14, 20-25) from this snapshot instead "
                         "of the last one; the file names stay the same, so use --figdir to keep both")
    ap.add_argument("--no-maps", action="store_true", help="skip the (slow) projection figures")
    ap.add_argument("--mesh-bhs", type=int, default=10,
                    help="3D page of the local Voronoi cell of this many most massive BHs (0 = skip; needs scipy and plotly)")
    ap.add_argument("--catalog-only", action="store_true",
                    help="build the history from the group catalogues alone -- seconds instead of hours, "
                         "at the cost of the particle-based apertures, gas phases and kinematics")
    ap.add_argument("--total-mass", action="store_true",
                    help="use SubhaloMassType (total bound) instead of SubhaloMassInRadType for galaxy masses")
    ap.add_argument("--title", default=None, help="title prefix for every figure")
    ap.add_argument("--show-lowres-bhs", action="store_true",
                    help="keep BHs that sit in low-res gas (zoom-edge seeding artefacts, see lowres.py) in the "
                         "per-BH figures; by default they are hidden from figs 02(d), 08, 16-19 and bh_table.txt")
    args = ap.parse_args(argv)
    if args.spectra_dir is None and os.path.isdir(uv.DEFAULT_SPECTRA_DIR):
        args.spectra_dir = uv.DEFAULT_SPECTRA_DIR
        print("  [run_all] spectrum tables from %s (default; --spectra-dir to change)" % args.spectra_dir)

    run_dir, out_dir = resolve_dirs(args.run)
    figdir = args.figdir or os.path.join(run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    apply_style()

    title = args.title or os.path.basename(run_dir.rstrip("/"))
    radii = tuple(float(x) for x in args.radii.split(","))
    centre_prefer = tuple(x.strip() for x in args.centre.split(","))
    nums = _parse_nums(args.snapshots)

    print("=" * 78)
    print("run directory   : %s" % run_dir)
    print("output directory: %s" % out_dir)
    print("figure directory: %s" % figdir)
    print("=" * 78)

    made, reports = [], []

    # -- 0. what was this run configured to do? -----------------------------
    snaps = io.available_snapshots(out_dir)
    summary, last_snap, units, params = None, None, None, {}
    if snaps:
        try:
            last_snap = io.Snapshot(out_dir, snaps[-1])
            units = last_snap.units
            summary = diagnostics.model_summary(last_snap)
            params = summary["params"]
            print("\n" + diagnostics.format_model_summary(summary) + "\n")
            reports.append("MODEL CONFIGURATION\n" + diagnostics.format_model_summary(summary))
        except Exception as exc:
            print("  [run_all] could not read the last snapshot: %s" % exc)
    map_snap = last_snap
    if args.map_snap is not None and snaps:
        if args.map_snap not in snaps:
            print("  [run_all] --map-snap %d: no such snapshot with particle data (have %s) -- maps skipped"
                  % (args.map_snap, _ranges(snaps)))
            map_snap = None
        else:
            try:
                map_snap = io.Snapshot(out_dir, args.map_snap)
                print("  [run_all] maps and kinematics from snapshot %d (z = %.2f), not the last (%d)"
                      % (args.map_snap, map_snap.z, snaps[-1]))
            except Exception as exc:
                print("  [run_all] could not read snapshot %d: %s -- maps skipped" % (args.map_snap, exc))
                map_snap = None
    else:
        print("  [run_all] no snapshots found -- log-only analysis")

    # -- 1. logs ------------------------------------------------------------
    print("\n--- reading logs ---")
    stdout_arg = [x for x in args.stdout.split(",") if x] if args.stdout else None
    logs = evolution.collect_logs(out_dir, stdout_path=stdout_arg, run_dir=run_dir)
    if logs["stdout_path"]:
        for _p in (logs["stdout_path"] if isinstance(logs["stdout_path"], (list, tuple)) else [logs["stdout_path"]]):
            print("  stdout: %s" % _p)
    if units is None:
        # No snapshots: fall back to the parameter file, then to THESAN defaults.
        from .units import THESAN_UNITS, Units

        pf = _find_parameterfile(run_dir, out_dir)
        units = Units.from_parameterfile(pf) if pf else THESAN_UNITS
        print("  units from %s" % (pf or "THESAN defaults"))
    print("  %s" % units)

    # -- 1b. low-res BHs: zoom-edge seeding artefacts, hidden by default ----
    #
    # Identified in the last snapshot (lowres.py).  Per-BH figures and the
    # details log drop them; box totals and the energy/mass books keep
    # everything, because those must close over the whole box.
    hide_lowres = not args.show_lowres_bhs
    hidden_ids, all_bh_lowres = set(), False
    if last_snap is not None:
        try:
            lr = lowres.classify(last_snap)
            print("  low-res BHs: %s" % lr["note"])
            hidden_ids = lowres.lowres_ids(last_snap) if hide_lowres else set()
            all_bh_lowres = hide_lowres and bool(lr["lowres"].size) and bool(lr["lowres"].all())
            if lr["testable"] and lr["lowres"].any():
                reports.append(lowres.report_lines(last_snap, units, hidden=hide_lowres))
        except Exception as exc:
            print("  [run_all] low-res BH check failed (nothing hidden): %s" % exc)
    logs_shown = dict(logs)
    if hidden_ids:
        logs_shown["details"] = bh_logs.drop_bhs(logs["details"], hidden_ids)
        logs_shown["mergers"] = bh_logs.drop_bhs(logs["mergers"], hidden_ids, ("id_survivor", "id_swallowed"))
        print("  hiding %d low-res BH(s) from the per-BH figures: %s"
              % (len(hidden_ids), " ".join(map(str, sorted(hidden_ids)))))

    # -- 2a. catalogue history (cheap) and merger tree ----------------------
    #
    # Galaxy-scale numbers come from Subfind; the snapshot particles are read
    # only for what Subfind does not measure.  Doing the catalogue first means
    # a run whose snapshots are too big to walk still gets its growth history.
    print("\n--- Subfind catalogue history ---")
    chist, mtree = None, None
    try:
        chist = history.build_catalog_history(out_dir, nums=nums, use_inrad=not args.total_mass, units=units)
        units = chist.get("_units") or units
    except Exception as exc:
        print("  [run_all] catalogue history unavailable: %s" % exc)

    mtree = tree_mod.Tree.open(out_dir, units=units)
    if mtree is not None:
        print("  " + mtree.summary().replace("\n", "\n  "))
    else:
        print("  no output/tree.hdf5 -- main-branch overlay skipped")

    if chist is not None:
        made += _try("catalogue evolution", evolution.plot_catalog_evolution, chist, units,
                     os.path.join(figdir, "15_catalog_evolution.png"), tree=mtree, title=title)

    # -- 2b. snapshot history (expensive) -----------------------------------
    hist = None
    if snaps and args.catalog_only:
        print("\n--- snapshot history skipped (--catalog-only) ---")
    elif snaps:
        print("\n--- building the snapshot history ---")
        try:
            hist = history.load_history(
                out_dir, rebuild=args.rebuild, radii_kpc=radii, centre_prefer=centre_prefer, bh_id=args.bh_id, nums=nums
            )
            if hist.get("_units") is not None:
                units = hist["_units"]
        except Exception as exc:
            print("  [run_all] history failed: %s" % exc)

    # -- 3. evolution figures ----------------------------------------------
    print("\n--- evolution ---")
    if hist is not None:
        made += _try("overview", evolution.plot_overview, hist, units, os.path.join(figdir, "01_overview.png"), title=title, logs=logs)
        made += _try("sfh", evolution.plot_sfh, logs, hist, units, os.path.join(figdir, "03_star_formation.png"), title=title)
    made += _try("bh growth", evolution.plot_bh_growth, logs_shown, units, os.path.join(figdir, "02_bh_growth.png"), title=title,
                 hidden=len(hidden_ids))
    made += _try("energy", evolution.plot_energy_budget, logs, units, os.path.join(figdir, "04_energy.png"), hist=hist, params=params,
                 title=title, summary=summary)

    # -- 4. scaling relations ----------------------------------------------
    print("\n--- scaling relations ---")
    pop = None
    if last_snap is not None:
        pop = _try_value(scaling.population_from_catalog, last_snap,
                         use_inrad=not args.total_mass, min_highres=0.9)
        if pop is not None:
            pop["_metal"] = _try_value(metallicity.sample, last_snap)     # clean centrals for the MZR panels
    # The scaling relations need exactly what Subfind measures, so they work
    # from the catalogue history when there is no particle history.
    shist = hist if hist is not None else chist
    if shist is not None:
        if hist is None:
            print("  using the catalogue history (Subfind apertures, not spherical ones)")
        made += _try_value(scaling.plot_all, shist, units, figdir, pop=pop, prefix="05_scaling", title=title) or []

    # -- 5. model diagnostics ----------------------------------------------
    print("\n--- diagnostics ---")
    inj = logs["injection"]
    made += _try(
        "injection counters",
        diagnostics.plot_injection_counters,
        inj,
        units,
        os.path.join(figdir, "06_injection_counters.png"),
        params=params,
        title=title,
    )
    reports.append(diagnostics.injection_report(inj, logs, units))

    made += _try("axis stability", diagnostics.plot_axis_stability, logs["bipolar"], units, os.path.join(figdir, "07_axis_stability.png"), title=title)
    made += _try(
        "accretion internals",
        diagnostics.plot_accretion_internals,
        logs_shown["details"],
        units,
        os.path.join(figdir, "08_accretion_internals.png"),
        params=params,
        summary=summary,
        title=title,
    )
    reports.append(diagnostics.check_mode_switch(logs["details"], inj, units, params, summary=summary))
    made += _try("conservation", diagnostics.plot_conservation, logs, units, os.path.join(figdir, "09_conservation.png"), params=params,
                 hist=hist, title=title, summary=summary)
    reports.append(diagnostics.ledger_report(logs.get("ledger"), units))

    # -- 5b. every black hole, not just the target galaxy's ------------------
    print("\n--- all black holes ---")
    if last_snap is not None:
        bht = _try_value(bh_population.bh_table, last_snap, hide_lowres=hide_lowres)
        if bht is not None:
            _try_value(bh_population.write_bh_table, os.path.join(figdir, "bh_table.txt"), bht, last_snap)
            made += _try("bh population", bh_population.plot_population, bht, os.path.join(figdir, "16_bh_population.png"),
                         snap=last_snap, title=title)
    made += _try("bh tracks", bh_population.plot_tracks, logs_shown["details"], logs_shown["mergers"], units,
                 os.path.join(figdir, "17_bh_tracks.png"), params=params, title=title, hidden=len(hidden_ids),
                 summary=summary)
    if last_snap is not None and args.mesh_bhs > 0:
        made += _try("bh local mesh", local_mesh.local_mesh_page, last_snap, os.path.join(figdir, "18_bh_local_mesh.html"),
                     max_bhs=args.mesh_bhs, hide_lowres=hide_lowres)
    if last_snap is not None and slr_mesh.slr_settings(last_snap) is not None:
        made += _try("slr mesh", slr_mesh.slr_mesh_figure, last_snap, os.path.join(figdir, "19_slr_mesh.png"),
                     bh_id=args.bh_id, title=title, hide_lowres=hide_lowres)

    # -- 6. maps and the collimation test -----------------------------------
    if map_snap is not None and not args.no_maps:
        print("\n--- maps and collimation ---")
        centre, cinfo = galaxy.auto_centre(map_snap, prefer=centre_prefer, bh_id=args.bh_id)
        if centre is None:
            print("  [run_all] no centre in the last snapshot (%s) -- skipping maps" % cinfo.get("reason"))
        else:
            print("  centred by '%s' %s" % (cinfo.get("method"), {k: v for k, v in cinfo.items() if k != "method"}))
            axis = _axis_for_snapshot(map_snap, centre, logs, bh_id=cinfo.get("bh_id"))
            made += _try("phase diagram", maps.plot_phase_diagram, map_snap, centre, os.path.join(figdir, "10_phase_diagram.png"), radius_kpc=max(radii), title=title)
            made += _try("radial profiles", maps.plot_radial_profiles, map_snap, centre, os.path.join(figdir, "11_radial_profiles.png"), title=title)
            made += _try(
                "projections",
                maps.plot_projection_grid,
                map_snap,
                centre,
                os.path.join(figdir, "12_projections.png"),
                extent_kpc=args.extent,
                axis=axis,
                title=title,
            )
            made += _try("galaxy view", maps.plot_galaxy_view, map_snap, centre, os.path.join(figdir, "20_galaxy_view.png"),
                         extent_kpc=args.galaxy_extent, fallback_kpc=args.extent, title=title,
                         spectra_dir=args.spectra_dir, uv_recipe=args.uv_recipe)
            made += _try("gas + UV", maps.plot_gas_uv, map_snap, centre, os.path.join(figdir, "21_gas_uv.png"),
                         extent_kpc=args.galaxy_extent, fallback_kpc=args.extent, title=title,
                         spectra_dir=args.spectra_dir, uv_recipe=args.uv_recipe)
            try:
                print("  " + uv.catalogue_check(map_snap, args.spectra_dir))
            except Exception as exc:  # a diagnostic line only
                print("  UV check failed: %s" % exc)
            made += _try(
                "collimation",
                diagnostics.plot_collimation,
                map_snap,
                centre,
                axis,
                os.path.join(figdir, "13_collimation.png"),
                rmax_kpc=args.extent,
                params=params,
                title=title,
            )
            made += _try(
                "stellar ages",
                evolution.plot_stellar_age_distribution,
                map_snap,
                centre,
                os.path.join(figdir, "14_stellar_ages.png"),
                radius_kpc=max(radii),
                title=title,
            )

    # -- 6b. kinematics (22), density profile (23), metallicity (24, 25) ----
    unavailable = {}                        # figure id -> why this run cannot have it
    if map_snap is None:
        unavailable.update(dict.fromkeys(("22", "23", "24", "25"), "needs a readable snapshot"))
    else:
        print("\n--- kinematics ---")
        kcen, kinfo = galaxy.auto_centre(map_snap, prefer=centre_prefer, bh_id=args.bh_id)
        if kcen is None:
            unavailable.update(dict.fromkeys(("22", "23"), "no galaxy centre (%s)" % kinfo.get("reason")))
        else:
            # the same bound stars and centre for both
            for fid, label, mod, png in (("22", "kinematics", kinematics, "22_kinematics.png"),
                                         ("23", "density profile", density, "23_density_profile.png")):
                tab, note = _analysed(mod.analyse, map_snap, kcen)
                if mod is kinematics:
                    print("  " + note)
                reports.append(mod.report_lines(tab, note))
                if tab is None:
                    unavailable[fid] = note
                else:
                    made += _try(label, mod.plot, tab, os.path.join(figdir, png), title=title, z=map_snap.z)

        print("\n--- metallicity ---")
        mpop = _try_value(metallicity.sample, map_snap)
        fe = _try_value(metallicity.iron, map_snap, mpop) if mpop is not None else None
        reports.append(metallicity.report_lines(mpop, fe))
        unavailable.update(metallicity.unavailable(mpop, fe))
        if "24" not in unavailable:
            made += _try("stellar MZR", metallicity.plot_stellar_mzr, mpop, os.path.join(figdir, "24_stellar_mzr.png"),
                         title=title, z=map_snap.z)
        if "25" not in unavailable:
            made += _try("iron", metallicity.plot_iron, mpop, fe, os.path.join(figdir, "25_iron.png"),
                         title=title, z=map_snap.z)
        try:
            reports.append(uv.dust_report(map_snap, args.spectra_dir, args.uv_recipe))
        except Exception as exc:  # a report line only
            reports.append("DUST-ATTENUATED M_UV\n  failed: %s: %s" % (type(exc).__name__, exc))

    # -- 6c. low-res contamination around the target (report only) --------
    if map_snap is not None:
        print("\n--- contamination ---")
        try:
            block = contamination.analyse(map_snap)
        except Exception as exc:
            block = "CONTAMINATION\n  failed: %s: %s" % (type(exc).__name__, exc)
        reports.append(block)               # write_report echoes it to stdout

    # -- 7. what applies to this build, and what is genuinely missing -------
    made = [m for m in made if m]
    skipped = set()
    if args.no_maps:
        skipped |= {"10", "11", "12", "13", "14", "20", "21"}
    if args.map_snap is not None and map_snap is None:
        skipped |= {"10", "11", "12", "13", "14", "20", "21", "22", "23", "24", "25"}
    if args.catalog_only:
        skipped |= {"01", "03"}
    if args.mesh_bhs <= 0:
        skipped |= {"18"}
    no_uv = _no_uv_reason(map_snap if snaps else None, args.spectra_dir)
    if no_uv:
        unavailable["21"] = "no stellar UV: %s" % no_uv
    rows = diagnostics.figure_applicability(
        summary, logs, made,
        has_snapshots=bool(snaps), has_history=hist is not None, has_catalog=chist is not None,
        skipped=skipped, all_bh_lowres=all_bh_lowres,
        unavailable=unavailable,
    )
    # write_report echoes to stdout, so appending is enough -- printing here as
    # well would show the block twice.
    reports.append(diagnostics.format_applicability(rows, summary))

    # -- 8. text report -----------------------------------------------------
    diagnostics.write_report(os.path.join(figdir, "report.txt"), reports)

    print("\n" + "=" * 78)
    print("%d figure(s) written to %s" % (len(made), figdir))
    for m in sorted(set(made)):
        print("  %s" % os.path.basename(m))
    print("=" * 78)
    return 0 if made else 1


def _analysed(fn, *args):
    """``fn(*args)`` returning (table, note); a failure becomes (None, "failed: ...")."""
    try:
        return fn(*args)
    except Exception as exc:
        return None, "failed: %s: %s" % (type(exc).__name__, exc)


def _no_uv_reason(snap, spectra_dir):
    """Why fig 21 cannot exist for this run (no spectrum tables), or None."""
    if snap is None:
        return None
    p2, p3, note = uv.table_paths(snap, spectra_dir)
    return None if (p2 or p3) else note


def _try_value(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:
        print("  [run_all] %s: %s" % (getattr(fn, "__name__", fn), exc))
        return None


def _ranges(nums):
    """[0,1,2,3,5,10,15] -> '0-3, 5, 10, 15' (for messages)."""
    nums = sorted(int(n) for n in nums)
    out, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            out.append(str(start) if start == n else "%d-%d" % (start, n))
            start = None
    return ", ".join(out)


def _parse_nums(spec):
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _find_parameterfile(run_dir, out_dir):
    import glob

    for pat in ("*.param-usedvalues", "param.txt-usedvalues", "*.txt-usedvalues", "*.param", "param*.txt"):
        for d in (out_dir, run_dir):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    return None


def _axis_for_snapshot(snap, centre, logs, bh_id=None):
    """Jet axis for this snapshot: prefer the logged one, else recompute.

    The logged axis is what the code actually used; recomputing from the
    snapshot reproduces the same definition but from the post-injection state,
    so the two can differ slightly.  Preferring the log keeps the figure honest.

    ``blackhole_bipolar`` is read one file per MPI rank and concatenated
    without being split by black hole, so with more than one feedback BH the
    array interleaves rows from all of them.  Picking the globally
    nearest-in-time row (as this used to do) can therefore silently return a
    *different* black hole's axis than the one the figure is centred on --
    restrict to ``bh_id`` first.
    """
    if bh_id is None:
        # Centring wasn't done by BH id (e.g. 'group'/'potmin' was used) --
        # fall back to whichever black hole sits nearest the chosen centre, so
        # the log lookup below can still be restricted to a single BH.
        d = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs"])
        if d.get("count", 0) and "ParticleIDs" in d:
            r = io.radius_from(d["Coordinates"], centre, snap.boxsize)
            bh_id = int(d["ParticleIDs"][int(np.argmin(r))])

    bip = logs.get("bipolar", {})
    if bip.get("time", np.zeros(0)).size:
        if bh_id is not None and bip.get("id", np.zeros(0)).size:
            bip = bh_logs.select_bh(bip, bh_id)
        else:
            print("  [run_all] no BH id available -- jet axis lookup cannot be restricted to")
            print("            one black hole; using the nearest-time row across ALL logged")
            print("            black holes (wrong with more than one feedback BH).")
        if bip.get("time", np.zeros(0)).size:
            t = snap.time
            k = int(np.argmin(np.abs(bip["time"] - t)))
            n = bip["jhat"][k]
            if np.linalg.norm(n) > 0 and abs(bip["time"][k] - t) < 0.05 * max(abs(t), 1e-3):
                print(
                    "  jet axis from blackhole_bipolar log (BH %s) at t=%g: [%.3f %.3f %.3f]"
                    % (bh_id, bip["time"][k], n[0], n[1], n[2])
                )
                return n
    axis, rad = diagnostics.bh_axis_from_snapshot(snap, centre)
    if axis is not None:
        print("  jet axis recomputed from the kernel gas (r < %.3g kpc): [%.3f %.3f %.3f]" % (rad, axis[0], axis[1], axis[2]))
    else:
        print("  no jet axis could be determined -- maps will use the simulation frame")
    return axis


if __name__ == "__main__":
    sys.exit(main())
