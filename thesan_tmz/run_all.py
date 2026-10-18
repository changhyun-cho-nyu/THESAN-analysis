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
from . import bh_logs, catalog, diagnostics, evolution, galaxy, history, maps, scaling, tree as tree_mod
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
    ap.add_argument("--no-maps", action="store_true", help="skip the (slow) projection figures")
    ap.add_argument("--catalog-only", action="store_true",
                    help="build the history from the group catalogues alone -- seconds instead of hours, "
                         "at the cost of the particle-based apertures, gas phases and kinematics")
    ap.add_argument("--total-mass", action="store_true",
                    help="use SubhaloMassType (total bound) instead of SubhaloMassInRadType for galaxy masses")
    ap.add_argument("--title", default=None, help="title prefix for every figure")
    args = ap.parse_args(argv)

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
    made += _try("bh growth", evolution.plot_bh_growth, logs, units, os.path.join(figdir, "02_bh_growth.png"), title=title)
    made += _try("energy", evolution.plot_energy_budget, logs, units, os.path.join(figdir, "04_energy.png"), hist=hist, params=params, title=title)

    # -- 4. scaling relations ----------------------------------------------
    print("\n--- scaling relations ---")
    pop = None
    if last_snap is not None:
        pop = _try_value(scaling.population_from_catalog, last_snap,
                         use_inrad=not args.total_mass, min_highres=0.9)
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
        logs["details"],
        units,
        os.path.join(figdir, "08_accretion_internals.png"),
        params=params,
        summary=summary,
        title=title,
    )
    reports.append(diagnostics.check_mode_switch(logs["details"], inj, units, params, summary=summary))
    made += _try("conservation", diagnostics.plot_conservation, logs, units, os.path.join(figdir, "09_conservation.png"), params=params, hist=hist, title=title)

    # -- 6. maps and the collimation test -----------------------------------
    if last_snap is not None and not args.no_maps:
        print("\n--- maps and collimation ---")
        centre, cinfo = galaxy.auto_centre(last_snap, prefer=centre_prefer, bh_id=args.bh_id)
        if centre is None:
            print("  [run_all] no centre in the last snapshot (%s) -- skipping maps" % cinfo.get("reason"))
        else:
            print("  centred by '%s' %s" % (cinfo.get("method"), {k: v for k, v in cinfo.items() if k != "method"}))
            axis = _axis_for_snapshot(last_snap, centre, logs, bh_id=cinfo.get("bh_id"))
            made += _try("phase diagram", maps.plot_phase_diagram, last_snap, centre, os.path.join(figdir, "10_phase_diagram.png"), radius_kpc=max(radii), title=title)
            made += _try("radial profiles", maps.plot_radial_profiles, last_snap, centre, os.path.join(figdir, "11_radial_profiles.png"), title=title)
            made += _try(
                "projections",
                maps.plot_projection_grid,
                last_snap,
                centre,
                os.path.join(figdir, "12_projections.png"),
                extent_kpc=args.extent,
                axis=axis,
                title=title,
            )
            made += _try(
                "collimation",
                diagnostics.plot_collimation,
                last_snap,
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
                last_snap,
                centre,
                os.path.join(figdir, "14_stellar_ages.png"),
                radius_kpc=max(radii),
                title=title,
            )

    # -- 7. what applies to this build, and what is genuinely missing -------
    made = [m for m in made if m]
    skipped = set()
    if args.no_maps:
        skipped |= {"10", "11", "12", "13", "14"}
    if args.catalog_only:
        skipped |= {"01", "03"}
    rows = diagnostics.figure_applicability(
        summary, logs, made,
        has_snapshots=bool(snaps), has_history=hist is not None, has_catalog=chist is not None,
        skipped=skipped,
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


def _try_value(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:
        print("  [run_all] %s: %s" % (getattr(fn, "__name__", fn), exc))
        return None


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
