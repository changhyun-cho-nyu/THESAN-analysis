"""
thesan_tmz -- analysis package for the AREPO2 THESAN-TMZ-newbh runs.

Written against the ``Arepo2-TMZ-newbh`` branch, whose black hole modules
(BH_BIPOLAR_INJECTION, BH_ALPHA_DISK_ACCRETION, BH_TORQUE_LIMITED_ACCRETION,
LOCAL_INJECTION_BLACK_HOLES) produce outputs no generic AREPO analysis suite
knows how to read.

Typical use::

    from thesan_tmz import arepo_io, history, evolution, scaling, diagnostics
    from thesan_tmz.plotstyle import apply_style

    apply_style()
    out = '/scratch/thesan_tmz/output'
    hist = history.load_history(out)
    logs = evolution.collect_logs(out)
    evolution.plot_overview(hist, hist['_units'], 'figs/overview.png', logs=logs)

or, for everything at once::

    python -m thesan_tmz.run_all /scratch/thesan_tmz --figdir figs

Module map
----------
``units``       code-unit / cosmology conversions, all the powers of a and h
``arepo_io``    HDF5 snapshot and group-catalogue readers
``catalog``     Subfind layer -- every galaxy-scale number comes from here
``tree``        the THESAN-Zoom merger tree, output/tree.hdf5
``bh_logs``     parsers for blackholes.txt, blackhole_details, bipolar logs, stdout;
                the AGN energy book
``bh_population`` every black hole: per-BH table, population and track figures
``lowres``      which BHs sit in low-res gas (zoom-edge seeding artefacts); hidden by default
``local_mesh``  each BH's cell in its local Voronoi mesh, rebuilt from the gas; 3D page
``slr_mesh``    how well SLR_BH refines the mesh around a BH (fig 19)
``galaxy``      centring, apertures, gas phases, kinematics, radial profiles
``history``     walk the snapshots once, cache the result (or the catalogues only)
``obs``         published scaling relations to compare against
``evolution``   time-evolution figures
``scaling``     scaling-relation figures
``diagnostics`` model-specific health checks unique to this branch
``maps``        projections, phase diagrams, jet-frame views, galaxy view
``uv``          intrinsic 1500 A luminosity per star from the run's spectrum tables
``kinematics``  spin directions of stars and gas, v/sigma of the bound stars (fig 22)
``contamination`` low-res (type 2, 3) particles around the target halo (report.txt)
``density``     stellar rho(<r), Sigma(<R) of the bound stars vs compact JWST systems (fig 23)
``metallicity`` gas MZR / DGR, stellar MZR, iron of the galaxy and the clean centrals (figs 05, 24, 25)
``plotstyle``   shared matplotlib style and helpers
``run_all``     driver that produces the whole figure set
"""

__version__ = "1.1.0"

__all__ = [
    "units",
    "arepo_io",
    "catalog",
    "tree",
    "bh_logs",
    "bh_population",
    "lowres",
    "uv",
    "kinematics",
    "contamination",
    "density",
    "metallicity",
    "local_mesh",
    "slr_mesh",
    "galaxy",
    "history",
    "obs",
    "evolution",
    "scaling",
    "diagnostics",
    "maps",
    "plotstyle",
]
