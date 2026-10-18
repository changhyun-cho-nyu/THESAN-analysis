# THESAN-analysis

**One command that turns an AREPO run into a folder of diagnostic figures**, with a
focus on the galaxy, its black holes and their feedback.

`thesan_tmz` is written for runs of the AREPO2 branch `Arepo2-TMZ-newbh` (THESAN-Zoom
physics with the new black-hole modules: Bondi / α-disk / torque-limited accretion,
collimated thermal and kinetic feedback, local Voronoi injection). Generic AREPO
analysis tools do not read these outputs; this package does. It also works on runs
without those modules: anything a run did not write is skipped, with a note saying why.

```bash
python3 -m thesan_tmz.run_all /path/to/my_run
```

…and look at `/path/to/my_run/figures/`.

---

## Contents

- [Install](#install)
- [Check that it works](#check-that-it-works-2-minutes)
- [Analyse a run](#analyse-a-run)
- [What you get](#what-you-get)
- [Useful options](#useful-options)
- [What it reads](#what-it-reads)
- [Using it from Python](#using-it-from-python)
- [Troubleshooting](#troubleshooting)
- [How it works, briefly](#how-it-works-briefly)
- [License](#license)

---

## Install

There is nothing to compile. Clone the repository and install the Python packages:

```bash
git clone https://github.com/changhyun-cho-nyu/THESAN-analysis.git
```

```bash
cd THESAN-analysis
```

```bash
pip install -r requirements.txt
```

| package | needed for |
|---|---|
| `numpy`, `h5py`, `matplotlib` | everything (required) |
| `scipy` | smoother projection maps; the 3D local-mesh page |
| `plotly` | the 3D local-mesh page (`18_bh_local_mesh.html`) |

Python 3.8 or newer. On a cluster, `module load python scipy-stack` usually provides
the first four.

Run the commands below from the repository folder (the one that contains
`thesan_tmz/`), or add that folder to your `PYTHONPATH`.

## Check that it works (2 minutes)

```bash
python3 -m thesan_tmz.selftest
```

This builds a small synthetic run in a temporary folder (snapshots, group catalogues,
every log file, six different feedback configurations), runs the whole pipeline on it
and checks the numbers against known answers. The last line should read
`N passed, 0 failed`. Add `--keep` to keep the synthetic run and its figures, to see
what the output looks like before you use a real run.

## Analyse a run

```bash
python3 -m thesan_tmz.run_all /path/to/my_run
```

`/path/to/my_run` can be the run folder (the one that contains `output/`) or the
`output/` folder itself. Figures go to `/path/to/my_run/figures/` (change it with
`--figdir`).

What happens:

1. It reads the configuration from the last snapshot and works out which feedback
   channels the run has.
2. It reads the log files (`blackholes.txt`, `blackhole_details/`, `sfr.txt`, …).
3. It measures the target galaxy in every snapshot (stellar mass, gas phases, black
   hole, …) and **caches** the result in `output/analysis_history.npz`. This is the slow
   part the first time; afterwards only new or changed snapshots are measured again.
4. It makes the figures and writes `report.txt`.

A run that is still going is fine: re-run the same command later and only the new
snapshots are processed.

**For a quick first look** at a large run, skip the slow parts:

```bash
python3 -m thesan_tmz.run_all /path/to/my_run --catalog-only --no-maps
```

This uses only the group catalogues (seconds instead of hours) and skips the
projection maps.

## What you get

| file | shows |
|---|---|
| `01_overview.png` | the galaxy over time: BH mass, stellar mass, gas phases, SFR, accretion rate, Eddington ratio, … |
| `02_bh_growth.png` | total BH mass and accretion rate, BH count and mergers, the most massive BHs |
| `03_star_formation.png` | star formation history and stellar growth |
| `04_energy.png` | energy budget of the box; AGN energy released and deposited, per channel, against the expected value |
| `05_scaling_*.png` | the galaxy against published relations (M★–M_halo, M_BH–M★, main sequence, MZR, M_BH–σ, …) |
| `06_injection_counters.png` | per-step injected energy of the collimated feedback (needs `--stdout`) |
| `07_axis_stability.png` | how much the jet axis wanders (runs with `BH_BIPOLAR_FEEDBACK`) |
| `08_accretion_internals.png` | accretion-model internals: raw vs capped rate, Eddington ratio, BH environment |
| `09_conservation.png` | BH mass and AGN energy bookkeeping |
| `10_phase_diagram.png` | density–temperature diagram |
| `11_radial_profiles.png` | density, temperature, radial velocity, circular velocity |
| `12_projections.png` | gas maps face-on and edge-on in the jet frame |
| `13_collimation.png` | outflow speed, temperature and mass vs angle from the jet axis: **the test of collimated feedback** |
| `14_stellar_ages.png` | star formation history from stellar ages |
| `15_catalog_evolution.png` | the galaxy as Subfind sees it, with the merger tree |
| `16_bh_population.png` | **every** BH in the last snapshot: mass vs host, Eddington ratio, offsets |
| `17_bh_tracks.png` | the history of every BH from `blackhole_details` |
| `18_bh_local_mesh.html` | interactive 3D view of each BH's cell in its local Voronoi mesh (open in a browser) |
| `bh_table.txt` | one row per BH in the last snapshot |
| `report.txt` | the run's configuration, checks, and **which figures apply** |

**Not every run produces every figure, and that is expected.** The end of
`report.txt` (also printed on screen) labels each figure as *produced*, *not
applicable* to this build (with the reason), *skipped* on request, or **MISSING**.
Only MISSING means something went wrong.

## Useful options

| option | meaning |
|---|---|
| `--figdir DIR` | where to write the figures |
| `--snapshots 0-40` or `0,10,20` | only these snapshots |
| `--catalog-only` | use the group catalogues only (fast; no apertures, gas phases or kinematics) |
| `--no-maps` | skip the projection figures (10–14), which are slow |
| `--stdout run.out` | the run's stdout file(s), comma-separated for restarts; enables fig 06 and the energy ledger |
| `--bh-id ID` | follow this BH instead of the default one |
| `--radii 1,2,5,10,30` | aperture radii in physical kpc |
| `--extent 20` | half-size of the maps, kpc |
| `--mesh-bhs 10` | 3D page for this many most massive BHs (`0` skips it) |
| `--total-mass` | use the total bound mass of each subhalo instead of the mass within twice the half-mass radius |
| `--rebuild` | ignore the cache and measure every snapshot again |
| `--title "Run A"` | text added to every figure title |

`python3 -m thesan_tmz.run_all --help` lists them all.

## What it reads

Everything in the run's `output/` folder that exists; nothing is required except
snapshots or catalogues.

| file(s) | used for |
|---|---|
| `snapdir_NNN/snap_NNN.*.hdf5` or `snap_NNN.hdf5` | galaxy measurements, maps, every BH |
| `groups_NNN/fof_subhalo_tab_NNN.*.hdf5` | galaxy masses, sizes, centres (Subfind) |
| `tree.hdf5` | the main branch of the merger tree |
| `blackholes.txt`, `blackhole_details/`, `blackhole_mergers/` | BH growth, accretion, mergers |
| `sfr.txt`, `energy.txt` | star formation and energy budget |
| the run's stdout (`--stdout`) | feedback counters and the local-injection energy ledger |

Units, cosmology and the feedback configuration are read from the snapshot headers,
so there is nothing to set by hand.

## Using it from Python

Every figure is also a function, and the measurements are plain arrays:

```python
from thesan_tmz import arepo_io, history, evolution, bh_population
from thesan_tmz.plotstyle import apply_style

apply_style()
out = "/path/to/my_run/output"

hist = history.load_history(out)          # cached measurements, one row per snapshot
print(hist["z"], hist["mstar_r30"], hist["bh_mass_msun"])

snap = arepo_io.Snapshot(out, 42)          # one snapshot
gas = snap.part(0, ["Coordinates", "Masses", "Density"])
table = bh_population.bh_table(snap)       # every BH in it
```

`thesan_tmz/__init__.py` lists the modules and what each one does.

## Troubleshooting

| symptom | what to do |
|---|---|
| `No module named 'numpy'` (or `h5py`, …) | activate the environment you installed into, or `module load python scipy-stack` |
| `No module named 'thesan_tmz'` | run from the repository folder, or add it to `PYTHONPATH` |
| a figure is listed as **MISSING** in `report.txt` | the reason column says which file it needed; check that file exists in `output/` |
| fig 06 is missing | pass the run's stdout with `--stdout` |
| fig 18 is not applicable | `pip install scipy plotly` |
| the first run takes hours | normal for many large snapshots; use `--catalog-only` for a quick look, or `--snapshots` for a subset. Later runs reuse the cache |
| numbers look stale after you changed the code | the cache notices code changes itself; `--rebuild` forces a full rebuild |
| a run on a cluster login node is killed | run it as a batch job |

Set `THESAN_TMZ_TRACEBACK=1` to see full Python tracebacks instead of one-line
failure messages.

## How it works, briefly

- **Two kinds of numbers, never mixed.** Galaxy-wide quantities (stellar mass, SFR,
  size, halo mass) come from the Subfind catalogue. Spatially resolved quantities
  (apertures, gas phases, profiles, maps) come from the snapshot particles.
- **The target galaxy** is the central galaxy of the most massive halo (FoF group 0),
  which in a zoom is the halo the run is about. Its black hole is the one the figures
  follow (or the one you choose with `--bh-id`); `16_bh_population.png` and
  `bh_table.txt` cover all the others.
- **Code units are converted once**, in `units.py`, including all the powers of the
  scale factor *a* and *h*. The Eddington rate uses the run's own radiative efficiency
  and is checked against the simulation's own `BH_MdotEddington`.
- **Every figure is independent**: if one fails, the others are still made and the
  failure is reported.

---

## License

MIT: free to use, change and share, including in your own projects, as long as the
copyright notice is kept. See [`LICENSE`](LICENSE).

*Questions or problems: open an issue on this repository.*
