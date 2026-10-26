#!/bin/bash
#SBATCH --job-name=tmz_analysis
#SBATCH --account=rrg-kannanr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --output=tmz_analysis.%j.out
#
# Run thesan_tmz.run_all on every run directory (one containing output/) under the
# test roots, several runs at a time on one node.
#
#   bash run_all_tests.sh --list            # only print the runs that would be analysed
#   sbatch run_all_tests.sh                 # analyse all of them
#   NPAR=4 EXTRA="--no-maps" sbatch run_all_tests.sh
#   EXTRA="--catalog-only" bash run_all_tests.sh   # quick look (seconds per run; fine on a login node)
#
# Each run's progress goes to <run>/figures/analysis_log.txt, figures to <run>/figures/.
# The run's stdout files (myjob.*.out) are found by thesan_tmz itself.

set -u

ROOTS=(/scratch/chcho/test_BH_ACC_wo_LI /scratch/chcho/test_LI_BH)
PKG_DIR=${PKG_DIR:-/scratch/chcho/BH_analysis}  # the directory that contains thesan_tmz/
NPAR=${NPAR:-6}                                 # how many runs to analyse at the same time
EXTRA=${EXTRA:-}                                # extra run_all options, e.g. "--no-maps"

# --- Python environment: edit to match your setup (needs numpy h5py matplotlib scipy plotly) ---
# module load python scipy-stack
# source ~/venvs/tmz/bin/activate

runs=()
for root in "${ROOTS[@]}"; do
  for d in "$root"/*/; do
    d=${d%/}
    [ -d "$d/output" ] && runs+=("$d")
  done
done

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${runs[@]}"
  echo "${#runs[@]} runs"
  exit 0
fi

if [ ! -d "$PKG_DIR/thesan_tmz" ]; then
  echo "thesan_tmz not found in $PKG_DIR (set PKG_DIR)" >&2
  exit 1
fi

analyse() {
  local run=$1
  local tag
  tag="$(basename "$(dirname "$run")")/$(basename "$run")"
  mkdir -p "$run/figures"
  echo "[$(date +%T)] start $tag"
  (cd "$PKG_DIR" && python3 -m thesan_tmz.run_all "$run" --figdir "$run/figures" --title "$tag" $EXTRA) \
    > "$run/figures/analysis_log.txt" 2>&1
  local status=$?
  echo "[$(date +%T)] done  $tag (exit $status) -> $run/figures/analysis_log.txt"
}
export -f analyse
export PKG_DIR EXTRA

echo "analysing ${#runs[@]} runs, $NPAR at a time"
printf '%s\n' "${runs[@]}" | xargs -P "$NPAR" -I{} bash -c 'analyse "$1"' _ {}
echo "all done"
