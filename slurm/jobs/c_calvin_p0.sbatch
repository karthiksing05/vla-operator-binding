#!/bin/bash
# CALVIN-C P0: freeze the three split seeds (+ ledger c_calvin_split), run the split tests, and build each seed's
# training-window filter from ABC's annotations (+ ledger c_calvin_windows).  CPU only.
#SBATCH --job-name=ccalvin-p0
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_p0.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_p0.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -u
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/coc/flash10/ksingara3/miniconda3/bin/python
echo "[p0] $(date) $(hostname)"
$PY experiments/c_calvin_split.py split --record || exit 1
$PY -m pytest -q tests/test_calvin_c_split.py || exit 1
for k in 0 1 2; do $PY experiments/c_calvin_split.py windows --seed $k --record || exit 1; done
echo "[p0] done $(date)"
