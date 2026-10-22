#!/bin/bash
# Plan D2 / G30f: LIBERO hdf5 for LOTUS's stream run: the base stage's 40 tasks (LIBERO-SPATIAL, -GOAL, -10; -OBJECT already
# there) and the W1 stream's LIBERO-90 tasks (symlinked from robot/data/libero_raw; the 9 absent there downloaded), from the HF
# mirror yifengzhu-hf/LIBERO-datasets.  Idempotent.
#SBATCH --job-name=g30lotdata
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
set -x
V=/coc/flash10/ksingara3/robot/venvs
D=/coc/flash10/ksingara3/robot/data/lotus_datasets
export HOME=$V/lotus_home HF_HOME=$V/lotus_home/hf
mkdir -p $D/libero_90
for f in /coc/flash10/ksingara3/robot/data/libero_raw/libero_90/*.hdf5; do [ -e $D/libero_90/$(basename $f) ] || ln -s $f $D/libero_90/; done
$V/lotus/bin/python -I -c "
from huggingface_hub import snapshot_download
pats = ['libero_spatial/*', 'libero_goal/*', 'libero_10/*'] + ['libero_90/%s_demo.hdf5' % n for n in '$MISSING'.split()]
snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', allow_patterns=pats, local_dir='$D')
" && touch $D/.stream.done
ls $D/*/ | wc -l
echo exit=$?
