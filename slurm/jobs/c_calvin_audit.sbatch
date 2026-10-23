#!/bin/bash
# CALVIN-C flow-loss contamination audit (experiments/c_calvin_audit.py).  MODELS is space-free: colon-separated specs.
#   sbatch ... --export=ALL,SEED=0,MODELS=pret:abc slurm/jobs/c_calvin_audit.sbatch
#SBATCH --job-name=ccalvin-audit
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_audit.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_audit.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
set -u
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
SEED=${SEED:-0}; MODELS=${MODELS:-pret:abc}; XARGS=${XARGS:-}
echo "[audit] $(date) $(hostname) seed=$SEED models=$MODELS"
$PY experiments/c_calvin_audit.py --split_seed $SEED --models ${MODELS//:/ } --record $XARGS
echo "[audit] exit=$? $(date)"
