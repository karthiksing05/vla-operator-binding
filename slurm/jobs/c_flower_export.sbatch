#!/bin/bash
# CALVIN-C P1: export a finished FLOWER-C run's last.ckpt (EMA weights + <Flow> row) to <run>/export_step<TAG>.
#   sbatch ... --export=ALL,RUN=<run dir>,TAG=final slurm/jobs/c_flower_export.sbatch
#SBATCH --job-name=cflower-export
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_export.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_export.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -u
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
RUN=${RUN:?}; TAG=${TAG:-final}
echo "[export] $(date) $(hostname) run=$RUN tag=$TAG"
$PY experiments/c_train_flower.py export --run $RUN --tag $TAG || exit 1
ls -la $RUN/export_step$TAG
echo "[export] done $(date)"
