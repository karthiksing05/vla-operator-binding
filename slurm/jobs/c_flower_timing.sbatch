#!/bin/bash
# CALVIN-C P1 timing + resume test: 200 steps (checkpoint at 100, 200), a second launch resuming to 300, then the export.
#SBATCH --job-name=cflower-time
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_timing.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_timing.out
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G
set -u
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
RUN=${RUN:-/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/timing_s0}; WORKERS=${WORKERS:-8}
rm -rf $RUN; mkdir -p $RUN
echo "[timing] $(date) $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
$PY experiments/c_train_flower.py train --split_seed 0 --run $RUN --workers $WORKERS --max_steps 200 --ckpt_every 100 || exit 1
T1=$(date +%s)
$PY experiments/c_train_flower.py train --split_seed 0 --run $RUN --workers $WORKERS --max_steps 300 --ckpt_every 100 || exit 1
T2=$(date +%s)
$PY experiments/c_train_flower.py export --run $RUN --tag 300 || exit 1
T3=$(date +%s)
ls -la $RUN/ckpt $RUN/export_step300
echo "[timing] launch1 $((T1-T0))s launch2(resume) $((T2-T1))s export $((T3-T2))s"
