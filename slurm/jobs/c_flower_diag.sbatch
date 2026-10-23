#!/bin/bash
# CALVIN-C P1 throughput diagnosis: s/step for (devices, per-GPU batch) settings, 40 steps each, NCCL transport logged.
#SBATCH --job-name=cflower-diag
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_diag.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/logs/%j_diag.out
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G
set -u
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
nvidia-smi topo -m
for cfgs in "1 8" "4 8" "2 16"; do set -- $cfgs
  RUN=/coc/flash10/ksingara3/lifelong-ood-learner/results/calvin_c/diag_d$1_b$2; rm -rf $RUN
  echo "[diag] devices=$1 batch=$2 $(date)"
  $PY experiments/c_train_flower.py train --split_seed 0 --run $RUN --devices $1 --batch $2 --workers 8 --max_steps 60 --ckpt_every 100000 2>&1 \
     | grep -E "c_train|NCCL INFO (Channel|Connected|comm|Using)|P2P|SHM|Error|Traceback" | grep -v "^\s*$" | head -60
  rm -rf $RUN/ckpt
done
