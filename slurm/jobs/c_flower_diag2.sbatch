#!/bin/bash
# CALVIN-C P1 throughput diagnosis 2: s/step (log every 10 steps) per setting; one line per setting summarised.
#SBATCH --job-name=cflower-diag2
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_diag2.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_diag2.out
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G
set -u
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
D=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c
nvidia-smi topo -m | head -6
run() { # name devices batch find_unused [env]
  R=$D/diag2_$1; rm -rf $R
  echo "[diag2] $1 devices=$2 batch=$3 find_unused=$4 env=${5:-} $(date)"
  env ${5:-} $PY experiments/c_train_flower.py train --split_seed 0 --run $R --devices $2 --batch $3 --find_unused $4 \
     --workers 8 --max_steps 40 --log_every 10 --ckpt_every 100000 > $R.log 2>&1
  tr '\r' '\n' < $R.log | grep -a -E "^\[c_train\]|Error|Traceback" | cut -c1-200
  rm -rf $R/ckpt
}
run d1b8 1 8 1
run d4b8_nofu 4 8 0
run d4b8_p2psys 4 8 1 NCCL_P2P_LEVEL=SYS
run d2b16 2 16 1
echo "[diag2] done $(date)"
