#!/bin/bash
# PLAN_SOLVE S1: one GPU command (S1CMD) in the openpi venv (or S1PY).  New file.
#SBATCH --job-name=s1gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
REPO=/coc/flash10/ksingara3/vla-operator-binding
ROBOT=/coc/flash10/ksingara3/robot/openpi
cd $REPO
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID|RESTART_COUNT|CPUS_PER_TASK)$'); do unset $v; done
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export BACKBONE_CONFIG=pi05_l90c BACKBONE_CKPT=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000
PY=${S1PY:-$ROBOT/.venv/bin/python}
echo "[s1 gpu] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); $PY $S1CMD"
$PY $S1CMD
rc=$?
echo "exit=$rc | $(date)"
exit $rc
