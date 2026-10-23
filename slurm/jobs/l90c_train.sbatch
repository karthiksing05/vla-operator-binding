#!/bin/bash
# Multi-GPU runner for P1 (pi05_l90c, experiments/c_train_pi05.py): one JAX process sees all the job's GPUs (FSDP).
#   sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00 --gres=gpu:a40:4 --exclude=... \
#          --export=ALL,SCRIPT="experiments/c_train_pi05.py --exp main --steps 30000 --batch 32" slurm/jobs/l90c_train.sbatch
# Resumable: resubmit the same command (chain with --dependency=afterany:<job>).  sbatch splits --export on COMMAS.
#SBATCH --job-name=l90c-train
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
NG=$(nvidia-smi -L 2>/dev/null | grep -c "GPU ")
if [ "$NG" -lt 1 ]; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[l90c-train] $SCRIPT"
echo "[l90c-train] $(date) on $(hostname): $NG x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (restart ${SLURM_RESTART_COUNT:-0})"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
