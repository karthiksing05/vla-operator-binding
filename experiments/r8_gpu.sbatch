#!/bin/bash
# PROBE R8 (docs/PLAN_R8.md): one openpi-env GPU command (R8CMD, e.g. "experiments/r8_serve.py --build-protos").  New file.
#SBATCH --job-name=r8gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
REPO=/coc/flash10/ksingara3/lifelong-ood-learner
ROBOT=/coc/flash10/ksingara3/robot/openpi
cd $REPO
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID|RESTART_COUNT|CPUS_PER_TASK)$'); do unset $v; done
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export BACKBONE_CONFIG=pi05_l90c BACKBONE_CKPT=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000
echo "[r8 gpu] $(date) on $(hostname): $R8CMD"
$ROBOT/.venv/bin/python $R8CMD
rc=$?
echo "exit=$rc | $(date)"
exit $rc
