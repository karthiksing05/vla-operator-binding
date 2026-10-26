#!/bin/bash
# PLAN_BIND: one LIBERO-env CPU command (BINDCMD).  New file.
#SBATCH --job-name=bindcpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
REPO=/coc/flash10/ksingara3/vla-operator-binding
ROBOT=/coc/flash10/ksingara3/robot/openpi
cd $REPO
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID|RESTART_COUNT|CPUS_PER_TASK)$'); do unset $v; done
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "[bind cpu] $(date) on $(hostname): $BINDCMD"
PYTHONPATH=$ROBOT/third_party/libero $ROBOT/examples/libero/.venv/bin/python $BINDCMD
rc=$?
echo "exit=$rc | $(date)"
exit $rc
