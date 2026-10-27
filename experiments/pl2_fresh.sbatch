#!/bin/bash
# Plan L / L2 (experiments/pl2_compose.py): two NEW fresh start-state sets on Li's OOD suites, never used before.
#   fresh_ood_pl2d  BASE 3300000, 8 per task   the one-demo pool (recorded demos 0 and 1)
#   fresh_ood_pl2   BASE 3400000, 20 per task  the closed-loop test set
#SBATCH --job-name=pl2-fresh
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --requeue
cd /coc/flash10/ksingara3/robot/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PYTHONPATH=$PYTHONPATH:/coc/flash10/ksingara3/robot/openpi/third_party/libero
for spec in "fresh_ood_pl2d 3300000 8" "fresh_ood_pl2 3400000 20"; do
  set -- $spec
  for suite in libero_goal_ood libero_spatial_ood; do
    FRESH_DIR=$1 BASE=$2 N=$3 SUITE=$suite TASKS=0+1+2+3+4+5+6+7+8+9 \
      examples/libero/.venv/bin/python /coc/flash10/ksingara3/vla-operator-binding/experiments/fresh_states.py 2>&1 | grep -v "WARNING\|overload"
  done
done
echo "[pl2-fresh] done $(date)"
