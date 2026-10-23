#!/bin/bash
# CPU runner for LIBERO-90-C (P0/P1 data work: downloads, conversion, norm stats).  ENV picks the interpreter:
#   ENV=openpi (default) -> /coc/flash10/ksingara3/robot/openpi/.venv ; ENV=chunks -> the repo's numpy env
#   sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00 --exclude=... \
#          --export=ALL,SCRIPT="experiments/c_data.py --fetch-base" slurm/jobs/l90c_cpu.sbatch
# sbatch splits --export on COMMAS: pass lists with colons.
#SBATCH --job-name=l90c-cpu
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
case "${ENV:-openpi}" in
  openpi) PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python ;;
  chunks) PY=$HOME/flash/miniconda3/envs/chunks/bin/python ;;
  *) echo "unknown ENV=$ENV"; exit 2 ;;
esac
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[l90c-cpu] ENV=${ENV:-openpi} $SCRIPT"
echo "[l90c-cpu] $(date) on $(hostname) (restart ${SLURM_RESTART_COUNT:-0})"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
