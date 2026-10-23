#!/bin/bash
# L2b offline criteria (experiments/pl2b_criteria.py --compute ...), one GPU; resumable per output file.
#   sbatch -p overcap -A overcap --qos=short --requeue --exclude=... --export=ALL,ARGS="--compute --backbone pi0_libero --shard 0/4" experiments/pl2b_criteria.sbatch
#SBATCH --job-name=pl2b
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
set -eo pipefail
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
PY=/coc/flash10/ksingara3/robot/openpi/.venv/bin/python
echo "[pl2b] $ARGS $(date) on $(hostname)"
$PY experiments/pl2b_criteria.py $ARGS 2>&1 | grep -v Warning
echo "exit=0 | $(date)"
