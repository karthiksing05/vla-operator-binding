#!/bin/bash
# X3 EgoDex GPU runner (GR00T venv).  ARGS = the python arguments (script first); exit code 3 = "time limit reached, state
# saved" -> the job requeues itself (overcap preemption also requeues, via --requeue), so a long run spans many 4-h jobs.
#   sbatch --export=ALL,ARGS="experiments/x3_egodex_train.py --steps 30000 --out results/robot/egodex/train/run1" slurm/jobs/x3_egodex_gpu.sbatch
#SBATCH --job-name=x3gpu
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/egodex/logs/gpu_%x_%j.log
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU on $(hostname); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot HF_HUB_OFFLINE=1
PY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python
echo "[x3] $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0}: $ARGS"
$PY -I $ARGS
rc=$?
echo "[x3] exit=$rc | $(date)"
if [ $rc -eq 3 ]; then scontrol requeue $SLURM_JOB_ID; fi
exit $rc
