#!/bin/bash
# Plan H4: one GPU script in a NEW backbone's venv (PY = its interpreter), e.g.
#   sbatch --export=ALL,PY=/coc/flash10/ksingara3/robot/groot/.venv/bin/python,SCRIPT="experiments/h4_backbone_check.py groot" \
#       --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw slurm/jobs/h4_gpu.sbatch
#SBATCH --job-name=lolh4
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname)"; if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi; exit 97
fi
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1 WANDB_MODE=disabled
echo "[h4] $(date) on $(hostname): PY=$PY SCRIPT=$SCRIPT"
srun $PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
