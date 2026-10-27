#!/bin/bash
# CPU runner in LIBERO's own interpreter (the rollout client's venv), for the failure map's offline typing (experiments/fm_types.py:
# a LIBERO env without a renderer evaluates the BDDL goal atoms at each episode's recorded final object poses).
#   sbatch --wait -p tail-lab -A tail-lab --qos=short --exclude=... --export=ALL,SCRIPT="experiments/fm_types.py --axes" slurm/jobs/fm_cpu.sbatch
# sbatch splits --export on COMMAS: pass lists with + or colons.
#SBATCH --job-name=fmcpu
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ROBOT=/coc/flash10/ksingara3/robot/openpi
export PYTHONPATH=${PYTHONPATH:-}:$ROBOT/third_party/libero
PY=$ROBOT/examples/libero/.venv/bin/python
if [ -z "$SCRIPT" ]; then echo "SCRIPT unset -- nothing to run"; exit 2; fi
echo "[fmcpu] $SCRIPT | $(date) on $(hostname)"
$PY $SCRIPT
rc=$?
echo "exit=$rc | $(date)"
exit $rc
