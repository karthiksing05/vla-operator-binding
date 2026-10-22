#!/bin/bash
# B1 (CompoSuite) runner.  SCRIPT = experiments/<x>.py, ARGS = its arguments (space separated; no commas, since
# sbatch --export splits on commas).  Runs the CompoSuite venv with -I (isolated: no cwd / user site on sys.path).
#   sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00 --exclude=... \
#          --export=ALL,SCRIPT=experiments/b1_env_smoke.py,ARGS="--render" slurm/jobs/b1_cpu.sbatch
# add --gres=gpu:a40:1 for GPU work (same script).
#SBATCH --job-name=b1
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/results/b1/logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --open-mode=append
cd /coc/flash10/ksingara3/lifelong-ood-learner
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-${MUJOCO_GL:-egl}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1
PY=/coc/flash10/ksingara3/robot/composuite/.venv/bin/python
echo "[b1] $(hostname) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(date -Is) $SCRIPT $ARGS"
$PY -I $SCRIPT $ARGS
echo "[b1] exit $? $(date -Is)"
