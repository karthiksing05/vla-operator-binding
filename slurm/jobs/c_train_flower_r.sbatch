#!/bin/bash
# (2026-10-22 copy of c_train_flower.sbatch for new submissions: chains with --requeue and sky1 excluded; the original is in use.)
# CALVIN-C P1: FLOWER-C fine-tune (experiments/c_train_flower.py train), resumable, self-chaining: each job trains until its
# wall budget, checkpoints, and (if not finished and it advanced) resubmits itself with CHAIN-1.
#   sbatch ... --export=ALL,RUN=<dir>,SEED=0,FULL=0,CHAIN=6 slurm/jobs/c_train_flower_r.sbatch
#SBATCH --job-name=cflower
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_train.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_train.out
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G
set -u
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1 NCCL_P2P_LEVEL=SYS
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
RUN=${RUN:?}; SEED=${SEED:-0}; FULL=${FULL:-0}; CHAIN=${CHAIN:-0}; WORKERS=${WORKERS:-8}; BUDGET=${BUDGET:-215}
MAXSTEPS=${MAXSTEPS:-0}; CKEVERY=${CKEVERY:-4000}
XA=""; [ "$FULL" = "1" ] && XA="--full"
step() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['global_step'])" $RUN/progress.json 2>/dev/null || echo 0; }
fin() { $PY -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('finished', False)))" $RUN/progress.json 2>/dev/null || echo 0; }
mkdir -p $RUN
B=$(step)
echo "[cflower] $(date) $(hostname) job=$SLURM_JOB_ID run=$RUN seed=$SEED full=$FULL chain=$CHAIN start_step=$B"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY experiments/c_train_flower.py train --split_seed $SEED --run $RUN $XA --workers $WORKERS --budget_min $BUDGET \
    --max_steps $MAXSTEPS --ckpt_every $CKEVERY
rc=$?
A=$(step); F=$(fin)
echo "[cflower] $(date) exit=$rc steps $B -> $A finished=$F"
echo "$SLURM_JOB_ID $B $A $F $rc $(date +%s)" >> $RUN/jobs.txt
if [ "$F" != "1" ] && [ "$A" -gt "$B" ] && [ "$CHAIN" -gt 0 ]; then
  sbatch --partition=overcap --account=overcap --qos=short --time=04:00:00 --gres=gpu:a40:4 \
    --requeue --exclude=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot \
    --export=ALL,RUN=$RUN,SEED=$SEED,FULL=$FULL,CHAIN=$(( CHAIN - 1 )),WORKERS=$WORKERS,BUDGET=$BUDGET,MAXSTEPS=$MAXSTEPS,CKEVERY=$CKEVERY \
    slurm/jobs/c_train_flower_r.sbatch
fi
