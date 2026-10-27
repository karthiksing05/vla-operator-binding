#!/bin/bash
# X1 parity: FLOWER (mbreuss/flower_calvin_abc) on CALVIN ABC->D, the authors' flower_evaluate.py main() via
# experiments/x1_calvin_flower_eval.py.  NPROC processes share one GPU, each a contiguous slice of chains [START, END).
#   sbatch --export=ALL,START=0,END=50,NPROC=5 slurm/jobs/x1_calvin_flower.sbatch
#SBATCH --job-name=x1flower
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
REPO=/coc/flash10/ksingara3/vla-operator-binding
FLOWER=/coc/flash10/ksingara3/robot/ext/flower_vla_calvin
CKPT=/coc/flash10/ksingara3/robot/data/hf/home/hub/models--mbreuss--flower_calvin_abc/snapshots/6424ee4c064eb6654a39280314793620b149dd97
VAL=/coc/flash10/ksingara3/robot/calvin/task_ABC_D/validation
START=${START:-0}; END=${END:-50}; NPROC=${NPROC:-5}; EPLEN=${EPLEN:-360}; FIX=${FIX:-0}
XA=""; [ "$FIX" = "1" ] && XA="--fix_weights"
OUT=${OUT:-$REPO/results/x1_calvin/flower_${START}_${END}_ep${EPLEN}_fix${FIX}_${SLURM_JOB_ID}}
mkdir -p $OUT
echo "[x1flower] $(date) $(hostname) range=[$START,$END) nproc=$NPROC ep_len=$EPLEN fix=$FIX out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
N=$(( END - START )); STEP=$(( (N + NPROC - 1) / NPROC ))
for (( s=START; s<END; s+=STEP )); do
  e=$(( s + STEP )); (( e > END )) && e=$END
  $PY $REPO/experiments/x1_calvin_flower_eval.py --flower_root $FLOWER --ckpt_dir $CKPT --val_dir $VAL \
     --out $OUT/shard_${s}_${e} --start $s --end $e --ep_len $EPLEN $XA > $OUT/log_${s}_${e}.txt 2>&1 &
done
wait
echo "[x1flower] done in $(( $(date +%s) - T0 ))s"
$PY $REPO/experiments/x1_calvin_merge.py --dir $OUT
