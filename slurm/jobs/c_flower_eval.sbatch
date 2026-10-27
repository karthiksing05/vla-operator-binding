#!/bin/bash
# CALVIN-C P1 parity: a FLOWER-C export (config.yaml + model.safetensors) on the standard 1,000 ABC->D chains with the authors'
# evaluation (experiments/x1_calvin_flower_eval.py, the load fix on: the export writes the training-time <Flow> row).
# NPROC processes share one GPU, each a contiguous slice [START, END).
#   sbatch ... --export=ALL,CKPT=<export dir>,OUT=<dir>,START=0,END=500,NPROC=5 slurm/jobs/c_flower_eval.sbatch
#SBATCH --job-name=cflower-eval
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_eval.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_eval.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -u
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
REPO=/coc/flash10/ksingara3/vla-operator-binding
FLOWER=/coc/flash10/ksingara3/robot/ext/flower_vla_calvin
VAL=/coc/flash10/ksingara3/robot/calvin/task_ABC_D/validation
CKPT=${CKPT:?}; OUT=${OUT:?}; START=${START:-0}; END=${END:-1000}; NPROC=${NPROC:-5}; EPLEN=${EPLEN:-360}
mkdir -p $OUT
echo "[ceval] $(date) $(hostname) ckpt=$CKPT range=[$START,$END) nproc=$NPROC out=$OUT"
T0=$(date +%s)
N=$(( END - START )); STEP=$(( (N + NPROC - 1) / NPROC ))
for (( s=START; s<END; s+=STEP )); do
  e=$(( s + STEP )); (( e > END )) && e=$END
  if [ -f $OUT/shard_${s}_${e}/results_${s}_${e}.json ]; then echo "[ceval] shard $s-$e done"; continue; fi
  $PY $REPO/experiments/x1_calvin_flower_eval.py --flower_root $FLOWER --ckpt_dir $CKPT --val_dir $VAL \
     --out $OUT/shard_${s}_${e} --start $s --end $e --ep_len $EPLEN --fix_weights > $OUT/log_${s}_${e}.txt 2>&1 &
done
wait
echo "[ceval] done in $(( $(date +%s) - T0 ))s"
