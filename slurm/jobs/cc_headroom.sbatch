#!/bin/bash
# CALVIN-C Stage A headroom rollouts (experiments/cc_headroom.py roll): NPROC processes share one GPU, each a part of the
# frozen episode list; resumable (per-episode jsonl, finished episodes are skipped on requeue).
#   sbatch -p overcap -A overcap --requeue ... --export=ALL,MODEL=flower_c_s0,NPROC=6 slurm/jobs/cc_headroom.sbatch
#SBATCH --job-name=cc-headroom
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/cc_headroom_%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/cc_headroom_%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
set -u
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
MODEL=${MODEL:?}; NPROC=${NPROC:-6}
CKPT=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/$MODEL/export_stepfinal
OUT=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/headroom/$MODEL
mkdir -p $OUT
echo "[cch] $(date) $(hostname) model=$MODEL nproc=$NPROC"
T0=$(date +%s)
for (( p=0; p<NPROC; p++ )); do
  $PY experiments/cc_headroom.py roll --ckpt $CKPT --model $MODEL --part $p --nparts $NPROC > $OUT/log_${p}of${NPROC}.txt 2>&1 &
done
wait
echo "[cch] done in $(( $(date +%s) - T0 ))s"
