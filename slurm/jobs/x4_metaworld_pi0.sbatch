#!/bin/bash
# X4 parity: RLinf pi0 / pi0.5 Meta-World SFT under the authors' standalone loop (experiments/x4_metaworld_pi0_eval.py).
# NPROC processes share one GPU, each a slice of the task list; per-task JSONs make it resumable (rerun = continue).
#   sbatch --export=ALL,PART=0,NPART=2,NPROC=4 slurm/jobs/x4_metaworld_pi0.sbatch     # tasks [PART::NPART] of the 50
#   CKPT=RLinf/RLinf-Pi05-MetaWorld-SFT CONFIG=pi05_metaworld for pi0.5; MAXSTEPS=100 for the RL-config horizon.
#SBATCH --job-name=x4mwpi0
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
ROOT=/coc/flash10/ksingara3/robot/metaworld
PY=$ROOT/.venv/bin/python
RL=$ROOT/ext/RLinf
REPO=/coc/flash10/ksingara3/vla-operator-binding
CKPT=${CKPT:-RLinf/RLinf-Pi0-MetaWorld-SFT}; CONFIG=${CONFIG:-pi0_metaworld}; TAG=${TAG:-pi0_sft}
MAXSTEPS=${MAXSTEPS:-160}; PART=${PART:-0}; NPART=${NPART:-1}; NPROC=${NPROC:-4}; TRIALS=${TRIALS:-10}
SNAP=$($PY -I -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CKPT'))")
OUT=${OUT:-$REPO/results/x4_metaworld/${TAG}_ms${MAXSTEPS}}
mkdir -p $OUT
echo "[x4pi0] $(date) $(hostname) ckpt=$SNAP config=$CONFIG part=$PART/$NPART nproc=$NPROC max_steps=$MAXSTEPS out=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -I $REPO/experiments/x4_metaworld_probe.py --out $OUT/probe_$SLURM_JOB_ID || { echo "[x4pi0] probe failed"; exit 1; }
T0=$(date +%s)
cd $RL
TASKS=$(PYTHONPATH=$RL $PY -c "
import json; d=json.load(open('rlinf/envs/sim/metaworld/metaworld_config.json'))
ts=list(d['TASK_DESCRIPTIONS'])[$PART::$NPART]; print(' '.join(ts))")
read -r -a TA <<< "$TASKS"
for (( p=0; p<NPROC; p++ )); do
  SUB=$(for (( i=p; i<${#TA[@]}; i+=NPROC )); do printf "%s:" "${TA[$i]}"; done); SUB=${SUB%:}
  [ -z "$SUB" ] && continue
  PYTHONPATH=$RL $PY $REPO/experiments/x4_metaworld_pi0_eval.py --rlinf_root $RL --config_name $CONFIG \
     --pretrained_path $SNAP --tasks $SUB --num_trials_per_task $TRIALS --max_steps $MAXSTEPS --out $OUT \
     > $OUT/log_${PART}_${p}_${SLURM_JOB_ID}.txt 2>&1 &
done
wait
echo "[x4pi0] done in $(( $(date +%s) - T0 ))s"
cd $REPO && $PY experiments/x4_metaworld_merge.py --dir $OUT --tag $TAG
