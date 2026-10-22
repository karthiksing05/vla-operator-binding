#!/bin/bash
# X4 wrapper parity: Pi0MetaWorldBackbone vs RLinf's own policy object (experiments/x4_metaworld_parity.py).
#   sbatch --export=ALL,RECORD=1 slurm/jobs/x4_metaworld_parity.sbatch
#SBATCH --job-name=x4mwpar
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache
export OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
ROOT=/coc/flash10/ksingara3/robot/metaworld; PY=$ROOT/.venv/bin/python; RL=$ROOT/ext/RLinf
REPO=/coc/flash10/ksingara3/lifelong-ood-learner
CKPT=${CKPT:-RLinf/RLinf-Pi0-MetaWorld-SFT}; CONFIG=${CONFIG:-pi0_metaworld}; RECORD=${RECORD:-0}; COMPILED=${COMPILED:-1}
SNAP=$($PY -I -c "from huggingface_hub import snapshot_download; print(snapshot_download('$CKPT'))")
XA=""; [ "$RECORD" = "1" ] && XA="--record"
echo "[x4par] $(date) $(hostname) $SNAP"
cd $RL && PYTHONPATH=$RL:$REPO $PY $REPO/experiments/x4_metaworld_parity.py --rlinf_root $RL --pretrained_path $SNAP \
  --config_name $CONFIG --compiled $COMPILED --out $REPO/results/x4_metaworld/parity_${CONFIG}.json $XA
echo "[x4par] done $(date)"
