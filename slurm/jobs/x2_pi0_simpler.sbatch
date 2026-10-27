#!/bin/bash
# X2 parity: open-pi-zero (allenzren) pi0 Bridge checkpoints on SimplerEnv (allenzren fork, proprio), the authors' eval
# (scripts/run.py --config-name=bridge, as in their slurm/eval_simpler_bridge.sh).
#   sbatch --export=ALL,CKPT=bridge_beta,TASK=widowx_carrot_on_plate,NEP=240,BF16=False slurm/jobs/x2_pi0_simpler.sbatch
#SBATCH --job-name=x2pi0
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
PY=/coc/flash10/ksingara3/robot/simpler/.venv_pi0/bin/python
OPZ=/coc/flash10/ksingara3/robot/ext/openpizero_src/open-pi-zero
REPO=/coc/flash10/ksingara3/vla-operator-binding
CKPT=${CKPT:-bridge_beta}; TASK=${TASK:-widowx_carrot_on_plate}; NEP=${NEP:-240}; BF16=${BF16:-False}; SEED=${SEED:-42}
case $CKPT in
  bridge_beta) CK=/coc/flash10/ksingara3/robot/data/openpizero/bridge_beta_step19296_2024-12-26_22-30_42.pt;;
  bridge_uniform) CK=/coc/flash10/ksingara3/robot/data/openpizero/bridge_uniform_step19296_2024-12-26_22-31_42.pt;;
  *) echo "unknown CKPT $CKPT"; exit 2;;
esac
export TRANSFORMERS_CACHE=/coc/flash10/ksingara3/robot/data/paligemma_tok    # holds paligemma-3b-pt-224/ (tokenizer only)
export VLA_LOG_DIR=$REPO/results/x2_simpler/pi0_${CKPT}_bf16${BF16}_${SLURM_JOB_ID}
mkdir -p $VLA_LOG_DIR; cd $VLA_LOG_DIR
echo "[x2pi0] $(date) $(hostname) ckpt=$CKPT task=$TASK nep=$NEP bf16=$BF16 seed=$SEED"
T0=$(date +%s)
$PY $OPZ/scripts/run.py --config-name=bridge --config-path=$OPZ/config/eval \
  device=cuda:0 seed=$SEED n_eval_episode=$NEP n_video=0 env.task=$TASK horizon_steps=4 act_steps=4 \
  use_bf16=$BF16 use_torch_compile=True name=$CKPT \
  env.adapter.dataset_statistics_path=$OPZ/config/bridge_statistics.json \
  "checkpoint_path=\"$CK\""
echo "[x2pi0] done in $(( $(date +%s) - T0 ))s"
