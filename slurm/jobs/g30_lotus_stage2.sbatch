#!/bin/bash
# Plan D2 / G30f: one resumable LOTUS run (arXiv 2311.02058, research use only) in its own venv, on its published LIBERO setting.
#   RUN=<name>  LIST=<file of "<suite>/<task>_demo" lines, stream order>  NBASE=<number of base tasks>
#   [NEWD=10 OLDD=5 BASED=50]  [EPOCHS=50]  [INIT=<dir of init-state files>]  [STOP=<last stage>]
# Steps, each skipped once its marker exists (a requeued job resumes): DINOv2 features of every task (GPU), hierarchical
# agglomeration + spectral clustering with continual skill addition (LOTUS's agglomoration_script: base stage, then one new
# task per step), then per stage N = NBASE, NBASE+1, ...: skill policies + meta-controller (lifelong.main, multitask_skill,
# BUDS subgoals), warm-started from stage N-1, eval at the end of the stage on every seen task (20 episodes, LOTUS's harness).
# Base stage: BASED demonstrations per base task; later stages: NEWD for the new task and OLDD replayed per earlier task (the
# paper: 50 / 10 / 5).  Skills and the meta-controller resume per skill (markers in the stage directory).
#SBATCH --job-name=g30lot
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
set -x
V=/coc/flash10/ksingara3/robot/venvs
W=/coc/flash10/ksingara3/robot/work/lotus/lotus
export HOME=$V/lotus_home LIBERO_CONFIG_PATH=$V/lotus_home/.libero HF_HOME=$V/lotus_home/hf XDG_CACHE_HOME=$V/lotus_home/.cache
export TORCH_HOME=$V/lotus_home/torch MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# DINOv2 initialises torch.distributed from the Slurm environment, which a batch step without srun lacks: one task, rank 0
export SLURM_NTASKS=${SLURM_NTASKS:-1} SLURM_PROCID=${SLURM_PROCID:-0} SLURM_LOCALID=${SLURM_LOCALID:-0} SLURM_NODEID=${SLURM_NODEID:-0}
P=$V/lotus/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
export LOTUS_DATASET_LIST=$LIST LOTUS_N_BASE=$NBASE
EXP=dinov2_${RUN}
M=$W/skill_learning/results/$EXP.markers; mkdir -p $M
cd $W/skill_learning
if [ ! -f $M/repr.done ]; then
  $P multisensory_repr/dinov2_repr.py --exp-name $EXP --modality-str dinov2_agentview_eye_in_hand --feature-dim 1536 || exit 11
  touch $M/repr.done
fi
if [ ! -f $M/hier.done ]; then
  $P skill_discovery/hierarchical_agglomoration.py exp_name=$EXP modality_str=dinov2_agentview_eye_in_hand repr.z_dim=1536 \
     agglomoration.dist=cos agglomoration.footprint=global_pooling || exit 12
  touch $M/hier.done
fi
if [ ! -f $M/spectral.done ]; then
  $P skill_discovery/agglomoration_script.py exp_name=$EXP modality_str=dinov2_agentview_eye_in_hand repr.z_dim=1536 \
     agglomoration.segment_scale=1 agglomoration.min_len_thresh=30 agglomoration.K=2 agglomoration.scale=0.01 agglomoration.dist=cos || exit 13
  touch $M/spectral.done
fi
cd $W
NT=$(grep -c . $LIST)
PREV=""
for N in $(seq $NBASE ${STOP:-$NT}); do
  SD=$W/experiments_g30/$RUN/stage$N
  if [ ! -f $SD/result.pt ]; then
    mkdir -p $SD
    if [ $N = $NBASE ]; then export LOTUS_NEW_DEMOS=0 LOTUS_OLD_DEMOS=${BASED:-50}; else export LOTUS_NEW_DEMOS=${NEWD:-10} LOTUS_OLD_DEMOS=${OLDD:-5}; fi
    # the stage's task list for the benchmark: the first N lines of LIST (g30 benchmark registered by the wrapper module)
    export LOTUS_BENCH_LIST=$LIST LOTUS_BENCH_N=$N LOTUS_EXP_DIR=$SD
    $P lifelong/main.py seed=1 benchmark_name=g30_stream policy=bc_transformer_policy lifelong=multitask_skill \
       skill_learning.exp_name=${EXP}_$N exp=$RUN goal_modality=BUDS use_wandb=false lifelong.eval_in_train=False eval.eval=True \
       train.n_epochs=${EPOCHS:-50} ${INIT:+init_states_folder=$INIT} ${PREV:+pretrain_model_path=$PREV} || exit 20
  fi
  PREV=$SD
done
echo "exit=0"
