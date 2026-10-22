#!/bin/bash
# Plan D2 stage-1 parity smoke for RLinf / pi_RL: evaluate the released checkpoints in RLinf's own LIBERO harness (no training).
#   MODEL=RLinf/RLinf-Pi05-LIBERO-SFT CFG=libero_spatial_openpi_pi05_eval NENV=50 EPOCHS=2
# HOME redirected (the installer and LIBERO write config / caches under it); the user's ~/.libero stays untouched.
#SBATCH --job-name=g30rlpar
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
V=/coc/flash10/ksingara3/robot/venvs
export HOME=$V/rlinf_home LIBERO_CONFIG_PATH=$V/rlinf_home/.libero HF_HOME=$V/rlinf_home/hf XDG_CACHE_HOME=$V/rlinf_home/.cache
export PATH=$V/rlinf/bin:$PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
M=$V/rlinf_models/${MODEL##*/}
if [ ! -f $M/.done ]; then
  python -I -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$M')" && touch $M/.done
fi
du -sh $M
cd /coc/flash10/ksingara3/robot/ext/RLinf
bash evaluations/run_eval.sh $CFG rollout.model.model_path=$M env.eval.total_num_envs=${NENV:-50} env.eval.rollout_epoch=${EPOCHS:-2} \
  env.eval.video_cfg.save_video=False ${EXTRA:-}
echo "exit=$?"
