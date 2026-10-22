#!/bin/bash
# Plan D2: RLinf's embodied stack for openpi (pi0 / pi0.5) + LIBERO, in its own venv under robot/venvs (untrusted code).
# HOME is redirected so the installer's LIBERO config reset (set_libero_default_path -> ~/.libero/config.yaml), uv / HF caches
# and any ~/.mujoco writes stay out of the user's real home; --no-root skips system packages; flash-attn / apex / natten skipped.
#SBATCH --job-name=g30rlinf
#SBATCH --output=/coc/flash10/ksingara3/lifelong-ood-learner/slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
set -x
V=/coc/flash10/ksingara3/robot/venvs
export HOME=$V/rlinf_home LIBERO_CONFIG_PATH=$V/rlinf_home/.libero
mkdir -p $HOME
export UV_CACHE_DIR=$V/uv_cache HF_HOME=$V/rlinf_home/hf XDG_CACHE_HOME=$V/rlinf_home/.cache PIP_CACHE_DIR=$V/rlinf_home/pip
if [ ! -x $V/uvboot/bin/uv ]; then python3 -m venv $V/uvboot && $V/uvboot/bin/pip install -q uv; fi
export PATH=$V/uvboot/bin:$PATH
cd /coc/flash10/ksingara3/robot/ext/RLinf
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv $V/rlinf --no-root --no-flash-attn --no-apex --no-natten
rc=$?
echo "install exit=$rc"
$V/rlinf/bin/python -c "import torch, openpi, libero; print('torch', torch.__version__); print('openpi', openpi.__file__); print('libero', libero.__file__)"
echo "real libero config unchanged?"; diff $V/libero_config_backup_2026-10-20.yaml /nethome/ksingara3/.libero/config.yaml && echo SAME
