#!/bin/bash
# X4 Meta-World: build the evaluation venv (CPU job; resumable: every step is idempotent).
#   robot/metaworld/.venv : Python 3.11, torch 2.7.1 (rlinf-openpi's pin), rlinf-openpi 0.1.1 (openpi PyTorch + its
#   patched transformers), metaworld 3.0.0 (Farama; needs gymnasium >= 1.1, overriding lerobot 0.3.3's < 1.0 pin), ray
#   (RLinf's import chain).  RLinf is cloned at a fixed commit into robot/metaworld/ext/RLinf (read via PYTHONPATH).
#   Also downloads RLinf/RLinf-Pi0-MetaWorld-SFT (7 GB) into the shared HF cache.
#   sbatch --partition=overcap --account=overcap --qos=short --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw slurm/jobs/x4_metaworld_install.sbatch
#SBATCH --job-name=x4mwinst
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -eu
ROOT=/coc/flash10/ksingara3/robot/metaworld
UV=/coc/flash10/ksingara3/robot/venvs/uvboot/bin/uv
PYBASE=/coc/flash10/ksingara3/robot/venvs/rlinf_home/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
RLINF_COMMIT=${RLINF_COMMIT:-91a302e}
export UV_CACHE_DIR=/coc/flash10/ksingara3/robot/venvs/uv_cache UV_LINK_MODE=copy
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_ENABLE_HF_TRANSFER=0
echo "[x4inst] $(date) $(hostname)"
mkdir -p $ROOT/ext $ROOT/build
# 1. RLinf at a fixed commit (its own directory; code is read, not installed)
if [ ! -d $ROOT/ext/RLinf/.git ]; then
  git clone -q https://github.com/RLinf/RLinf $ROOT/ext/RLinf
fi
git -C $ROOT/ext/RLinf checkout -q $RLINF_COMMIT
git -C $ROOT/ext/RLinf log -1 --format='[x4inst] RLinf %H %ci'
# 2. the venv
[ -x $ROOT/.venv/bin/python ] || $UV venv -q --python $PYBASE $ROOT/.venv
cat > $ROOT/build/overrides.txt <<'EOF'
gymnasium==1.1.1
torch==2.7.1
EOF
$UV pip install --python $ROOT/.venv/bin/python --override $ROOT/build/overrides.txt \
  "torch==2.7.1" "torchvision==0.22.1" "rlinf-openpi==0.1.1" "lerobot==0.3.3" "metaworld==3.0.0" "gymnasium==1.1.1" \
  "ray[default]>=2.47.0" "hydra-core<1.4.0.dev8" omegaconf "imageio[ffmpeg]" safetensors "huggingface_hub<1.0" \
  einops scipy pyarrow"<26" "numpy<2" opencv-python-headless pytest accelerate "datasets==3.6.0" torchdata peft timm
# 2b. stock transformers (pulled in by peft / timm / accelerate) and rlinf-transformer-openpi (openpi's patched 4.53.2) share
#     the transformers/ dir; as RLinf's install.sh does, drop the stock one and reinstall the patched one last
if $UV pip freeze --python $ROOT/.venv/bin/python | grep -q "^transformers=="; then
  $UV pip uninstall --python $ROOT/.venv/bin/python transformers
  $UV pip install --python $ROOT/.venv/bin/python --reinstall --no-deps "rlinf-transformer-openpi==4.53.2"
fi
$ROOT/.venv/bin/python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print('[x4inst] transformers_replace OK')"
$UV pip freeze --python $ROOT/.venv/bin/python > $ROOT/build/venv_freeze.txt
grep -i -E "^(torch|rlinf|openpi|transformers|metaworld|mujoco|gymnasium|numpy|jax|lerobot|ray)" $ROOT/build/venv_freeze.txt
# 3. the checkpoint
$ROOT/.venv/bin/python -I -c "
from huggingface_hub import snapshot_download
for r in ['RLinf/RLinf-Pi0-MetaWorld-SFT']:
    print(r, snapshot_download(r))
"
# 4. import check (CPU): metaworld registers its envs; RLinf's metaworld train config resolves
cd $ROOT/ext/RLinf
PYTHONPATH=$ROOT/ext/RLinf $ROOT/.venv/bin/python -c "
import metaworld, gymnasium, mujoco; metaworld.register_mw_envs(); print('metaworld', metaworld.__file__, 'gymnasium', gymnasium.__version__, 'mujoco', mujoco.__version__)
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
c = _CONFIGS_DICT['pi0_metaworld']; print(c.name, c.model.action_horizon, c.model.action_dim)
"
echo "[x4inst] done $(date)"
