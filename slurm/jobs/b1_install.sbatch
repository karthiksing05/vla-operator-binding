#!/bin/bash
#SBATCH --job-name=b1install
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/b1/logs/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
hostname
B=/coc/flash10/ksingara3/robot/composuite
UV=/coc/flash10/ksingara3/robot/venvs/uvboot/bin/uv
export UV_CACHE_DIR=/coc/flash10/ksingara3/robot/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/coc/flash10/ksingara3/robot/venvs/uv_python
cd $B
[ -x $B/.venv/bin/python ] || $UV venv --python 3.10 $B/.venv
PY=$B/.venv/bin/python
# robosuite 1.4.0 = CompoSuite requirements_default.txt pin (mujoco bindings, not mujoco-py)
$UV pip install --python $PY "robosuite==1.4.0" "mujoco==2.3.7" "gym==0.25.2" "h5py" "termcolor==2.4.0" "numpy<2" "scipy" "tqdm"
$UV pip install --python $PY --index-url https://download.pytorch.org/whl/cu121 "torch==2.4.1"
# CompoSuite: setup.py read 2026-10-21 (plain setuptools, deps robosuite/gym/h5py); install without deps, editable
$UV pip install --python $PY --no-deps -e $B/src/CompoSuite   # editable as in its README: the subpackages lack __init__.py, so find_packages misses them
$PY -I -c "import robosuite, mujoco, torch, composuite; print('robosuite', robosuite.__version__, 'mujoco', mujoco.__version__, 'torch', torch.__version__)"
