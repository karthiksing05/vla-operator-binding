#!/bin/bash
# Plan D2 / G30f: LOTUS (arXiv 2311.02058; github.com/UT-Austin-RPL/Lotus @ce38726, NO LICENCE: research use only, never
# redistributed) in its own venv.  py3.9 + torch 1.11.0+cu113 as its README; robosuite / mujoco / robomimic pinned to the
# versions of openpi's LIBERO client venv (LOTUS's requirements leave robosuite unpinned).  The code is cloned to a work copy
# (robot/work/lotus) so the pristine clone in robot/ext/Lotus is never written; HOME and LIBERO_CONFIG_PATH are redirected so
# the user's ~/.libero is untouched.  Also downloads LIBERO-OBJECT's hdf5 (HF mirror yifengzhu-hf/LIBERO-datasets; the Box
# links in LOTUS's downloader return 404).  Idempotent: each step is skipped once done.
#SBATCH --job-name=g30lotinst
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
set -x
V=/coc/flash10/ksingara3/robot/venvs
W=/coc/flash10/ksingara3/robot/work/lotus
D=/coc/flash10/ksingara3/robot/data/lotus_datasets
export HOME=$V/lotus_home LIBERO_CONFIG_PATH=$V/lotus_home/.libero
mkdir -p $HOME $LIBERO_CONFIG_PATH $D
export UV_CACHE_DIR=$V/uv_cache HF_HOME=$V/lotus_home/hf XDG_CACHE_HOME=$V/lotus_home/.cache PIP_CACHE_DIR=$V/lotus_home/pip
export UV_PYTHON_INSTALL_DIR=$V/uv_python
export PATH=$V/uvboot/bin:$PATH
[ -d $W/.git ] || git clone /coc/flash10/ksingara3/robot/ext/Lotus $W
[ -e $W/lotus/datasets ] || ln -s $D $W/lotus/datasets
[ -e $W/datasets ] || ln -s $D $W/datasets
cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
assets: $W/lotus/libero/assets
bddl_files: $W/lotus/libero/bddl_files
benchmark_root: $W/lotus/libero
datasets: $D
init_states: $W/lotus/libero/init_files
EOF
P=$V/lotus/bin/python
if [ ! -f $V/lotus/.done ]; then
  [ -x $P ] || uv venv --python 3.9 $V/lotus || exit 2
  uv pip install --python $P "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" "torchaudio==0.11.0" \
      --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match || exit 3
  uv pip install --python $P hydra-core==1.2.0 numpy==1.22.4 wandb==0.13.1 easydict==1.9 transformers==4.21.1 \
      opencv-python==4.6.0.66 einops==0.4.1 thop==0.1.1-2209072238 robosuite==1.4.1 mujoco==3.2.3 bddl==1.0.1 future==0.18.2 \
      matplotlib==3.5.3 cloudpickle==2.1.0 gym==0.25.2 scikit-learn omegaconf torchmetrics==0.10.3 fvcore iopath submitit \
      h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg pyyaml huggingface_hub "protobuf<4" \
      "setuptools<70" || exit 4
  uv pip install --python $P --no-deps robomimic==0.2.0 || exit 5
  uv pip install --python $P --no-deps -e $W || exit 6
  touch $V/lotus/.done
fi
$P -c "import torch, robosuite, robomimic, transformers; print('torch', torch.__version__, torch.version.cuda); print('robosuite', robosuite.__version__)"
cd $W && $P -c "import lotus.libero as L; from lotus.libero.benchmark import get_benchmark; b=get_benchmark('libero_object_exp6')(0); print(b.n_tasks, b.get_task_names()); print(L.get_libero_path('datasets'))"
if [ ! -f $D/.libero_object.done ]; then
  $P -I -c "from huggingface_hub import snapshot_download; snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', allow_patterns=['libero_object/*'], local_dir='$D')" \
    && touch $D/.libero_object.done
fi
ls -la $D/libero_object | head -20
echo "install exit=0"
