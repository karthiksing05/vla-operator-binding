#!/bin/bash
# CALVIN-C P1 data prep: check task_ABC_D against the zip index and write the authors' extracted/ rel_actions files
# (experiments/c_calvin_extract.py); download mbreuss/flower_vla_pret into the HF cache.  CPU only; resumable (each step
# skips finished output).
#SBATCH --job-name=ccalvin-data
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_data.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/results/calvin_c/logs/%j_data.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
set -u
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=0
PY=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
echo "[data] $(date) $(hostname)"
$PY -c "from huggingface_hub import snapshot_download as s; print(s('mbreuss/flower_vla_pret', revision='ffd4134a2167a2be32bd7ad93455ed78dc2dcda2'))" || echo "[data] pret download FAILED"
$PY experiments/c_calvin_extract.py --root /coc/flash10/ksingara3/robot/data/calvin_abc --workers 32
echo "[data] exit=$? $(date)"
