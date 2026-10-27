#!/bin/bash
#SBATCH --job-name=lolcmg
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
cd /coc/flash10/ksingara3/vla-operator-binding
export PYTHONIOENCODING=utf-8
srun $HOME/flash/miniconda3/envs/chunks/bin/python tests/diag_ascent_colormnist.py
rc=$?
echo "exit=$rc | $(date)"
exit $rc
