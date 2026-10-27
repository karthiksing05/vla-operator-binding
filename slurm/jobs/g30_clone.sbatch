#!/bin/bash
# Plan D2: clone the published baselines' code, each into its own directory under robot/ext (untrusted code: nothing is run).
#SBATCH --job-name=g30clone
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
cd /coc/flash10/ksingara3/robot/ext
for spec in RLinf=https://github.com/RLinf/RLinf ReinFlow=https://github.com/ReinFlow/ReinFlow Lotus=https://github.com/UT-Austin-RPL/Lotus \
            IsCiL=https://github.com/L2dulgi/IsCiL SDP=https://github.com/AnthonyHuo/SDP demodiffusion=https://github.com/demodiffusion/demodiffusion; do
  name=${spec%%=*}; url=${spec#*=}
  if [ -d "$name/.git" ]; then echo "[clone] $name exists"; continue; fi
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-recurse-submodules "$url" "$name" && echo "[clone] $name $(git -C $name rev-parse HEAD) $(du -sh $name | cut -f1)"
done
echo done
