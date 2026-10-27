#!/bin/bash
# X1 parity: X-VLA (2toINF/X-VLA-Calvin-ABC_D) on CALVIN ABC->D, the authors' server (X-VLA deploy.py) + the authors' client
# (evaluation/calvin/calvin_client.py via experiments/x1_calvin_xvla_client.py), one server per client as in their launch.sh.
#   sbatch --export=ALL,START=0,END=50,NPROC=5,EPLEN=720 slurm/jobs/x1_calvin_xvla.sbatch
#SBATCH --job-name=x1xvla
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYOPENGL_PLATFORM=egl
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
PYC=/coc/flash10/ksingara3/robot/calvin/.venv/bin/python
PYX=/coc/flash10/ksingara3/robot/calvin/.venv-xvla/bin/python
REPO=/coc/flash10/ksingara3/vla-operator-binding
XVLA=/coc/flash10/ksingara3/robot/ext/X-VLA/X-VLA
CKPT=/coc/flash10/ksingara3/robot/data/hf/home/hub/models--2toINF--X-VLA-Calvin-ABC_D/snapshots/d76710ee314ee1fa8506f421664c989b40bae415
WORK=/coc/flash10/ksingara3/robot/calvin
CLIENT=${CLIENT:-$XVLA/evaluation/calvin/calvin_client.py}
START=${START:-0}; END=${END:-50}; NPROC=${NPROC:-5}; EPLEN=${EPLEN:-720}
BASE_PORT=$(( 20000 + (SLURM_JOB_ID % 1000) * 20 ))
OUT=${OUT:-$REPO/results/x1_calvin/xvla_${START}_${END}_ep${EPLEN}_${SLURM_JOB_ID}}
mkdir -p $OUT
echo "[x1xvla] $(date) $(hostname) range=[$START,$END) nproc=$NPROC ep_len=$EPLEN client=$CLIENT out=$OUT base_port=$BASE_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
N=$(( END - START )); STEP=$(( (N + NPROC - 1) / NPROC ))
SPIDS=(); i=0
for (( s=START; s<END; s+=STEP )); do
  P=$(( BASE_PORT + i )); mkdir -p $OUT/server_$i
  ( cd $XVLA && exec $PYX -m deploy --model_path $CKPT --output_dir $OUT/server_$i --host 127.0.0.1 --port $P \
      --disable_slurm ) > $OUT/server_$i.log 2>&1 &
  SPIDS+=($!); i=$(( i + 1 ))
done
for (( j=0; j<i; j++ )); do
  for k in $(seq 1 900); do [ -f $OUT/server_$j/info.json ] && break; sleep 2; done
  P=$(( BASE_PORT + j ))
  for k in $(seq 1 300); do curl -s -o /dev/null http://127.0.0.1:$P/docs && break; sleep 2; done
done
echo "[x1xvla] servers up after $(( $(date +%s) - T0 ))s"
CPIDS=(); i=0
for (( s=START; s<END; s+=STEP )); do
  e=$(( s + STEP )); (( e > END )) && e=$END; P=$(( BASE_PORT + i ))
  $PYC $REPO/experiments/x1_calvin_xvla_client.py --client $CLIENT --work_dir $WORK \
     --out $OUT/shard_${s}_${e} --port $P --start $s --end $e --ep_len $EPLEN --no_video > $OUT/client_${s}_${e}.txt 2>&1 &
  CPIDS+=($!); i=$(( i + 1 ))
done
wait ${CPIDS[@]}
kill ${SPIDS[@]} 2>/dev/null
echo "[x1xvla] done in $(( $(date +%s) - T0 ))s"
$PYC $REPO/experiments/x1_calvin_merge.py --dir $OUT
