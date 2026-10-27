#!/bin/bash
# X2 parity: X-VLA (2toINF) on SimplerEnv with the authors' server (deploy.py) and clients (evaluation/simpler/*).
#   sbatch --export=ALL,MODEL=2toINF/X-VLA-WidowX,SUITE=WidowX,TASKS=spoon:carrot:blocks:eggplant,TAG=run0 slurm/jobs/x2_xvla_simpler.sbatch
# TASKS is colon-separated (sbatch --export splits on commas). Clients for all TASKS run in parallel against one server.
#SBATCH --job-name=x2xvla
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --error=/coc/flash10/ksingara3/vla-operator-binding/slurm/%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -u
export PYTHONIOENCODING=utf-8
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home
PY=/coc/flash10/ksingara3/robot/simpler/.venv/bin/python
export PATH=/coc/flash10/ksingara3/robot/simpler/.venv/bin:$PATH   # ffmpeg (imageio-ffmpeg static binary) for the clients' write_video
XVLA=/coc/flash10/ksingara3/robot/ext/xvla_src/X-VLA
REPO=/coc/flash10/ksingara3/vla-operator-binding
MODEL=${MODEL:-2toINF/X-VLA-WidowX}; SUITE=${SUITE:-WidowX}; TASKS=${TASKS:-spoon:carrot:blocks:eggplant}; TAG=${TAG:-run0}
PORT=${PORT:-$((8100 + SLURM_JOB_ID % 800))}
OUT=$REPO/results/x2_simpler/xvla_${SUITE}_${TAG}_${SLURM_JOB_ID}
mkdir -p $OUT; cd $OUT
echo "[x2] $(date) $(hostname) model=$MODEL suite=$SUITE tasks=$TASKS port=$PORT out=$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PY -c "import torch; print('[x2] cuda', torch.cuda.is_available())" || exit 3
$PY -I $REPO/experiments/x2_simpler_probe.py widowx_spoon_on_towel || { echo "[x2] SimplerEnv probe failed"; exit 4; }
$PY $XVLA/deploy.py --model_path $MODEL --host 127.0.0.1 --port $PORT --output_dir $OUT/server --disable_slurm > $OUT/server.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do
  if $PY -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))"; then break; fi; sleep 5
done
echo "[x2] server up after ~$((i*5))s"
T0=$(date +%s)
PIDS=""
for t in ${TASKS//:/ }; do
  SIMPLER_DIR=/coc/flash10/ksingara3/robot/ext/simpler_255isWhite/SimplerEnv \
  $PY $XVLA/evaluation/simpler/$SUITE/client_$t.py --server_ip 127.0.0.1 --server_port $PORT --output_dir $OUT/$t > $OUT/client_$t.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
echo "[x2] clients done in $(( $(date +%s) - T0 ))s"
kill $SPID
for t in ${TASKS//:/ }; do
  f=$(ls $OUT/$t/*results*.txt 2>/dev/null | head -1)
  [ -n "$f" ] && $PY -c "
import json,sys
rows=[json.loads(l) for l in open('$f') if l.strip()]
ok=[r for r in rows if 'done' in r]
print('[x2] $t', 'episodes', len(ok), 'errors', len(rows)-len(ok), 'success', sum(r['done'] for r in ok), '/', len(ok), '=', round(100*sum(r['done'] for r in ok)/max(1,len(ok)),1))
"
done
