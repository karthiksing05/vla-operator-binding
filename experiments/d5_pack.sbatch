#!/bin/bash
# PROBE D5 / R6 (docs/PLAN_D5.md): N (batched policy server, sandbox client) pairs on ONE GPU.  New file, used only by D5.
#   sbatch -p overcap -A overcap --qos=short --requeue --exclude=... -o slurm/d5_sb_<s>_%j.out \
#          --export=ALL,D5SET=train,D5SHARD0=0,D5NSHARD=12,D5N=3 experiments/d5_pack.sbatch
# Instance i of this job runs shard (D5SHARD0 + i) / D5NSHARD of set D5SET.  D5EXTRA: extra client flags (e.g. --smoke).
#SBATCH --job-name=d5sb
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
if ! nvidia-smi -L 2>/dev/null | grep -q "GPU "; then
  echo "[guard] no GPU allocated on $(hostname) (restart ${SLURM_RESTART_COUNT:-0}); requeueing"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then scontrol requeue $SLURM_JOB_ID; sleep 30; fi
  exit 97
fi
REPO=/coc/flash10/ksingara3/lifelong-ood-learner
ROBOT=/coc/flash10/ksingara3/robot/openpi
cd $REPO
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -v -E '^SLURM_(JOB_ID|RESTART_COUNT|CPUS_PER_TASK)$'); do unset $v; done
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENPI_DATA_HOME=/coc/flash10/ksingara3/robot/openpi_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/coc/flash10/ksingara3/robot/data/hf/home HF_LEROBOT_HOME=/coc/flash10/ksingara3/robot/data/hf/lerobot
export BACKBONE_CONFIG=pi05_l90c BACKBONE_CKPT=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000
N=${D5N:-3}
echo "[d5 pack] $(date) on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); set $D5SET shards ${D5SHARD0}+${N} of ${D5NSHARD}"
mkdir -p results/d5/logs
base=$((9300 + (SLURM_JOB_ID % 200) * 3))
pids=()
for i in $(seq 0 $((N - 1))); do
  PORT=$((base + i))
  SH=$((D5SHARD0 + i))
  SLOG=results/d5/logs/server_${SLURM_JOB_ID}_$i.log
  ( $ROBOT/.venv/bin/python experiments/d5_serve.py --port $PORT > $SLOG 2>&1 & SRV=$!
    up=0
    for k in $(seq 1 240); do sleep 5; if grep -q "server listening" $SLOG 2>/dev/null; then up=1; break; fi; kill -0 $SRV 2>/dev/null || break; done
    if [ $up != 1 ]; then echo "[d5 pack] server $i failed"; tail -30 $SLOG; kill $SRV 2>/dev/null; exit 3; fi
    grep "parity" $SLOG
    PYTHONPATH=$ROBOT/third_party/libero $ROBOT/examples/libero/.venv/bin/python experiments/d5_goal.py --sandbox --set $D5SET \
        --shard $SH/$D5NSHARD --port $PORT ${D5EXTRA:-}
    rc=$?
    kill $SRV 2>/dev/null
    exit $rc ) > slurm/d5_inst_${SLURM_JOB_ID}_$i.out 2>&1 &
  pids+=($!)
  sleep 20
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=$?; done
for i in $(seq 0 $((N - 1))); do echo "--- instance $i"; tail -5 slurm/d5_inst_${SLURM_JOB_ID}_$i.out; done
echo "exit=$rc | $(date)"
exit $rc
