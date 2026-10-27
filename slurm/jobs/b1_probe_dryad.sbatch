#!/bin/bash
#SBATCH --job-name=b1probe
#SBATCH --output=/coc/flash10/ksingara3/vla-operator-binding/results/b1/logs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
hostname
for u in "https://datadryad.org/api/v2/files/3215085/download" "https://datadryad.org/downloads/file_stream/3215085"; do
  echo "== $u"; curl -s -L -A "Mozilla/5.0" -o /tmp/probe_$$ -w "%{http_code} %{content_type} %{size_download}\n" "$u"; head -c 200 /tmp/probe_$$; echo; rm -f /tmp/probe_$$
done
