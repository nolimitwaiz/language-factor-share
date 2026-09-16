#!/bin/bash
# Sync the metric project to CLSP. Usage: bash cluster/sync_mlm.sh
set -eo pipefail
HOST="${CLUSTER_HOST:-wkhan12@login.clsp.jhu.edu}"
DEST="~/multilingual-metrics"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" "mkdir -p $DEST/cluster/logs $DEST/results"
rsync -avz "$LOCAL/code" "$LOCAL/cluster" "$LOCAL/METRIC_PROPOSAL.md" "$HOST:$DEST/"
# NTREX text files only (~30 MB)
rsync -avz --include='newstest2019-*.txt' --exclude='*' \
    "$LOCAL/data/NTREX/NTREX-128/" "$HOST:$DEST/data/NTREX/NTREX-128/"
echo "[sync] done -> $HOST:$DEST"
echo "Next: ssh $HOST 'cd multilingual-metrics && sbatch cluster/submit_grid.sh'"
