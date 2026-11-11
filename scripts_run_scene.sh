#!/usr/bin/env bash
# Full pipeline for one scene, with per-stage wall-clock timing.
#   ./scripts_run_scene.sh configs/data_island.yaml [--fresh]
# --fresh removes the ONBOARD caches (features, matches, sweep signals) so every onboard stage is really timed;
# evaluation-side caches (LiDAR maps, Terra alignment) are kept. Timings: <outputs>/timings.csv
set -euo pipefail
cd "$(dirname "$0")"
D=${1:-configs/data.yaml}; FRESH=${2:-}; PY=.venv/bin/python
O=$($PY -c "import sys; sys.path.insert(0,'src'); from aerodiag.config import load_config; d=load_config('$D'); print(d['root'] + '/' + d['paths']['outputs'])")
mkdir -p "$O"
if [ "$FRESH" = "--fresh" ]; then
  rm -rf "$O"/cache/features_sift* "$O"/cache/track_b_matches.npz "$O"/step2/cell_pair_signals.hash
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)
CPU=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
echo "stage,seconds,backend,gpu,cpu,fresh" > "$O/timings.csv"
run() {  # stage name, command...
  local s=$1; shift; local t0=$(date +%s.%N)
  "$@" > "$O/$s.log" 2>&1
  echo "$s,$(echo "$(date +%s.%N) - $t0" | bc),${AERODIAG_BACKEND:-auto},$GPU,$CPU,${FRESH:-no}" >> "$O/timings.csv"
  echo "$s done ($(tail -1 "$O/timings.csv" | cut -d, -f2 | cut -c1-6) s)"
}
run phase0   $PY -m aerodiag.cli.phase0  --data "$D"
run track_a  $PY -m aerodiag.cli.track_a --data "$D"
run track_b  $PY -m aerodiag.cli.track_b --data "$D"
run step2    $PY -m aerodiag.cli.step2   --data "$D"
run step3    $PY -m aerodiag.cli.step3   --data "$D"
run calibrate_logreg $PY -m aerodiag.cli.calibrate --data "$D" --model logreg
run calibrate_gbdt   $PY -m aerodiag.cli.calibrate --data "$D" --model gbdt
run summary  $PY -m aerodiag.cli.summary --data "$D"
