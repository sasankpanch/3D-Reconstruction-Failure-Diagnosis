#!/usr/bin/env bash
# Full pipeline for one scene:  ./scripts_run_scene.sh configs/data_island.yaml
set -euo pipefail
cd "$(dirname "$0")"
D=${1:-configs/data.yaml}; PY=.venv/bin
$PY/aerodiag-phase0 --data "$D"
$PY/aerodiag-track-a --data "$D"
$PY/aerodiag-track-b --data "$D"
$PY/aerodiag-step2 --data "$D"
$PY/aerodiag-step3 --data "$D"
$PY/aerodiag-summary --data "$D"
