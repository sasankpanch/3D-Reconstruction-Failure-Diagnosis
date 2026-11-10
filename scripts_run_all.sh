#!/usr/bin/env bash
# Full pipeline, end to end. Cached intermediates (outputs/cache, per-stage hashes) make re-runs skip completed work.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin
$PY/aerodiag-phase0
$PY/aerodiag-track-a
$PY/aerodiag-track-b
$PY/aerodiag-step2
$PY/aerodiag-step3
$PY/aerodiag-summary
