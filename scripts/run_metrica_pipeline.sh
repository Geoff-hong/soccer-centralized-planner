#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MATCH_IDS=(1 2)

for MID in "${MATCH_IDS[@]}"; do
  echo "=== Metrica match ${MID}: track2event ==="
  python scripts/track2event.py --match_id "${MID}"

  echo "=== Metrica match ${MID}: data_processing ==="
  python scripts/data_processing.py --match_id "${MID}"

  echo "=== Metrica match ${MID}: check_data ==="
  python scripts/check_data.py --match_id "${MID}"
done
