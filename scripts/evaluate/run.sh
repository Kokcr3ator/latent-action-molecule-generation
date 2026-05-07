#!/usr/bin/env bash
# Generate molecules from a checkpoint and report VUN metrics.
# Usage: bash scripts/evaluate/run.sh [--override key=val ...]
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 -m scripts.evaluate --config scripts/evaluate/config.yaml "$@"
