#!/usr/bin/env bash
# Pretrain base GPT on SMILES.
# Usage: bash scripts/pretrain_base/run.sh [--override key=val ...]
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 -m scripts.train --config scripts/pretrain_base/config.yaml "$@"
