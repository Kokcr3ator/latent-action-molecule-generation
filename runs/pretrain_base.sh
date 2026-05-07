#!/usr/bin/env bash
# Pretrain base GPT on SMILES.
# Usage: bash runs/pretrain_base.sh [--override key=val ...]
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m scripts.train --config configs/pretrain_base.yaml "$@"
