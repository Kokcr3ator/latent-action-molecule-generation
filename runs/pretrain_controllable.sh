#!/usr/bin/env bash
# Pretrain ControllableGPT (VQ-VAE latent action model + GPT).
# Usage: bash runs/pretrain_controllable.sh [--override key=val ...]
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m scripts.train --config configs/pretrain_controllable.yaml "$@"
