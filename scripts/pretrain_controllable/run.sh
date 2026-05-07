#!/usr/bin/env bash
# Pretrain ControllableGPT (VQ-VAE latent action model + GPT).
# Usage: bash scripts/pretrain_controllable/run.sh [--override key=val ...]
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 -m scripts.train --config scripts/pretrain_controllable/config.yaml "$@"
