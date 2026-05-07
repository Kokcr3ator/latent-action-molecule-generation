#!/usr/bin/env bash
# Policy distillation: train PolicyNetwork to imitate ControllableGPT in latent action space.
# Requires: loader.controllable_gpt_path and model.lm_head_out_size overrides.
# Usage: bash runs/policy_distillation.sh --override loader.controllable_gpt_path=<path> model.lm_head_out_size=<N> [...]
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m scripts.train --config configs/policy_distillation.yaml "$@"
