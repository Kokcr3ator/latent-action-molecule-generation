#!/usr/bin/env bash
# Policy distillation: train PolicyNetwork to imitate ControllableGPT in latent action space.
# Requires: loader.controllable_gpt_path and model.num_latents overrides.
# Usage: bash scripts/policy_distillation/run.sh --override loader.controllable_gpt_path=<path> model.num_latents=<N> [...]
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 -m scripts.train --config scripts/policy_distillation/config.yaml "$@"
