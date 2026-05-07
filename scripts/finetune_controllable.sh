#!/usr/bin/env bash
# PPO finetune ControllableGPT in latent action space.
# Requires: ckpt.init_from=resume, ckpt.path, ckpt.ckpt_name,
#           loader.ckpt_controllable_path, loader.ckpt_name, reward.task overrides.
# Usage: bash runs/finetune_controllable.sh --override ckpt.init_from=resume ckpt.path=<dir> ckpt.ckpt_name=best.pt \
#          loader.ckpt_controllable_path=<dir> loader.ckpt_name=best.pt reward.task=qed [...]
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m scripts.train_ppo --config configs/finetune_controllable.yaml "$@"
