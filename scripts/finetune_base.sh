#!/usr/bin/env bash
# PPO finetune base GPT in token vocabulary space.
# Requires: ckpt.init_from=resume, ckpt.path, ckpt.ckpt_name, reward.task overrides.
# Usage: bash runs/finetune_base.sh --override ckpt.init_from=resume ckpt.path=<dir> ckpt.ckpt_name=best.pt reward.task=qed [...]
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m scripts.train_ppo --config configs/finetune_base.yaml "$@"
