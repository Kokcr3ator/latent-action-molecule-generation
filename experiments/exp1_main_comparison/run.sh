#!/usr/bin/env bash
# =============================================================================
# Experiment 1 — Main comparison: base GPT vs ControllableGPT
#
# Research questions addressed:
#   RQ1: Does a latent action model help RL fine-tuning?
#   RQ5: Does the model generalise across different molecular properties?
#
# Design:
#   - Base GPT (token vocab, vocab_size=500) vs ControllableGPT (num_latents=128)
#   - All 5 reward tasks: qed, logp, sa, mw, tpsa
#   - 3 RL seeds; 1 pretraining seed
#
# Total runs: 1 + 1 + 1 + (5×3) + (5×3) = 33
# =============================================================================
set -euo pipefail

ROOT="$(dirname "$0")/../.."
CFG="$(dirname "$0")/config.yaml"
cd "${ROOT}"

PRETRAIN_SEED=42
RL_SEEDS=(42 43 44)
TASKS=(qed logp sa mw tpsa)
VOCAB_SIZE=500
NUM_LATENTS=128

BASE_CKPT_DIR="ckpts/pretrain_base_vocab${VOCAB_SIZE}_seed${PRETRAIN_SEED}"
CTRL_CKPT_DIR="ckpts/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${PRETRAIN_SEED}"
DISTILL_CKPT_DIR="ckpts/policydistillation_nlatents${NUM_LATENTS}_vocab${VOCAB_SIZE}_seed${PRETRAIN_SEED}"

# ---------------------------------------------------------------------------
echo "===== [1/5] Pretrain base GPT (vocab=${VOCAB_SIZE}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config "${CFG}" --stage pretrain_base \
  --override seed=${PRETRAIN_SEED} \
             tokenizer.vocab_size=${VOCAB_SIZE}

# ---------------------------------------------------------------------------
echo "===== [2/5] Pretrain ControllableGPT (num_latents=${NUM_LATENTS}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config "${CFG}" --stage pretrain_controllable \
  --override seed=${PRETRAIN_SEED} \
             model.num_latents=${NUM_LATENTS} \
             tokenizer.vocab_size=${VOCAB_SIZE}

# ---------------------------------------------------------------------------
echo "===== [3/5] Policy distillation (num_latents=${NUM_LATENTS}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config "${CFG}" --stage policy_distillation \
  --override seed=${PRETRAIN_SEED} \
             tokenizer.vocab_size=${VOCAB_SIZE} \
             loader.controllable_gpt_path="${CTRL_CKPT_DIR}/best.pt" \
             model.lm_head_out_size=${NUM_LATENTS}

# ---------------------------------------------------------------------------
echo "===== [4/5] PPO finetune base GPT — all tasks, all seeds ====="
for TASK in "${TASKS[@]}"; do
  for S in "${RL_SEEDS[@]}"; do
    echo "  task=${TASK}, seed=${S}"
    python3 -m scripts.train_ppo \
      --config "${CFG}" --stage finetune_base \
      --override seed=${S} \
                 reward.task=${TASK} \
                 tokenizer.vocab_size=${VOCAB_SIZE} \
                 ckpt.init_from=resume \
                 ckpt.path="${BASE_CKPT_DIR}" \
                 ckpt.ckpt_name="best.pt"
  done
done

# ---------------------------------------------------------------------------
echo "===== [5/5] PPO finetune ControllableGPT — all tasks, all seeds ====="
for TASK in "${TASKS[@]}"; do
  for S in "${RL_SEEDS[@]}"; do
    echo "  task=${TASK}, seed=${S}"
    python3 -m scripts.train_ppo \
      --config "${CFG}" --stage finetune_controllable \
      --override seed=${S} \
                 reward.task=${TASK} \
                 tokenizer.vocab_size=${VOCAB_SIZE} \
                 ckpt.init_from=resume \
                 ckpt.path="${DISTILL_CKPT_DIR}" \
                 ckpt.ckpt_name="best.pt" \
                 loader.ckpt_controllable_path="${CTRL_CKPT_DIR}" \
                 loader.ckpt_name="best.pt" \
                 experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${NUM_LATENTS}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
  done
done

echo "===== Experiment 1 complete ====="
