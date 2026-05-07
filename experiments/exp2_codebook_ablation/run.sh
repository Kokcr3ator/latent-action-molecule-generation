#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Codebook size ablation
#
# Research questions addressed:
#   RQ2: Can we compress the action space without hurting RL performance?
#   RQ3: What is the impact of codebook size on generation quality (VUN)?
#
# Design:
#   - Base GPT: vocab_size ∈ {50, 512, 4096}
#   - ControllableGPT: num_latents ∈ {25, 128, 256}
#   - All 5 reward tasks: qed, logp, sa, mw, tpsa
#   - 3 RL seeds; 1 pretraining seed
#
# Total runs: 3 + 3 + 3 + (3×5×3) + (3×5×3) = 99
# =============================================================================
set -euo pipefail

ROOT="$(dirname "$0")/../.."
CFG="$(dirname "$0")/config.yaml"
cd "${ROOT}"

PRETRAIN_SEED=42
RL_SEEDS=(42 43 44)
TASKS=(qed logp sa mw tpsa)

BASE_VOCAB_SIZES=(50 512 4096)
CTRL_NUM_LATENTS=(25 128 256)
CTRL_VOCAB_SIZE=500   # ControllableGPT always uses the standard vocab

# ---------------------------------------------------------------------------
echo "===== [1/5] Pretrain base GPT — all vocab sizes ====="
for V in "${BASE_VOCAB_SIZES[@]}"; do
  echo "  vocab_size=${V}"
  python3 -m scripts.train \
    --config "${CFG}" --stage pretrain_base \
    --override seed=${PRETRAIN_SEED} \
               tokenizer.vocab_size=${V}
done

# ---------------------------------------------------------------------------
echo "===== [2/5] Pretrain ControllableGPT — all codebook sizes ====="
for N in "${CTRL_NUM_LATENTS[@]}"; do
  echo "  num_latents=${N}"
  python3 -m scripts.train \
    --config "${CFG}" --stage pretrain_controllable \
    --override seed=${PRETRAIN_SEED} \
               model.num_latents=${N} \
               tokenizer.vocab_size=${CTRL_VOCAB_SIZE}
done

# ---------------------------------------------------------------------------
echo "===== [3/5] Policy distillation — all codebook sizes ====="
for N in "${CTRL_NUM_LATENTS[@]}"; do
  echo "  num_latents=${N}"
  CTRL_CKPT="ckpts/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${PRETRAIN_SEED}"
  python3 -m scripts.train \
    --config "${CFG}" --stage policy_distillation \
    --override seed=${PRETRAIN_SEED} \
               tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
               loader.controllable_gpt_path="${CTRL_CKPT}/best.pt" \
               model.lm_head_out_size=${N}
done

# ---------------------------------------------------------------------------
echo "===== [4/5] PPO finetune base GPT — all vocab sizes, tasks, and seeds ====="
for V in "${BASE_VOCAB_SIZES[@]}"; do
  BASE_CKPT="ckpts/pretrain_base_vocab${V}_seed${PRETRAIN_SEED}"
  for TASK in "${TASKS[@]}"; do
    for S in "${RL_SEEDS[@]}"; do
      echo "  vocab=${V}, task=${TASK}, seed=${S}"
      python3 -m scripts.train_ppo \
        --config "${CFG}" --stage finetune_base \
        --override seed=${S} \
                   reward.task=${TASK} \
                   tokenizer.vocab_size=${V} \
                   ckpt.init_from=resume \
                   ckpt.path="${BASE_CKPT}" \
                   ckpt.ckpt_name="best.pt" \
                   experiment.wandb_run_name="ppo_${TASK}_base_vocab${V}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
    done
  done
done

# ---------------------------------------------------------------------------
echo "===== [5/5] PPO finetune ControllableGPT — all codebook sizes, tasks, and seeds ====="
for N in "${CTRL_NUM_LATENTS[@]}"; do
  CTRL_CKPT="ckpts/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${PRETRAIN_SEED}"
  DISTILL_CKPT="ckpts/policydistillation_nlatents${N}_vocab${CTRL_VOCAB_SIZE}_seed${PRETRAIN_SEED}"
  for TASK in "${TASKS[@]}"; do
    for S in "${RL_SEEDS[@]}"; do
      echo "  num_latents=${N}, task=${TASK}, seed=${S}"
      python3 -m scripts.train_ppo \
        --config "${CFG}" --stage finetune_controllable \
        --override seed=${S} \
                   reward.task=${TASK} \
                   tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                   ckpt.init_from=resume \
                   ckpt.path="${DISTILL_CKPT}" \
                   ckpt.ckpt_name="best.pt" \
                   loader.ckpt_controllable_path="${CTRL_CKPT}" \
                   loader.ckpt_name="best.pt" \
                   experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${N}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
    done
  done
done

echo "===== Experiment 2 complete ====="
