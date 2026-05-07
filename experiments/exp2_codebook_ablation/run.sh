#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Codebook size ablation (ControllableGPT only)
#
# Research questions:
#   RQ2: Can we compress the action space without hurting RL performance?
#   RQ3: What is the impact of codebook size on generation quality (VUN)?
#
# Sweeps ControllableGPT codebook sizes; base GPT baseline is in exp1.
# Runs the full pipeline for each seed before moving to the next.
# Total runs: (1+1+5) × |num_latents| × |seeds| = 7 × 6 × 3 = 126
# =============================================================================
set -euo pipefail

ROOT="$(dirname "$0")/../.."
CFG="$(dirname "$0")/config.yaml"
cd "${ROOT}"

_cfg() { python3 -c "
import yaml
c = yaml.safe_load(open('${CFG}'))
v = c['$1']
print(' '.join(map(str, v)) if isinstance(v, list) else v)
"; }

_done() {
  [ -f "$1/best.pt" ] || return 1
  echo "    skipping — $1/best.pt already exists"
}

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
read -ra NUM_LATENTS_LIST <<< "$(_cfg num_latents_list)"
CTRL_VOCAB_SIZE=$(_cfg ctrl_vocab_size)
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
WANDB_GROUP=$(_cfg wandb_group)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="

  # -------------------------------------------------------------------------
  echo "  [1/3] Pretrain ControllableGPT — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${S}"
    if ! _done "${CTRL_CKPT}"; then
      python3 -m scripts.train \
        --config scripts/pretrain_controllable/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   model.num_latents=${N} \
                   tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                   training.max_iters=${CONTROLLABLE_ITERS} \
                   log.group=${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [2/3] Policy distillation — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${CTRL_VOCAB_SIZE}_seed${S}"
    if ! _done "${DISTILL_CKPT}"; then
      python3 -m scripts.train \
        --config scripts/policy_distillation/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                   loader.controllable_gpt_path="${CTRL_CKPT}/best.pt" \
                   model.lm_head_out_size=${N} \
                   training.max_iters=${DISTILLATION_ITERS} \
                   log.group=${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [3/3] PPO finetune ControllableGPT — all codebook sizes and tasks"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${CTRL_VOCAB_SIZE}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "    num_latents=${N}, task=${TASK}"
      CTRL_PPO_CKPT="${CKPT_ROOT}/ppo_${TASK}_controllable_nlatents${N}_envs16_steps256_seed${S}"
      if ! _done "${CTRL_PPO_CKPT}"; then
        python3 -m scripts.train_ppo \
          --config scripts/finetune_controllable/config.yaml \
          --override seed=${S} \
                     data.smiles=${DATA_DIR} \
                     reward.task=${TASK} \
                     tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                     ckpt.init_from=resume \
                     ckpt.path="${DISTILL_CKPT}" \
                     ckpt.ckpt_name="best.pt" \
                     loader.ckpt_controllable_path="${CTRL_CKPT}" \
                     loader.ckpt_name="best.pt" \
                     ppo.budget=${RL_BUDGET} \
                     log.group=${WANDB_GROUP} \
                     experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${N}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
      fi
    done
  done

done

echo "===== Experiment 2 complete ====="
