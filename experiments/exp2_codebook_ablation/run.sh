#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Compression comparison: base GPT vs ControllableGPT
#
# Research question:
#   RQ2: Can we compress the action space without hurting RL performance?
#
# Trains a single base GPT baseline and sweeps ControllableGPT codebook sizes,
# comparing RL fine-tuning performance at each level of compression.
# Runs the full pipeline for each seed before moving to the next.
# Total runs: (1 + (1+1+5)×|num_latents| + 5) × |seeds|
#           = (1 + 7×6 + 5) × 5 = 48 × 5 = 240
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$(cd "$(dirname "$0")" && pwd)/config.yaml"
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
VOCAB_SIZE=$(_cfg vocab_size)
read -ra NUM_LATENTS_LIST <<< "$(_cfg num_latents_list)"
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
PRETRAIN_ITERS=$(_cfg pretrain_iters)
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_LOG=$(_cfg wandb_log)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="

  BASE_CKPT="${CKPT_ROOT}/pretrain_base_vocab${VOCAB_SIZE}_seed${S}"

  # -------------------------------------------------------------------------
  echo "  [1/5] Pretrain base GPT"
  if ! _done "${BASE_CKPT}"; then
    python3 -m scripts.train \
      --config scripts/pretrain_base/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 tokenizer.vocab_size=${VOCAB_SIZE} \
                 training.max_iters=${PRETRAIN_ITERS} \
                 wandb_log=${WANDB_LOG} \
                 experiment.wandb_project=${WANDB_PROJECT} \
                 experiment.wandb_entity=${WANDB_ENTITY} \
                 log.group=${WANDB_GROUP}
  fi

  # -------------------------------------------------------------------------
  echo "  [2/5] Pretrain ControllableGPT — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    if ! _done "${CTRL_CKPT}"; then
      python3 -m scripts.train \
        --config scripts/pretrain_controllable/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   model.num_latents=${N} \
                   tokenizer.vocab_size=${VOCAB_SIZE} \
                   training.max_iters=${CONTROLLABLE_ITERS} \
                   wandb_log=${WANDB_LOG} \
                   experiment.wandb_project=${WANDB_PROJECT} \
                   experiment.wandb_entity=${WANDB_ENTITY} \
                   log.group=${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [3/5] Policy distillation — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${VOCAB_SIZE}_seed${S}"
    if ! _done "${DISTILL_CKPT}"; then
      python3 -m scripts.train \
        --config scripts/policy_distillation/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   tokenizer.vocab_size=${VOCAB_SIZE} \
                   loader.controllable_gpt_path="${CTRL_CKPT}/best.pt" \
                   model.lm_head_out_size=${N} \
                   training.max_iters=${DISTILLATION_ITERS} \
                   wandb_log=${WANDB_LOG} \
                   experiment.wandb_project=${WANDB_PROJECT} \
                   experiment.wandb_entity=${WANDB_ENTITY} \
                   log.group=${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [4/5] PPO finetune base GPT — all tasks"
  for TASK in "${TASKS[@]}"; do
    echo "    task=${TASK}"
    BASE_PPO_CKPT="${CKPT_ROOT}/ppo_${TASK}_envs16_steps256_seed${S}"
    if ! _done "${BASE_PPO_CKPT}"; then
      python3 -m scripts.train_ppo \
        --config scripts/finetune_base/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   reward.task=${TASK} \
                   tokenizer.vocab_size=${VOCAB_SIZE} \
                   ckpt.init_from=resume \
                   ckpt.path="${BASE_CKPT}" \
                   ckpt.ckpt_name="best.pt" \
                   ppo.budget=${RL_BUDGET} \
                   wandb_log=${WANDB_LOG} \
                   experiment.wandb_project=${WANDB_PROJECT} \
                   experiment.wandb_entity=${WANDB_ENTITY} \
                   log.group=${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [5/5] PPO finetune ControllableGPT — all codebook sizes and tasks"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${VOCAB_SIZE}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "    num_latents=${N}, task=${TASK}"
      CTRL_PPO_CKPT="${CKPT_ROOT}/ppo_${TASK}_controllable_nlatents${N}_envs16_steps256_seed${S}"
      if ! _done "${CTRL_PPO_CKPT}"; then
        python3 -m scripts.train_ppo \
          --config scripts/finetune_controllable/config.yaml \
          --override seed=${S} \
                     data.smiles=${DATA_DIR} \
                     reward.task=${TASK} \
                     tokenizer.vocab_size=${VOCAB_SIZE} \
                     ckpt.init_from=resume \
                     ckpt.path="${DISTILL_CKPT}" \
                     ckpt.ckpt_name="best.pt" \
                     loader.ckpt_controllable_path="${CTRL_CKPT}" \
                     loader.ckpt_name="best.pt" \
                     ppo.budget=${RL_BUDGET} \
                     wandb_log=${WANDB_LOG} \
                     experiment.wandb_project=${WANDB_PROJECT} \
                     experiment.wandb_entity=${WANDB_ENTITY} \
                     log.group=${WANDB_GROUP} \
                     experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${N}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
      fi
    done
  done

done

echo "===== Experiment 2 complete ====="
