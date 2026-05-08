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
  [ -f "$1/done" ] || return 1
  echo "    skipping — $1/done sentinel exists"
}

_wandb_flag() { [ "$1" = "true" ] && echo "--wandb.enabled" || echo "--wandb.no-enabled"; }

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
BATCH_SIZE=$(_cfg batch_size)
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
    python3 -m scripts.train pretrain-base \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --training.max-iters ${PRETRAIN_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      $(_wandb_flag ${WANDB_LOG})
  fi

  # -------------------------------------------------------------------------
  echo "  [2/5] Pretrain ControllableGPT — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    if ! _done "${CTRL_CKPT}"; then
      python3 -m scripts.train pretrain-controllable \
        --seed ${S} \
        --data-smiles ${DATA_DIR} \
        --tokenizer.vocab-size ${VOCAB_SIZE} \
        --model.num-latents ${N} \
        --training.max-iters ${CONTROLLABLE_ITERS} \
        --training.batch-size ${BATCH_SIZE} \
        --wandb.project ${WANDB_PROJECT} \
        --wandb.entity ${WANDB_ENTITY} \
        --wandb.group ${WANDB_GROUP} \
        $(_wandb_flag ${WANDB_LOG})
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [3/5] Policy distillation — all codebook sizes"
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${VOCAB_SIZE}_seed${S}"
    if ! _done "${DISTILL_CKPT}"; then
      python3 -m scripts.train policy-distill \
        --seed ${S} \
        --data-smiles ${DATA_DIR} \
        --tokenizer.vocab-size ${VOCAB_SIZE} \
        --controllable-gpt-path "${CTRL_CKPT}/best.pt" \
        --model.num-latents ${N} \
        --training.max-iters ${DISTILLATION_ITERS} \
        --training.batch-size ${BATCH_SIZE} \
        --wandb.project ${WANDB_PROJECT} \
        --wandb.entity ${WANDB_ENTITY} \
        --wandb.group ${WANDB_GROUP} \
        $(_wandb_flag ${WANDB_LOG})
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [4/5] PPO finetune base GPT — all tasks"
  for TASK in "${TASKS[@]}"; do
    echo "    task=${TASK}"
    BASE_PPO_CKPT="${CKPT_ROOT}/ppo_${TASK}_envs16_steps256_seed${S}"
    if ! _done "${BASE_PPO_CKPT}"; then
      python3 -m scripts.train_ppo finetune-base \
        --seed ${S} \
        --data-smiles ${DATA_DIR} \
        --task ${TASK} \
        --tokenizer.vocab-size ${VOCAB_SIZE} \
        --pretrained-ckpt "${BASE_CKPT}" \
        --ppo.budget ${RL_BUDGET} \
        --wandb.project ${WANDB_PROJECT} \
        --wandb.entity ${WANDB_ENTITY} \
        --wandb.group ${WANDB_GROUP} \
        $(_wandb_flag ${WANDB_LOG})
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
        python3 -m scripts.train_ppo finetune-controllable \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --task ${TASK} \
          --tokenizer.vocab-size ${VOCAB_SIZE} \
          --pretrained-ckpt "${DISTILL_CKPT}" \
          --controllable-gpt-path "${CTRL_CKPT}" \
          --num-latents ${N} \
          --ppo.budget ${RL_BUDGET} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group ${WANDB_GROUP} \
          $(_wandb_flag ${WANDB_LOG})
      fi
    done
  done

done

echo "===== Experiment 2 complete ====="
