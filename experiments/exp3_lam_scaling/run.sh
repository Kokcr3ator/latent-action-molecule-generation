#!/usr/bin/env bash
# =============================================================================
# Experiment 3 — LAM scaling: how does ControllableGPT scale with codebook size?
#
# Research question:
#   RQ3: What is the impact of codebook size on generation quality (VUN) and
#        RL performance?
#
# ControllableGPT only — no baseline.  Sweeps codebook sizes both smaller and
# larger than the token vocabulary to characterise the scaling behaviour of
# the latent action model.
# Runs the full pipeline for each seed before moving to the next.
# Total runs: (1+1+5) × |num_latents| × |seeds| = 7 × 6 × 5 = 210
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


DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
VOCAB_SIZE=$(_cfg vocab_size)
read -ra NUM_LATENTS_LIST <<< "$(_cfg num_latents_list)"
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
BATCH_SIZE=$(_cfg batch_size)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_DIR=$(_cfg wandb_dir)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="

  # -------------------------------------------------------------------------
  echo "  [1/3] Pretrain ControllableGPT — all codebook sizes"
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
        --wandb.dir ${WANDB_DIR}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [2/3] Policy distillation — all codebook sizes"
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
        --wandb.group ${WANDB_GROUP}
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [3/3] PPO finetune ControllableGPT — all codebook sizes and tasks"
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
          --wandb.group ${WANDB_GROUP}
      fi
    done
  done

done

echo "===== Experiment 3 complete ====="
