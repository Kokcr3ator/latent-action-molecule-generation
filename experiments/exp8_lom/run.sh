#!/usr/bin/env bash
# =============================================================================
# Experiment 8 — LOM architecture: bidirectional encoder + masked action token
#
# Research question:
#   Does the new GENIE-style LAM converge, and how does its pretraining and
#   distillation loss compare to the old causal-encoder LAM (exp3/exp6)?
#
# Stages run for each seed:
#   1. Pretrain ControllableGPT (new LOM architecture)
#   2. Policy distillation
#
# Uses a separate ckpt_root (_lom) to avoid colliding with old causal-LAM
# checkpoints from exp3/exp6 that share the same run-name pattern.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$(cd "$(dirname "$0")" && pwd)/config.yaml"
cd "${ROOT}"

_cfg() { python3 -c "
import yaml, os
c = yaml.safe_load(open('${CFG}'))
v = c['$1']
expand = lambda s: os.path.expandvars(str(s))
print(' '.join(expand(x) for x in v) if isinstance(v, list) else expand(v))
"; }

_done() {
  [ -f "$1/done" ] || return 1
  echo "    skipping — $1/done sentinel exists"
}

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
VOCAB_SIZE=$(_cfg vocab_size)
NUM_LATENTS=$(_cfg num_latents)
read -ra SEEDS <<< "$(_cfg seeds)"
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
BATCH_SIZE=$(_cfg batch_size)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_DIR=$(_cfg wandb_dir)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="

  CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"
  DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${NUM_LATENTS}_vocab${VOCAB_SIZE}_seed${S}"

  # -------------------------------------------------------------------------
  echo "  [1/2] Pretrain ControllableGPT (LOM encoder)"
  if ! _done "${CTRL_CKPT}"; then
    python3 -m scripts.train pretrain-controllable \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --ckpt-root ${CKPT_ROOT} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --model.num-latents ${NUM_LATENTS} \
      --training.max-iters ${CONTROLLABLE_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      --wandb.dir ${WANDB_DIR}
  fi

  # -------------------------------------------------------------------------
  echo "  [2/2] Policy distillation"
  if ! _done "${DISTILL_CKPT}"; then
    python3 -m scripts.train policy-distill \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --ckpt-root ${CKPT_ROOT} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --controllable-gpt-path "${CTRL_CKPT}/best.pt" \
      --model.num-latents ${NUM_LATENTS} \
      --training.max-iters ${DISTILLATION_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      --wandb.dir ${WANDB_DIR}
  fi

done

echo "===== Experiment 8 complete ====="
