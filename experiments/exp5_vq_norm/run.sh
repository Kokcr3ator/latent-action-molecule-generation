#!/usr/bin/env bash
# =============================================================================
# Experiment 5 — VQ normalisation ablation
#
# Sweeps four codebook normalisation strategies (loss, codebook, step, norm_penalty)
# across codebook sizes and seeds.
# Total runs: 4 strategies × 4 codebook sizes × 3 seeds = 48
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
read -ra NORM_MODES      <<< "$(_cfg norm_modes)"
read -ra SEEDS           <<< "$(_cfg seeds)"
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
BATCH_SIZE=$(_cfg batch_size)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="
  for NORM in "${NORM_MODES[@]}"; do
    echo "  === norm_mode=${NORM} ==="
    for N in "${NUM_LATENTS_LIST[@]}"; do
      echo "    num_latents=${N}"
      CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_norm${NORM}_seed${S}"
      if ! _done "${CKPT}"; then
        python3 -m scripts.train pretrain-controllable \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --tokenizer.vocab-size ${VOCAB_SIZE} \
          --model.num-latents ${N} \
          --model.norm-mode ${NORM} \
          --training.max-iters ${CONTROLLABLE_ITERS} \
          --training.batch-size ${BATCH_SIZE} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group ${WANDB_GROUP}
      fi
    done
  done
done

echo "===== Experiment 5 complete ====="
