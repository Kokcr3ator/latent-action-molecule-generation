#!/usr/bin/env bash
# =============================================================================
# Experiment 4 — Temperature ablation
#
# Research question:
#   RQ4: Does temperature have an impact on validity, uniqueness and novelty?
#
# Evaluates both pretrained models (base GPT and ControllableGPT) at each
# temperature value and reports VUN metrics.  No RL training is run here.
#
# Prerequisite: exp1 must have been run (or at least stages 1–2) so that
#   pretrain_base_vocab<V>_seed<S> and
#   pretrain_controllable_vocab<V>_nlatent<N>_seed<S>
# checkpoints exist.
#
# NOTE: requires scripts.evaluate — see scripts/evaluate/config.yaml.
#
# Total runs: 2 models × |temperatures| × |seeds| = 2 × 6 × 5 = 60
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
  [ -f "$1" ] || return 1
  echo "    skipping — $1 already exists"
}

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
VOCAB_SIZE=$(_cfg vocab_size)
NUM_LATENTS=$(_cfg num_latents)
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TEMPERATURES <<< "$(_cfg temperatures)"
N_MOLS=$(_cfg n_mols)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_LOG=$(_cfg wandb_log)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="

  BASE_CKPT="${CKPT_ROOT}/pretrain_base_vocab${VOCAB_SIZE}_seed${S}"
  CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"

  # -------------------------------------------------------------------------
  echo "  [1/2] Evaluate base GPT — all temperatures"
  for T in "${TEMPERATURES[@]}"; do
    echo "    temperature=${T}"
    RESULT="${CKPT_ROOT}/eval_base_vocab${VOCAB_SIZE}_temp${T}_seed${S}/results.json"
    if ! _done "${RESULT}"; then
      python3 -m scripts.evaluate \
        --config scripts/evaluate/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   model.type=base \
                   tokenizer.vocab_size=${VOCAB_SIZE} \
                   generation.temperature=${T} \
                   generation.n_mols=${N_MOLS} \
                   loader.ckpt_path="${BASE_CKPT}" \
                   loader.ckpt_name="best.pt" \
                   wandb_log=${WANDB_LOG} \
                   experiment.wandb_project=${WANDB_PROJECT} \
                   experiment.wandb_entity=${WANDB_ENTITY} \
                   log.group=${WANDB_GROUP} \
                   experiment.wandb_run_name="eval_base_vocab${VOCAB_SIZE}_temp${T}_seed${S}"
    fi
  done

  # -------------------------------------------------------------------------
  echo "  [2/2] Evaluate ControllableGPT — all temperatures"
  for T in "${TEMPERATURES[@]}"; do
    echo "    temperature=${T}"
    RESULT="${CKPT_ROOT}/eval_controllable_vocab${VOCAB_SIZE}_nlatents${NUM_LATENTS}_temp${T}_seed${S}/results.json"
    if ! _done "${RESULT}"; then
      python3 -m scripts.evaluate \
        --config scripts/evaluate/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   model.type=controllable \
                   tokenizer.vocab_size=${VOCAB_SIZE} \
                   generation.temperature=${T} \
                   generation.n_mols=${N_MOLS} \
                   loader.ckpt_path="${CTRL_CKPT}" \
                   loader.ckpt_name="best.pt" \
                   wandb_log=${WANDB_LOG} \
                   experiment.wandb_project=${WANDB_PROJECT} \
                   experiment.wandb_entity=${WANDB_ENTITY} \
                   log.group=${WANDB_GROUP} \
                   experiment.wandb_run_name="eval_controllable_vocab${VOCAB_SIZE}_nlatents${NUM_LATENTS}_temp${T}_seed${S}"
    fi
  done

done

echo "===== Experiment 4 complete ====="
