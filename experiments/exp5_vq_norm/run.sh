#!/usr/bin/env bash
# =============================================================================
# Experiment 5 — VQ normalisation ablation
#
# Sweeps four codebook normalisation strategies (loss, codebook, step, norm_penalty)
# across codebook sizes and seeds.
# Total runs: 4 strategies × 4 codebook sizes × 3 seeds = 48
#
# If 2+ GPUs are available, runs two jobs in parallel (one per GPU).
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

# ---------------------------------------------------------------------------
# GPU parallelism: fill one slot per GPU, wait when all slots are full.
# Respects CUDA_VISIBLE_DEVICES if set; otherwise detects all GPUs via nvidia-smi.
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  NUM_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
  GPU_IDS=()
  for i in $(seq 0 $((NUM_DETECTED - 1))); do GPU_IDS+=("$i"); done
fi
NUM_GPUS=${#GPU_IDS[@]}
echo "Using ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

_GPU_SLOT=0
_PIDS=()

_launch() {
  if [ "${NUM_GPUS}" -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${GPU_IDS[${_GPU_SLOT}]} "$@" &
    _PIDS+=($!)
    _GPU_SLOT=$(( (_GPU_SLOT + 1) % NUM_GPUS ))
    if [ ${#_PIDS[@]} -ge "${NUM_GPUS}" ]; then
      wait "${_PIDS[@]}"
      _PIDS=()
    fi
  else
    "$@"
  fi
}

_flush() {
  if [ ${#_PIDS[@]} -gt 0 ]; then
    wait "${_PIDS[@]}"
    _PIDS=()
  fi
}

# ---------------------------------------------------------------------------

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
WANDB_DIR=$(_cfg wandb_dir)

for S in "${SEEDS[@]}"; do
  echo "========== Seed ${S} =========="
  for NORM in "${NORM_MODES[@]}"; do
    echo "  === norm_mode=${NORM} ==="
    for N in "${NUM_LATENTS_LIST[@]}"; do
      echo "    num_latents=${N}"
      CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_norm${NORM}_seed${S}"
      if ! _done "${CKPT}"; then
        _launch python3 -m scripts.train pretrain-controllable \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --tokenizer.vocab-size ${VOCAB_SIZE} \
          --model.num-latents ${N} \
          --model.norm-mode ${NORM} \
          --training.max-iters ${CONTROLLABLE_ITERS} \
          --training.batch-size ${BATCH_SIZE} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group ${WANDB_GROUP} \
          --wandb.dir ${WANDB_DIR}
      fi
    done
  done
done

_flush
echo "===== Experiment 5 complete ====="
