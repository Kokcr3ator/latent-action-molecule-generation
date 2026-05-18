#!/usr/bin/env bash
# =============================================================================
# Experiment 9 — Staged LOM training (GENIE-style two-stage pretraining)
#
# Research question:
#   Does training the LAM and dynamics model in two separate stages (LAM first,
#   then frozen-LAM dynamics) improve convergence or final quality compared to
#   the joint training in exp8?
#
# Stages:
#   1. Pretrain LAM only        (all seeds in parallel, one per GPU)
#   2. Pretrain dynamics only   (all seeds in parallel; LAM frozen from stage 1)
#   3. Policy distillation      (all seeds in parallel)
# =============================================================================
set -euo pipefail
FORCE=0
for _arg in "$@"; do [ "$_arg" = "--force" ] && FORCE=1; done

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
  [ "${FORCE}" = "1" ] && return 1
  [ -f "$1/done" ] || return 1
  echo "    skipping — $1/done sentinel exists"
}

# ---------------------------------------------------------------------------
# GPU parallelism
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

_COMPILE_STAGGER=30

_launch() {
  if [ "${NUM_GPUS}" -ge 2 ]; then
    if [ ${#_PIDS[@]} -ge "${NUM_GPUS}" ]; then
      wait -n
      local alive=()
      for pid in "${_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
      done
      _PIDS=("${alive[@]+"${alive[@]}"}")
    fi
    CUDA_VISIBLE_DEVICES=${GPU_IDS[${_GPU_SLOT}]} "$@" &
    _PIDS+=($!)
    _GPU_SLOT=$(( (_GPU_SLOT + 1) % NUM_GPUS ))
    sleep "${_COMPILE_STAGGER}" & wait $!
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

_cleanup() {
  echo ""
  echo "Caught signal — killing all background jobs..."
  if [ ${#_PIDS[@]} -gt 0 ]; then
    kill "${_PIDS[@]}" 2>/dev/null || true
  fi
  exit 1
}
trap _cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
VOCAB_SIZE=$(_cfg vocab_size)
NUM_LATENTS=$(_cfg num_latents)
HORIZON=$(_cfg horizon)
read -ra SEEDS <<< "$(_cfg seeds)"
LAM_ITERS=$(_cfg lam_iters)
DYNAMICS_ITERS=$(_cfg dynamics_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
BATCH_SIZE=$(_cfg batch_size)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_DIR=$(_cfg wandb_dir)

# ---------------------------------------------------------------------------
echo "===== [1/3] Pretrain LAM only — all seeds ====="
for S in "${SEEDS[@]}"; do
  LAM_CKPT="${CKPT_ROOT}/pretrain_lam_staged_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"
  if ! _done "${LAM_CKPT}"; then
    echo "  seed=${S}"
    _launch python3 -m scripts.train pretrain-lam-staged \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --ckpt-root ${CKPT_ROOT} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --model.num-latents ${NUM_LATENTS} \
      --model.horizon ${HORIZON} \
      --training.max-iters ${LAM_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      --wandb.dir ${WANDB_DIR}
  fi
done
_flush

# ---------------------------------------------------------------------------
echo "===== [2/3] Pretrain dynamics (frozen LAM) — all seeds ====="
for S in "${SEEDS[@]}"; do
  LAM_CKPT="${CKPT_ROOT}/pretrain_lam_staged_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"
  DYN_CKPT="${CKPT_ROOT}/pretrain_dynamics_staged_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"
  if ! _done "${DYN_CKPT}"; then
    echo "  seed=${S}"
    _launch python3 -m scripts.train pretrain-dynamics-staged \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --ckpt-root ${CKPT_ROOT} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --model.num-latents ${NUM_LATENTS} \
      --model.horizon ${HORIZON} \
      --lam-checkpoint-path "${LAM_CKPT}/best.pt" \
      --training.max-iters ${DYNAMICS_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      --wandb.dir ${WANDB_DIR}
  fi
done
_flush

# ---------------------------------------------------------------------------
echo "===== [3/3] Policy distillation — all seeds ====="
for S in "${SEEDS[@]}"; do
  DYN_CKPT="${CKPT_ROOT}/pretrain_dynamics_staged_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${S}"
  DISTILL_CKPT="${CKPT_ROOT}/policydistillation_staged_nlatents${NUM_LATENTS}_vocab${VOCAB_SIZE}_seed${S}"
  if ! _done "${DISTILL_CKPT}"; then
    echo "  seed=${S}"
    _launch python3 -m scripts.train policy-distill \
      --seed ${S} \
      --data-smiles ${DATA_DIR} \
      --ckpt-root ${CKPT_ROOT} \
      --tokenizer.vocab-size ${VOCAB_SIZE} \
      --controllable-gpt-path "${DYN_CKPT}/best.pt" \
      --model.num-latents ${NUM_LATENTS} \
      --training.max-iters ${DISTILLATION_ITERS} \
      --training.batch-size ${BATCH_SIZE} \
      --wandb.project ${WANDB_PROJECT} \
      --wandb.entity ${WANDB_ENTITY} \
      --wandb.group ${WANDB_GROUP} \
      --wandb.dir ${WANDB_DIR}
  fi
done
_flush

echo "===== Experiment 9 complete ====="
