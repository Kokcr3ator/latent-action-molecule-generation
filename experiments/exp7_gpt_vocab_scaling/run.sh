#!/usr/bin/env bash
# =============================================================================
# Experiment 7 — GPT baseline scaling with vocab size
#
# Research question:
#   RQ7: How does a vanilla GPT baseline perform when its action space
#        matches the codebook sizes from exp3?
#
# Runs all pretrain seeds, then all RL seeds.
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
_COMPILE_STAGGER=90

_launch() {
  if [ "${NUM_GPUS}" -ge 2 ]; then
    if [ ${#_PIDS[@]} -ge "${MAX_JOBS}" ]; then
      wait -n
      local alive=()
      for pid in "${_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive+=("$pid")
      done
      _PIDS=("${alive[@]+"${alive[@]}"}")
    fi
    local gpu_idx=$(( _GPU_SLOT % NUM_GPUS ))
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$gpu_idx]} "$@" &
    _PIDS+=($!)
    _GPU_SLOT=$(( _GPU_SLOT + 1 ))
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
read -ra VOCAB_SIZES_LIST <<< "$(_cfg vocab_sizes_list)"
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
PRETRAIN_ITERS=$(_cfg pretrain_iters)
RL_BUDGET=$(_cfg rl_budget)
BATCH_SIZE=$(_cfg batch_size)
JOBS_PER_GPU=$(_cfg jobs_per_gpu)
MAX_JOBS=$(( NUM_GPUS * JOBS_PER_GPU ))
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_DIR=$(_cfg wandb_dir)
USE_LARGE_GPT=$(_cfg use_large_gpt)
LARGE_N_LAYER=$(_cfg large_gpt_n_layer)
LARGE_N_HEAD=$(_cfg large_gpt_n_head)
LARGE_N_EMBD=$(_cfg large_gpt_n_embd)

# Helper: run one pretrain + PPO sweep for a given model size tag and extra model flags
_run_sweep() {
  local TAG="$1"; shift        # e.g. "base" or "large"
  local MODEL_FLAGS=("$@")    # extra --model.* flags (empty for small GPT)

  echo "===== [pretrain:${TAG}] Pretrain GPT (${TAG}) — all seeds × all vocab sizes ====="
  for S in "${SEEDS[@]}"; do
    echo "  === Seed ${S} ==="
    for V in "${VOCAB_SIZES_LIST[@]}"; do
      echo "    vocab_size=${V}"
      CKPT="${CKPT_ROOT}/pretrain_${TAG}_vocab${V}_seed${S}"
      if ! _done "${CKPT}"; then
        _launch python3 -m scripts.train pretrain-base \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --tokenizer.vocab-size ${V} \
          --training.max-iters ${PRETRAIN_ITERS} \
          --training.batch-size ${BATCH_SIZE} \
          --ckpt-root ${CKPT_ROOT} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group ${WANDB_GROUP} \
          --wandb.dir ${WANDB_DIR} \
          "${MODEL_FLAGS[@]}"
      fi
    done
  done
  _flush

  echo "===== [ppo:${TAG}] PPO finetune — all seeds × all vocab sizes × all tasks ====="
  for S in "${SEEDS[@]}"; do
    echo "  === Seed ${S} ==="
    for V in "${VOCAB_SIZES_LIST[@]}"; do
      CKPT="${CKPT_ROOT}/pretrain_${TAG}_vocab${V}_seed${S}"
      for TASK in "${TASKS[@]}"; do
        echo "    vocab_size=${V}, task=${TASK}"
        PPO_CKPT="${CKPT_ROOT}/ppo_${TASK}_${TAG}_vocab${V}_envs16_steps256_seed${S}"
        if ! _done "${PPO_CKPT}"; then
          _launch python3 -m scripts.train_ppo finetune-base \
            --seed ${S} \
            --data-smiles ${DATA_DIR} \
            --task ${TASK} \
            --tokenizer.vocab-size ${V} \
            --pretrained-ckpt "${CKPT}" \
            --ckpt-root ${CKPT_ROOT} \
            --ppo.budget ${RL_BUDGET} \
            --wandb.project ${WANDB_PROJECT} \
            --wandb.entity ${WANDB_ENTITY} \
            --wandb.group ${WANDB_GROUP}
        fi
      done
    done
  done
  _flush
}

# -----------------------------------------------------------------------------
_run_sweep "base"

if [ "${USE_LARGE_GPT}" = "true" ]; then
  _run_sweep "large" \
    --model.n-layer ${LARGE_N_LAYER} \
    --model.n-head  ${LARGE_N_HEAD} \
    --model.n-embd  ${LARGE_N_EMBD}
fi

echo "===== Experiment 7 complete ====="
