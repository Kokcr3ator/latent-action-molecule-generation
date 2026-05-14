#!/usr/bin/env bash
# =============================================================================
# Experiment 6 — Impact of policy distillation
#
# Reuses pretrained ControllableGPT checkpoints from exp3.
# Runs two RL conditions:
#   nodistill — PolicyNetwork from scratch, straight to RL
#   distill   — policy distillation warm-start, then RL
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

EXP3_CKPT_ROOT=$(_cfg exp3_ckpt_root)
CKPT_ROOT=$(_cfg ckpt_root)
DATA_DIR=$(_cfg data_dir)
VOCAB_SIZE=$(_cfg vocab_size)
read -ra NUM_LATENTS_LIST <<< "$(_cfg num_latents_list)"
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
BATCH_SIZE=$(_cfg batch_size)
WANDB_GROUP=$(_cfg wandb_group)
WANDB_PROJECT=$(_cfg wandb_project)
WANDB_ENTITY=$(_cfg wandb_entity)
WANDB_DIR=$(_cfg wandb_dir)

# RL checkpoint roots for each condition
CKPT_ROOT_NODISTILL="${CKPT_ROOT}/exp6_nodistill"
CKPT_ROOT_DISTILL="${CKPT_ROOT}/exp6_distill"

# -----------------------------------------------------------------------------
echo "===== [1/3] Policy distillation — all seeds × all codebook sizes ====="
for S in "${SEEDS[@]}"; do
  echo "  === Seed ${S} ==="
  for N in "${NUM_LATENTS_LIST[@]}"; do
    echo "    num_latents=${N}"
    CTRL_CKPT="${EXP3_CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${VOCAB_SIZE}_seed${S}"
    if ! _done "${DISTILL_CKPT}"; then
      _launch python3 -m scripts.train policy-distill \
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
        --wandb.dir ${WANDB_DIR}
    fi
  done
done
_flush

# -----------------------------------------------------------------------------
echo "===== [2/3] RL without distillation — PolicyNetwork from scratch ====="
for S in "${SEEDS[@]}"; do
  echo "  === Seed ${S} ==="
  for N in "${NUM_LATENTS_LIST[@]}"; do
    CTRL_CKPT="${EXP3_CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "    num_latents=${N}, task=${TASK}"
      PPO_CKPT="${CKPT_ROOT_NODISTILL}/ppo_${TASK}_controllable_nlatents${N}_envs16_steps256_seed${S}"
      if ! _done "${PPO_CKPT}"; then
        _launch python3 -m scripts.train_ppo finetune-controllable \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --task ${TASK} \
          --tokenizer.vocab-size ${VOCAB_SIZE} \
          --controllable-gpt-path "${CTRL_CKPT}" \
          --num-latents ${N} \
          --ckpt-root ${CKPT_ROOT_NODISTILL} \
          --ppo.budget ${RL_BUDGET} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group "${WANDB_GROUP}_nodistill"
      fi
    done
  done
done
_flush

# -----------------------------------------------------------------------------
echo "===== [3/3] RL with distillation warm-start ====="
for S in "${SEEDS[@]}"; do
  echo "  === Seed ${S} ==="
  for N in "${NUM_LATENTS_LIST[@]}"; do
    CTRL_CKPT="${EXP3_CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${VOCAB_SIZE}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "    num_latents=${N}, task=${TASK}"
      PPO_CKPT="${CKPT_ROOT_DISTILL}/ppo_${TASK}_controllable_nlatents${N}_envs16_steps256_seed${S}"
      if ! _done "${PPO_CKPT}"; then
        _launch python3 -m scripts.train_ppo finetune-controllable \
          --seed ${S} \
          --data-smiles ${DATA_DIR} \
          --task ${TASK} \
          --tokenizer.vocab-size ${VOCAB_SIZE} \
          --pretrained-ckpt "${DISTILL_CKPT}" \
          --controllable-gpt-path "${CTRL_CKPT}" \
          --num-latents ${N} \
          --ckpt-root ${CKPT_ROOT_DISTILL} \
          --ppo.budget ${RL_BUDGET} \
          --wandb.project ${WANDB_PROJECT} \
          --wandb.entity ${WANDB_ENTITY} \
          --wandb.group "${WANDB_GROUP}_distill"
      fi
    done
  done
done
_flush

echo "===== Experiment 6 complete ====="
