#!/usr/bin/env bash
# =============================================================================
# Experiment 1 — Main comparison: base GPT vs ControllableGPT
#
# Research questions:
#   RQ1: Does a latent action model help RL fine-tuning?
#   RQ5: Does the model generalise across different molecular properties?
#
# Total runs: 1 + 1 + 1 + (5×3) + (5×3) = 33
# =============================================================================
set -euo pipefail

ROOT="$(dirname "$0")/../.."
CFG="$(dirname "$0")/config.yaml"
cd "${ROOT}"

# Read experiment config
_cfg() { python3 -c "
import yaml
c = yaml.safe_load(open('${CFG}'))
v = c['$1']
print(' '.join(map(str, v)) if isinstance(v, list) else v)
"; }

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
VOCAB_SIZE=$(_cfg vocab_size)
NUM_LATENTS=$(_cfg num_latents)
PRETRAIN_SEED=$(_cfg pretrain_seed)
read -ra RL_SEEDS <<< "$(_cfg rl_seeds)"
read -ra TASKS    <<< "$(_cfg tasks)"
PRETRAIN_ITERS=$(_cfg pretrain_iters)
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
WANDB_GROUP=$(_cfg wandb_group)

BASE_CKPT_DIR="${CKPT_ROOT}/pretrain_base_vocab${VOCAB_SIZE}_seed${PRETRAIN_SEED}"
CTRL_CKPT_DIR="${CKPT_ROOT}/pretrain_controllable_vocab${VOCAB_SIZE}_nlatent${NUM_LATENTS}_seed${PRETRAIN_SEED}"
DISTILL_CKPT_DIR="${CKPT_ROOT}/policydistillation_nlatents${NUM_LATENTS}_vocab${VOCAB_SIZE}_seed${PRETRAIN_SEED}"

# ---------------------------------------------------------------------------
echo "===== [1/5] Pretrain base GPT (vocab=${VOCAB_SIZE}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config scripts/pretrain_base/config.yaml \
  --override seed=${PRETRAIN_SEED} \
             data.smiles=${DATA_DIR} \
             tokenizer.vocab_size=${VOCAB_SIZE} \
             training.max_iters=${PRETRAIN_ITERS} \
             log.group=${WANDB_GROUP}

# ---------------------------------------------------------------------------
echo "===== [2/5] Pretrain ControllableGPT (num_latents=${NUM_LATENTS}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config scripts/pretrain_controllable/config.yaml \
  --override seed=${PRETRAIN_SEED} \
             data.smiles=${DATA_DIR} \
             model.num_latents=${NUM_LATENTS} \
             tokenizer.vocab_size=${VOCAB_SIZE} \
             training.max_iters=${CONTROLLABLE_ITERS} \
             log.group=${WANDB_GROUP}

# ---------------------------------------------------------------------------
echo "===== [3/5] Policy distillation (num_latents=${NUM_LATENTS}, seed=${PRETRAIN_SEED}) ====="
python3 -m scripts.train \
  --config scripts/policy_distillation/config.yaml \
  --override seed=${PRETRAIN_SEED} \
             data.smiles=${DATA_DIR} \
             tokenizer.vocab_size=${VOCAB_SIZE} \
             loader.controllable_gpt_path="${CTRL_CKPT_DIR}/best.pt" \
             model.lm_head_out_size=${NUM_LATENTS} \
             training.max_iters=${DISTILLATION_ITERS} \
             log.group=${WANDB_GROUP}

# ---------------------------------------------------------------------------
echo "===== [4/5] PPO finetune base GPT — all tasks, all seeds ====="
for TASK in "${TASKS[@]}"; do
  for S in "${RL_SEEDS[@]}"; do
    echo "  task=${TASK}, seed=${S}"
    python3 -m scripts.train_ppo \
      --config scripts/finetune_base/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 reward.task=${TASK} \
                 tokenizer.vocab_size=${VOCAB_SIZE} \
                 ckpt.init_from=resume \
                 ckpt.path="${BASE_CKPT_DIR}" \
                 ckpt.ckpt_name="best.pt" \
                 ppo.budget=${RL_BUDGET} \
                 log.group=${WANDB_GROUP}
  done
done

# ---------------------------------------------------------------------------
echo "===== [5/5] PPO finetune ControllableGPT — all tasks, all seeds ====="
for TASK in "${TASKS[@]}"; do
  for S in "${RL_SEEDS[@]}"; do
    echo "  task=${TASK}, seed=${S}"
    python3 -m scripts.train_ppo \
      --config scripts/finetune_controllable/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 reward.task=${TASK} \
                 tokenizer.vocab_size=${VOCAB_SIZE} \
                 ckpt.init_from=resume \
                 ckpt.path="${DISTILL_CKPT_DIR}" \
                 ckpt.ckpt_name="best.pt" \
                 loader.ckpt_controllable_path="${CTRL_CKPT_DIR}" \
                 loader.ckpt_name="best.pt" \
                 ppo.budget=${RL_BUDGET} \
                 log.group=${WANDB_GROUP} \
                 experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${NUM_LATENTS}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
  done
done

echo "===== Experiment 1 complete ====="
