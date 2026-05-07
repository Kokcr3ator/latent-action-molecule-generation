#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Codebook size ablation
#
# Research questions:
#   RQ2: Can we compress the action space without hurting RL performance?
#   RQ3: What is the impact of codebook size on generation quality (VUN)?
#
# Total runs: (3+3+3)×|seeds| + (3×5×|seeds|)×2 = 27 + 90 = 117
# =============================================================================
set -euo pipefail

ROOT="$(dirname "$0")/../.."
CFG="$(dirname "$0")/config.yaml"
cd "${ROOT}"

_cfg() { python3 -c "
import yaml
c = yaml.safe_load(open('${CFG}'))
v = c['$1']
print(' '.join(map(str, v)) if isinstance(v, list) else v)
"; }

DATA_DIR=$(_cfg data_dir)
CKPT_ROOT=$(_cfg ckpt_root)
read -ra VOCAB_SIZES       <<< "$(_cfg vocab_sizes)"
read -ra NUM_LATENTS_LIST  <<< "$(_cfg num_latents_list)"
CTRL_VOCAB_SIZE=$(_cfg ctrl_vocab_size)
read -ra SEEDS <<< "$(_cfg seeds)"
read -ra TASKS <<< "$(_cfg tasks)"
PRETRAIN_ITERS=$(_cfg pretrain_iters)
CONTROLLABLE_ITERS=$(_cfg controllable_iters)
DISTILLATION_ITERS=$(_cfg distillation_iters)
RL_BUDGET=$(_cfg rl_budget)
WANDB_GROUP=$(_cfg wandb_group)

# ---------------------------------------------------------------------------
echo "===== [1/5] Pretrain base GPT — all vocab sizes, all seeds ====="
for V in "${VOCAB_SIZES[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo "  vocab=${V}, seed=${S}"
    python3 -m scripts.train \
      --config scripts/pretrain_base/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 tokenizer.vocab_size=${V} \
                 training.max_iters=${PRETRAIN_ITERS} \
                 log.group=${WANDB_GROUP}
  done
done

# ---------------------------------------------------------------------------
echo "===== [2/5] Pretrain ControllableGPT — all codebook sizes, all seeds ====="
for N in "${NUM_LATENTS_LIST[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo "  num_latents=${N}, seed=${S}"
    python3 -m scripts.train \
      --config scripts/pretrain_controllable/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 model.num_latents=${N} \
                 tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                 training.max_iters=${CONTROLLABLE_ITERS} \
                 log.group=${WANDB_GROUP}
  done
done

# ---------------------------------------------------------------------------
echo "===== [3/5] Policy distillation — all codebook sizes, all seeds ====="
for N in "${NUM_LATENTS_LIST[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo "  num_latents=${N}, seed=${S}"
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${S}"
    python3 -m scripts.train \
      --config scripts/policy_distillation/config.yaml \
      --override seed=${S} \
                 data.smiles=${DATA_DIR} \
                 tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                 loader.controllable_gpt_path="${CTRL_CKPT}/best.pt" \
                 model.lm_head_out_size=${N} \
                 training.max_iters=${DISTILLATION_ITERS} \
                 log.group=${WANDB_GROUP}
  done
done

# ---------------------------------------------------------------------------
echo "===== [4/5] PPO finetune base GPT — all vocab sizes, tasks, seeds ====="
for V in "${VOCAB_SIZES[@]}"; do
  for S in "${SEEDS[@]}"; do
    BASE_CKPT="${CKPT_ROOT}/pretrain_base_vocab${V}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "  vocab=${V}, seed=${S}, task=${TASK}"
      python3 -m scripts.train_ppo \
        --config scripts/finetune_base/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   reward.task=${TASK} \
                   tokenizer.vocab_size=${V} \
                   ckpt.init_from=resume \
                   ckpt.path="${BASE_CKPT}" \
                   ckpt.ckpt_name="best.pt" \
                   ppo.budget=${RL_BUDGET} \
                   log.group=${WANDB_GROUP} \
                   experiment.wandb_run_name="ppo_${TASK}_base_vocab${V}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
    done
  done
done

# ---------------------------------------------------------------------------
echo "===== [5/5] PPO finetune ControllableGPT — all codebook sizes, tasks, seeds ====="
for N in "${NUM_LATENTS_LIST[@]}"; do
  for S in "${SEEDS[@]}"; do
    CTRL_CKPT="${CKPT_ROOT}/pretrain_controllable_vocab${CTRL_VOCAB_SIZE}_nlatent${N}_seed${S}"
    DISTILL_CKPT="${CKPT_ROOT}/policydistillation_nlatents${N}_vocab${CTRL_VOCAB_SIZE}_seed${S}"
    for TASK in "${TASKS[@]}"; do
      echo "  num_latents=${N}, seed=${S}, task=${TASK}"
      python3 -m scripts.train_ppo \
        --config scripts/finetune_controllable/config.yaml \
        --override seed=${S} \
                   data.smiles=${DATA_DIR} \
                   reward.task=${TASK} \
                   tokenizer.vocab_size=${CTRL_VOCAB_SIZE} \
                   ckpt.init_from=resume \
                   ckpt.path="${DISTILL_CKPT}" \
                   ckpt.ckpt_name="best.pt" \
                   loader.ckpt_controllable_path="${CTRL_CKPT}" \
                   loader.ckpt_name="best.pt" \
                   ppo.budget=${RL_BUDGET} \
                   log.group=${WANDB_GROUP} \
                   experiment.wandb_run_name="ppo_${TASK}_controllable_nlatents${N}_envs\${ppo.num_envs}_steps\${ppo.num_steps}_seed${S}"
    done
  done
done

echo "===== Experiment 2 complete ====="
