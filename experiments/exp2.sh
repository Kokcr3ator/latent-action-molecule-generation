set -e

# We want to perform 2 experiments here:
# 1. Set vocabulary size to 4000, train controllableGPT with num_latents = 128 and finetune on 3 different properties
# 2. Set vocabulary size to 128  and finetune on 3 different properties

PRETRAIN_SEED=42
FINETUNE_SEEDS=(42 43 44)

VOCAB_SIZE1=4000
VOCAB_SIZE2=128
N_LATENTS=128

# Pretrain controllableGPT with vocab size 4000 and num_latents = 128
echo "Pretraining controllableGPT with vocab size ${VOCAB_SIZE1} and num_latents = ${N_LATENTS}"
python3 -m scripts.train \
--config configs/pretrain_controllable.yaml \
--override seed=${PRETRAIN_SEED} \
            model.num_latents=${N_LATENTS} \
            tokenizer.vocab_size=${VOCAB_SIZE1}


echo "Policy distillation with vocab size ${VOCAB_SIZE1} and num_latents = ${N_LATENTS}"
python3 -m scripts.train \
--config configs/policy_distillation.yaml \
--override seed=${PRETRAIN_SEED} \
            tokenizer.vocab_size=${VOCAB_SIZE1} \
            loader.controllable_gpt_path="ckpts/pretrain_controllable_vocab${VOCAB_SIZE1}_nlatent${N_LATENTS}_seed${PRETRAIN_SEED}/best.pt" \
            model.lm_head_out_size=${N_LATENTS}

# Pretrain with vocab size 128
echo "Pretraining with vocab size ${VOCAB_SIZE2}"
python3 -m scripts.train \
  --config configs/pretrain_base.yaml \
  --override seed=${PRETRAIN_SEED} \
             tokenizer.vocab_size=${VOCAB_SIZE2} 

BASE_CKPT_DIR2="ckpts/pretrain_base_vocab${VOCAB_SIZE2}_seed${PRETRAIN_SEED}"


CONTROLLABLE_CKPT_DIR="ckpts/pretrain_controllable_vocab${VOCAB_SIZE1}_nlatent${N_LATENTS}_seed${PRETRAIN_SEED}"
POLICY_DISTILLATION_CKPT_DIR="ckpts/policydistillation_nlatents${N_LATENTS}_vocab${VOCAB_SIZE1}_seed${PRETRAIN_SEED}"

echo "Finetuning base model with vocab size ${VOCAB_SIZE2} on logp"
# logp
for S in "${FINETUNE_SEEDS[@]}"; do
  echo "  -- seed=${S}"
  python3 -m scripts.train_ppo \
    --config configs/finetune_base.yaml \
    --override seed=${S} \
               reward.task=logp \
               tokenizer.vocab_size=${VOCAB_SIZE2} \
               ckpt.init_from=resume \
               ckpt.path="${BASE_CKPT_DIR2}" \
               ckpt.ckpt_name="best.pt" \
               experiment.wandb_run_name="ppo_base_logp_vocab${VOCAB_SIZE2}_seed${S}"
done

echo "Finetuning base model with vocab size ${VOCAB_SIZE2} on sa"
# sa
for S in "${FINETUNE_SEEDS[@]}"; do
  echo "  -- seed=${S}"
  python3 -m scripts.train_ppo \
    --config configs/finetune_base.yaml \
    --override seed=${S} \
               reward.task=sa \
               tokenizer.vocab_size=${VOCAB_SIZE2} \
               ckpt.init_from=resume \
               ckpt.path="${BASE_CKPT_DIR2}" \
               ckpt.ckpt_name="best.pt" \
               experiment.wandb_run_name="ppo_base_sa_vocab${VOCAB_SIZE2}_seed${S}"
done

echo "Finetuning base model with vocab size ${VOCAB_SIZE2} on qed"
# qed
for S in "${FINETUNE_SEEDS[@]}"; do
  echo "  -- seed=${S}"
  python3 -m scripts.train_ppo \
    --config configs/finetune_base.yaml \
    --override seed=${S} \
               reward.task=qed \
               tokenizer.vocab_size=${VOCAB_SIZE2} \
               ckpt.init_from=resume \
               ckpt.path="${BASE_CKPT_DIR2}" \
               ckpt.ckpt_name="best.pt" \
               experiment.wandb_run_name="ppo_base_qed_vocab${VOCAB_SIZE2}_seed${S}"
done

# controllableGPT finetuning
# logp
echo "Finetuning controllableGPT with vocab size ${VOCAB_SIZE1} and num_latents ${N_LATENTS} on logp"
for S in "${FINETUNE_SEEDS[@]}"; do
    echo "  -- seed=${S}"
    python3 -m scripts.train_ppo \
    --config configs/finetune_controllable.yaml \
    --override seed=${S} \
                reward.task=logp \
                tokenizer.vocab_size=${VOCAB_SIZE1} \
                ckpt.init_from=resume \
                ckpt.path="${POLICY_DISTILLATION_CKPT_DIR}" \
                ckpt.ckpt_name="best.pt" \
                loader.ckpt_controllable_path="${CONTROLLABLE_CKPT_DIR}" \
                loader.ckpt_name="best.pt" \
                experiment.wandb_run_name="ppo_controllable_logp_nlatents${N_LATENTS}_seed${S}"
done

# sa
echo "Finetuning controllableGPT with vocab size ${VOCAB_SIZE1} and num_latents ${N_LATENTS} on sa"
for S in "${FINETUNE_SEEDS[@]}"; do
    echo "  -- seed=${S}"
    python3 -m scripts.train_ppo \
    --config configs/finetune_controllable.yaml \
    --override seed=${S} \
                reward.task=sa \
                tokenizer.vocab_size=${VOCAB_SIZE1} \
                ckpt.init_from=resume \
                ckpt.path="${POLICY_DISTILLATION_CKPT_DIR}" \
                ckpt.ckpt_name="best.pt" \
                loader.ckpt_controllable_path="${CONTROLLABLE_CKPT_DIR}" \
                loader.ckpt_name="best.pt" \
                experiment.wandb_run_name="ppo_controllable_sa_nlatents${N_LATENTS}_seed${S}"
done

# qed
echo "Finetuning controllableGPT with vocab size ${VOCAB_SIZE1} and num_latents ${N_LATENTS} on qed"
for S in "${FINETUNE_SEEDS[@]}"; do
    echo "  -- seed=${S}"
    python3 -m scripts.train_ppo \
    --config configs/finetune_controllable.yaml \
    --override seed=${S} \
                reward.task=qed \
                tokenizer.vocab_size=${VOCAB_SIZE1} \
                ckpt.init_from=resume \
                ckpt.path="${POLICY_DISTILLATION_CKPT_DIR}" \
                ckpt.ckpt_name="best.pt" \
                loader.ckpt_controllable_path="${CONTROLLABLE_CKPT_DIR}" \
                loader.ckpt_name="best.pt" \
                experiment.wandb_run_name="ppo_controllable_qed_nlatents${N_LATENTS}_seed${S}"
done  


