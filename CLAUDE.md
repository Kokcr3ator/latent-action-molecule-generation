# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
pip install -e .
```

## Commands

```bash
# Stage 1 – pretrain base GPT on SMILES
python scripts/train.py --config configs/pretrain_base.yaml

# Stage 2 – pretrain ControllableGPT (LAM + DynamicsModel)
python scripts/train.py --config configs/pretrain_controllable.yaml

# Stage 3 – policy distillation (train PolicyNetwork to imitate LAM codes)
python scripts/train.py --config configs/policy_distillation.yaml

# Stage 4a – PPO fine-tune base GPT (baseline)
python scripts/train_ppo.py --config configs/finetune_base.yaml

# Stage 4b – PPO fine-tune controllable policy (method)
python scripts/train_ppo.py --config configs/finetune_controllable.yaml

# Override any config value at the CLI
python scripts/train.py --config configs/pretrain_base.yaml --override training.batch_size=256 seed=123
python scripts/train_ppo.py --config configs/finetune_base.yaml --override reward.task=logp ppo.num_envs=64

# Run tests
pytest test/
```

## Architecture

The package is called `interdiff` (legacy name). All source lives under `interdiff/`.

### Training pipeline

The project has a four-stage training pipeline. Stages 1–3 use `scripts/train.py`; stage 4 uses `scripts/train_ppo.py`.

1. **Pretrain base GPT** (`pretrain_base.yaml`) — standard autoregressive next-token prediction on ZINC SMILES. Produces the baseline model.
2. **Pretrain ControllableGPT** (`pretrain_controllable.yaml`) — trains a `LatentActionModel` (VQ-VAE) jointly with a `DynamicsModel`. The LAM encoder maps transitions $(s_t, s_{t+1})$ to discrete latent codes; the decoder + dynamics model reconstruct $s_{t+1}$ from $(s_t, z)$.
3. **Policy distillation** (`policy_distillation.yaml`) — trains a `PolicyNetwork` (a `GPT` with `lm_head_out_size = num_latents`) via imitation learning: predict the latent code the LAM assigns to the observed transition.
4. **PPO fine-tuning** — `finetune_base.yaml` fine-tunes `GPT` with token-level actions (baseline); `finetune_controllable.yaml` fine-tunes `PolicyNetwork` with latent-code actions via `ControllableMoleculeGenerationEnv`.

### Models (`interdiff/models.py`)

| Class | Role |
|---|---|
| `GPT` | Shared transformer backbone (nanoGPT-style). `forward(idx)` → `(logits, hidden_states)`. |
| `LatentActionModel` | VQ-VAE: `vq_encode(tokens)` → `(z_q, vq_loss_dict, indices)`; `decode(tokens, actions)` → logits. The encoder reads the *full* sequence; the action at position $t$ is the hidden state at $t+1$ (future-conditioned). |
| `DynamicsModel` | `forward(tokens, actions)` → logits. Adds action embeddings to token embeddings before the LM head; no separate attention over actions. |
| `ControllableGPT` | Wraps `LatentActionModel` + `DynamicsModel`. In `forward`, the LAM encodes actions then they are `.detach()`-ed before passing to the dynamics model, so gradients do not flow back from the dynamics loss into the LAM encoder. |
| `PolicyNetwork` | Subclass of `GPT` with `lm_head_out_size = num_latents`. Overrides `generate` to call the `DynamicsModel` for each step rather than directly appending its own token predictions. |

All models inherit `SerialisableModule` (from `interdiff/modules.py`) which provides `.save()` / `.load()` over a `{"model_config": ..., "model_state_dict": ...}` checkpoint format.

### Config system (`interdiff/config.py`)

Uses OmegaConf (not Hydra, despite the `conf_hydra_deprecated/` directory). Every config node with a `_target_` key is instantiated via `interdiff.config.instantiate(cfg)`, which does a lazy `importlib` import. OmegaConf `${...}` interpolations are used heavily to share values (e.g., `vocab_size` appears once and is referenced everywhere). CLI overrides use dotlist syntax: `key=value` or `nested.key=value`.

### Environments (`interdiff/envs.py`)

Both environments extend the base `Env` dataclass and implement auto-reset (done envs are reset at the start of the next `step` call).

- `MoleculeGenerationEnv` — actions are token IDs; next state appends the action token directly.
- `ControllableMoleculeGenerationEnv` — actions are latent code indices; `_get_next_state` looks up the codebook embedding and calls `DynamicsModel.forward` to sample the next token. Holds references to `lam` and `dynamics_model`.

Reward is zero at every step except when EOS is emitted, where RDKit evaluates the full SMILES string. Invalid SMILES returns 0.

### Data

Raw data: `interdiff/data/zinc/zinc.txt` (249,455 SMILES, one per line).  
`scripts/tokenise_dataset.py` trains a BPE tokenizer and saves a `.safetensors` file under `interdiff/data/processed/`. This is run automatically at the start of every training script. Processed tokenizers for vocab sizes 50, 500, and 5000 are already committed.

### Logging

Set `wandb_log: true` in the config (default) to log to W&B project `interdiff`. Set to `false` to disable.
