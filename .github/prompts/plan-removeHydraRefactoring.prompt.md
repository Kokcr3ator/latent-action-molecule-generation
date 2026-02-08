## Plan: Remove Hydra, Use Self-Contained Experiment YAMLs

**TL;DR**: Replace Hydra with simple YAML loading via OmegaConf. Create 5 complete, readable experiment config files (one per experiment type). Add a config utility module with a custom `instantiate()` function. Refactor `train.py` and `train_ppo.py` to accept a config path via argparse. This eliminates config composition complexity while preserving interpolation support and making hyperparameter tuning straightforward.

**Steps**

1. **Create config utility module** at [interdiff/config.py](interdiff/config.py)
   - Function `load_config(path: str)` → loads YAML with OmegaConf, resolves interpolations
   - Function `instantiate(cfg)` → custom implementation that reads `_target_` and creates objects (maps to: `GPT`, `ControllableGPT`, `PolicyNetwork`, `GPTTrainer`, etc.)
   - Function `merge_with_overrides(cfg, overrides: dict)` → for HP tuning integration
   - Registry dict mapping `_target_` strings to actual classes

2. **Create `configs/` directory** at project root with 5 experiment YAMLs:
   - `configs/pretrain_base.yaml` — all settings for base GPT pretraining
   - `configs/pretrain_controllable.yaml` — all settings for ControllableGPT pretraining
   - `configs/policy_distillation.yaml` — all settings for PolicyNetwork distillation (includes `controllable_gpt_path`)
   - `configs/finetune_base.yaml` — all PPO settings for base GPT finetuning (includes pretrained checkpoint path)
   - `configs/finetune_controllable.yaml` — all PPO settings for controllable finetuning (includes both checkpoint paths)

3. **Structure each experiment YAML** with clear sections:
   ```yaml
   # ===== EXPERIMENT =====
   experiment_name: pretrain_base
   wandb_project: molecule-generation
   seed: 42
   
   # ===== MODEL =====
   model:
     _target_: interdiff.models.GPT
     vocab_size: 500
     n_embd: 256
     ...
   
   # ===== TRAINING =====
   training:
     batch_size: 512
     max_iters: 100000
     ...
   
   # ===== OPTIMIZER =====
   optimizer:
     lr: 1e-4
     ...
   ```

4. **Refactor [scripts/train.py](scripts/train.py)**:
   - Remove `@hydra.main` decorator and Hydra imports
   - Add argparse with `--config` argument (path to YAML)
   - Add `--override` argument for CLI overrides (e.g., `--override training.batch_size=256`)
   - Replace `instantiate(cfg.model)` calls with custom `instantiate()` from [interdiff/config.py](interdiff/config.py)
   - Remove `get_original_cwd()` usage (no longer needed)

5. **Refactor [scripts/train_ppo.py](scripts/train_ppo.py)**:
   - Same changes as train.py
   - Keep RL-specific logic (reference model creation, env setup)
   - Replace all 4 `instantiate()` calls with custom implementation

6. **Update [interdiff/data/RLLoader.py](interdiff/data/RLLoader.py#L42)**:
   - Replace `instantiate(cfg.ppo)` with direct `HParams(**cfg.ppo)` or custom instantiate

7. **Preserve interpolation** by using OmegaConf resolvers:
   - Register custom resolvers if needed (e.g., for computed values)
   - Move commonly interpolated values to top-level keys for clarity

8. **Update experiment run scripts** in `experiments/*/run.ssh`:
   - Change from `python scripts/train.py model=base_gpt train_cfg.batch_size=512`
   - To: `python scripts/train.py --config configs/pretrain_base.yaml --override training.batch_size=512`

9. **Archive old Hydra configs** (optional):
   - Move `interdiff/conf/` to `interdiff/conf_hydra_deprecated/` or delete
   - Keep as reference during migration

10. **Update [scripts/tokenise_dataset.py](scripts/tokenise_dataset.py)** if it uses Hydra (check and adapt)

**Verification**

- Run each experiment type with its new config and verify training starts correctly
- Compare a few training iterations between old Hydra config and new YAML config for identical behavior
- Test CLI overrides: `python scripts/train.py --config configs/pretrain_base.yaml --override training.lr=5e-5`
- Verify interpolations resolve correctly (e.g., `vocab_size` appears in both model and loader sections)

**Decisions**

- Keep `_target_` pattern for instantiation: maintains familiarity, allows future HP tuning tools to understand class types
- Use OmegaConf (not pure PyYAML): preserves interpolation syntax `${...}` which is already used in config values
- Single YAML per experiment (no inheritance): maximizes readability, user sees ALL parameters in one file
- argparse over click/typer: minimal dependencies, standard library
