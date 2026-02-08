## Plan: Log Hyperparameters to wandb

All 5 experiment configs use OmegaConf-loaded YAML files, but `wandb.init()` in `WandbLogger` is called with only `project`, `name`, and `group` — **no hyperparameters are logged**. The fix adds a `log_config()` method to `WandbLogger` and calls it from both training scripts right after logger creation, passing the full resolved config as a nested dict.

**Steps**

1. **Add `log_config()` method to `WandbLogger`** in [interdiff/logging/wandb_logger.py](interdiff/logging/wandb_logger.py)
   - Add a method `log_config(self, config: dict)` that calls `wandb.config.update(config)`.
   - This keeps the constructor signature untouched (still instantiated via `instantiate(cfg.log)`), avoiding coupling between the `log:` YAML section and the full config.

2. **Add `log_config()` as a no-op on the base `Logger`** in [interdiff/logging/logger.py](interdiff/logging/logger.py)
   - Add `def log_config(self, config: dict): pass` so callers don't need `isinstance` checks — the interface stays uniform.

3. **Call `log_config()` from `scripts/train.py`** (covers pretrain_base, pretrain_controllable, policy_distillation)
   - After [line ~49](scripts/train.py#L49) where `logger` is created, convert the config: `OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)` and call `logger.log_config(cfg_dict)`.
   - Guard with the existing `if logger is not None` check.

4. **Call `log_config()` from `scripts/train_ppo.py`** (covers finetune_base, finetune_controllable)
   - Same pattern after [line ~109](scripts/train_ppo.py#L109): resolve config to nested dict, call `logger.log_config(cfg_dict)`.

5. **Set `wandb_log: true` in all 5 YAML configs**
   - [configs/pretrain_base.yaml](configs/pretrain_base.yaml) — change `wandb_log: false` → `true`
   - [configs/pretrain_controllable.yaml](configs/pretrain_controllable.yaml) — change `wandb_log: false` → `true`
   - [configs/policy_distillation.yaml](configs/policy_distillation.yaml) — change `wandb_log: false` → `true`
   - [configs/finetune_base.yaml](configs/finetune_base.yaml) and [configs/finetune_controllable.yaml](configs/finetune_controllable.yaml) — already `true`, no change needed.

6. **Handle OmegaConf `MISSING` values** — Some configs have `???` placeholders (e.g., `loader.controllable_gpt_path` in policy_distillation). Using `OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)` will convert these to `None` instead of raising, so they appear as `null` in the wandb config panel — acceptable behavior.

**Verification**

- Run one experiment (e.g., `python3 -m scripts.train --config configs/pretrain_base.yaml`) and check the wandb run's **Config** tab — it should show the full nested hyperparameter tree (model, training, optimizer, scheduler, etc.).
- Confirm PPO experiments also log hyperparameters by running `python3 -m scripts.train_ppo --config configs/finetune_base.yaml` and inspecting the wandb Config panel.
- Verify that CLI overrides (e.g., `--override model.n_layer=8`) are reflected in the logged config, since `merge_with_overrides` is called before logger creation.

**Decisions**
- **`log_config()` method over modifying `__init__`**: The `WandbLogger` is instantiated via `instantiate(cfg.log)`, which only passes fields from the `log:` YAML section. Adding a separate method avoids mixing full-config data into that section.
- **Nested dict**: Preserves YAML hierarchy in the wandb config panel as requested.
- **`throw_on_missing=False`**: Avoids crashes on `???` placeholders while still logging everything else.
