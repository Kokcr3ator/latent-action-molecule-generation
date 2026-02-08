## Plan: Log property satisfaction percentages at eval

Currently, `GPTTrainer` (pretrain_base) and `PretrainPolicyTrainer` (policy_distillation) log **raw mean** property values (qed, logp, mw, tpsa, sa) during evaluation. `RLTrainerBase` (finetune_base, finetune_controllable) logs only the optimized task's binary reward mean, but over **all** generated molecules (invalids included as 0). The goal is to log the **percentage of valid molecules satisfying the property threshold** — computed as `satisfied / valid_count`.

**Steps**

1. **Add helper function in [interdiff/metrics.py](interdiff/metrics.py)**

   Add a `property_satisfaction_rate(smiles_list, property_fn, **kwargs) -> float` function:
   - Filter to valid SMILES using `filter_valid_smiles`
   - For each valid SMILES, call `property_fn(smi, as_reward=True, **kwargs)` and check `>= 1.0` (handles QED's shaped reward where unsatisfied returns raw value < 0.7, not 0.0)
   - Return `satisfied_count / len(valid_smiles)`, or `0.0` if no valid molecules

   Also add a convenience function `all_property_satisfaction_rates(smiles_list) -> dict` that returns a dict `{qed_pct_satisfied, logp_pct_satisfied, mw_pct_satisfied, tpsa_pct_satisfied, sa_pct_satisfied}` by calling `property_satisfaction_rate` with each of the 5 property functions. This avoids duplicating the 5-property loop in multiple trainers.

2. **Update [interdiff/trainers/GPTTrainer.py](interdiff/trainers/GPTTrainer.py) `evaluate()`** (pretrain_base)

   - Remove the individual raw property computation blocks ([lines 34–49](interdiff/trainers/GPTTrainer.py#L34-L49): `qed_scores`, `sa_scores`, `logp_scores_list`, `mw_list`, `tpsa_list` and their means)
   - Import the new `all_property_satisfaction_rates` from `interdiff.metrics` (replace the current individual metric imports for `qed`, `synthetic_accessibility`, `logp`, `molecular_weight`, `tpsa`)
   - After generating SMILES, call `pct_metrics = all_property_satisfaction_rates(generated_smiles)`
   - In the return dict ([lines 73–78](interdiff/trainers/GPTTrainer.py#L73-L78)), replace `qed`, `sa`, `logp`, `mw`, `tpsa` keys with `**pct_metrics` (which unpacks `qed_pct_satisfied`, `sa_pct_satisfied`, etc.)

3. **Update [interdiff/trainers/PretrainPolicyTrainer.py](interdiff/trainers/PretrainPolicyTrainer.py) `evaluate()`** (policy_distillation)

   - Same changes as step 2 — the code is nearly identical ([lines 40–55](interdiff/trainers/PretrainPolicyTrainer.py#L40-L55) for raw scores, [lines 79–84](interdiff/trainers/PretrainPolicyTrainer.py#L79-L84) for return dict)

4. **Update [interdiff/trainers/base_RL.py](interdiff/trainers/base_RL.py) `evaluate()`** (finetune_base & finetune_controllable)

   - Keep the existing `eval/<task>` metric as-is (mean of `as_reward=True` over all generated SMILES, including invalid as 0)
   - Import `property_satisfaction_rate` and the individual property function for the task (or map from task name)
   - After the existing `task_scores` computation ([line 295](interdiff/trainers/base_RL.py#L295)), compute the new metric:
     - Map `task` name to the corresponding property function (e.g., `"qed"` → `qed`, `"sa"` → `synthetic_accessibility`, etc.)
     - Call `property_satisfaction_rate(generated_smiles, property_fn)`
   - Add `f"{task}_pct_satisfied": <result>` to the return dict ([lines 307–314](interdiff/trainers/base_RL.py#L307-L314))

**Verification**

- Run `pretrain_base` for a few eval steps and confirm wandb logs show `qed_pct_satisfied`, `logp_pct_satisfied`, `mw_pct_satisfied`, `tpsa_pct_satisfied`, `sa_pct_satisfied` instead of the old raw `qed`, `logp`, `mw`, `tpsa`, `sa`
- Run `policy_distillation` similarly and confirm same new keys appear
- Run `finetune_base` (with e.g. `reward.task=qed`) and confirm both `eval/qed` (existing, all-generated denominator) and `qed_pct_satisfied` (valid-only denominator) appear in wandb
- Unit test: compute `property_satisfaction_rate` on a known set of SMILES and verify the result matches manual calculation

**Decisions**

- Naming: `<property>_pct_satisfied` (e.g. `qed_pct_satisfied`)
- Denominator: satisfied / valid (invalid molecules excluded)
- pretrain_base & policy_distillation: raw means **replaced** by percentages (not kept alongside)
- Finetuning: existing `eval/<task>` kept, new `<task>_pct_satisfied` added (valid-only denominator), only for the optimized task (not all 5)
- `>= 1.0` check used as the threshold test — works for all 5 properties including QED (whose `as_reward=True` returns raw QED < 0.7 on failure, never reaching 1.0)
