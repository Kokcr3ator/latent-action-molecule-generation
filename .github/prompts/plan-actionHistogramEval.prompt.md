## Plan: Log Action‑Index Histogram in Eval

During `pretrain_controllable` evaluation, collect all VQ codebook indices across eval batches and log a single aggregated histogram to wandb. This requires threading the indices through the existing return path (they are currently discarded) and accumulating them in `evaluate()`.

**Steps**

1. **Add indices to `vq_loss_dict` in `VectorQuantizer.forward()`** — In [modules.py](interdiff/modules.py#L453), add `'indices': indices` to the `vq_loss_dict` dictionary at [line 459](interdiff/modules.py#L459) (alongside the existing `entropy`, `q_loss`, etc.).

2. **Propagate indices through `LatentActionModel.forward()`** — In [models.py](interdiff/models.py#L362), change `actions, vq_loss_dict, _ = self.vq_encode(tokens)` to `actions, vq_loss_dict, indices = self.vq_encode(tokens)` and either let the indices ride in `vq_loss_dict` (added in step 1) or explicitly keep them. Since they are already in the dict from step 1, no further change is needed here — just stop discarding the third return value if you want, but the dict alone suffices.

3. **Surface indices in `ControllableGPTTrainer.forward_loss_with_components()`** — In [ControllableGPTTrainer.py](interdiff/trainers/ControllableGPTTrainer.py#L86), the method already receives `vq_loss_dict` from `self.model(x)`. Add `'vq_indices': vq_loss_dict['indices']` to the returned dict (after [line 109](interdiff/trainers/ControllableGPTTrainer.py#L109)).

4. **Accumulate indices in `evaluate()`** — In [ControllableGPTTrainer.py](interdiff/trainers/ControllableGPTTrainer.py#L193), inside the eval loop:
   - Before the loop, create `all_indices = []`.
   - Inside the per-batch loop (after [line 214](interdiff/trainers/ControllableGPTTrainer.py#L214)), append `loss_dict['vq_indices'].detach().cpu()` to `all_indices`.
   - After the loop, concatenate: `all_indices = torch.cat(all_indices).view(-1)`.

5. **Log the histogram** — After computing `result` dict and before `return result`, add `result['val/action_histogram'] = wandb.Histogram(all_indices.numpy(), num_bins=self.model.lam.vq.num_latents)`. Add `import wandb` at the top of [ControllableGPTTrainer.py](interdiff/trainers/ControllableGPTTrainer.py).

6. **Guard non-wandb runs** — Wrap the histogram assignment in `if self.logger is not None:` (or check `isinstance(self.logger, WandbLogger)`) so runs without wandb don't crash.

**Verification**

- Run `pretrain_controllable` for a few steps with eval enabled and confirm in the wandb dashboard that `val/action_histogram` appears as a histogram panel.
- Verify the histogram has `num_latents` (128) bins and the counts sum to `eval_iters × batch_size × (seq_len - 1)` (5 × 512 × 127 = 325,120).
- Verify no regressions: scalar metrics (`val_vq_entropy`, `val_loss`, etc.) remain unchanged.

**Decisions**

- **Aggregated histogram**: one histogram across all eval batches per eval round (not per-batch).
- **Propagation via `vq_loss_dict`**: indices added to the existing dict to avoid changing function signatures.
