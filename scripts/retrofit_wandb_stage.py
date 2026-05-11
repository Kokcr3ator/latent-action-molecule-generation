"""
Backfill the 'stage' and 'method' config fields on existing W&B runs by
inferring them from the run name.

Name patterns:
  pretrain_base_*              -> stage=pretrain   method=gpt
  pretrain_controllable_*      -> stage=pretrain   method=cgpt
  policydistillation_*         -> stage=distillation  method=cgpt
  ppo_*_controllable_*         -> stage=rl         method=cgpt
  ppo_*                        -> stage=rl         method=gpt

Usage:
    python -m scripts.retrofit_wandb_stage --entity latent-action-interdiff --project interdiff
    python -m scripts.retrofit_wandb_stage ... --dry-run
"""
import argparse
import wandb


def _infer_fields(run_name: str) -> tuple[str, str] | None:
    """Return (stage, method) or None if unrecognised."""
    if run_name.startswith("pretrain_base_"):
        return "pretrain", "gpt"
    if run_name.startswith("pretrain_controllable_"):
        return "pretrain", "cgpt"
    if run_name.startswith("policydistillation_"):
        return "distillation", "cgpt"
    if run_name.startswith("ppo_") and "_controllable_" in run_name:
        return "rl", "cgpt"
    if run_name.startswith("ppo_"):
        return "rl", "gpt"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    patched = skipped = unrecognised = 0
    for run in runs:
        if run.config.get("stage") and run.config.get("method"):
            skipped += 1
            continue

        fields = _infer_fields(run.name)
        if fields is None:
            print(f"  unrecognised: {run.name}")
            unrecognised += 1
            continue

        stage, method = fields
        print(f"{'[dry-run] ' if args.dry_run else ''}patching {run.name} -> stage={stage} method={method}")
        if not args.dry_run:
            run.config["stage"] = stage
            run.config["method"] = method
            run.update()
        patched += 1

    print(f"\nPatched: {patched}  |  Already set: {skipped}  |  Unrecognised: {unrecognised}")


if __name__ == "__main__":
    main()
