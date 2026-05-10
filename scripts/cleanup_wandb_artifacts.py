"""
Delete all but the latest artifact version for runs belonging to a W&B group.
Scoping to a group avoids touching artifacts from unrelated projects or collaborators.

Usage:
    python -m scripts.cleanup_wandb_artifacts --entity latent-action-interdiff --project interdiff --group exp3_lam_scaling
    python -m scripts.cleanup_wandb_artifacts ... --dry-run   # preview only
"""
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True, help="W&B run group to scope cleanup to")
    parser.add_argument("--type", default="model", help="Artifact type (default: model)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    api = wandb.Api()

    # Collect artifact names produced by runs in the target group
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
    artifact_names = set()
    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type == args.type:
                artifact_names.add(artifact.name.rsplit(":", 1)[0])  # strip :vN suffix

    if not artifact_names:
        print(f"No {args.type} artifacts found for group '{args.group}'.")
        return

    print(f"Found {len(artifact_names)} artifact collection(s) in group '{args.group}'.")

    total_deleted = 0
    for name in sorted(artifact_names):
        collection = api.artifact_versions(args.type, f"{args.entity}/{args.project}/{name}")
        versions = sorted(collection, key=lambda a: a.version)
        stale = versions[:-1]  # keep only the latest
        for artifact in stale:
            size_mb = artifact.size / 1e6
            print(f"{'[dry-run] ' if args.dry_run else ''}deleting {artifact.name} ({size_mb:.1f} MB)")
            if not args.dry_run:
                artifact.delete(delete_aliases=True)
            total_deleted += 1

    print(f"\n{'Would delete' if args.dry_run else 'Deleted'} {total_deleted} artifact version(s).")


if __name__ == "__main__":
    main()
