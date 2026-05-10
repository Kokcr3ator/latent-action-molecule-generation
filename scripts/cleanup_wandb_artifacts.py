"""
Delete all but the latest version of every artifact in a W&B project.
Usage:
    python -m scripts.cleanup_wandb_artifacts --entity latent-action-interdiff --project interdiff
"""
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--type", default="model", help="Artifact type to clean (default: model)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    api = wandb.Api()
    collections = api.artifact_type(args.type, f"{args.entity}/{args.project}").collections()

    total_deleted = 0
    for collection in collections:
        versions = sorted(collection.versions(), key=lambda a: a.version)
        stale = versions[:-1]  # keep the last one
        for artifact in stale:
            size_mb = artifact.size / 1e6
            print(f"{'[dry-run] ' if args.dry_run else ''}deleting {artifact.name} ({size_mb:.1f} MB)")
            if not args.dry_run:
                artifact.delete(delete_aliases=True)
            total_deleted += 1

    print(f"\n{'Would delete' if args.dry_run else 'Deleted'} {total_deleted} artifact version(s).")


if __name__ == "__main__":
    main()
