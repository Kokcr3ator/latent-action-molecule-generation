"""
Delete all but the latest artifact version for runs belonging to a W&B group.
Scoping to a group avoids touching artifacts from unrelated projects or collaborators.

Usage:
    python -m scripts.cleanup_wandb_artifacts --entity latent-action-interdiff --project interdiff --group exp3_lam_scaling
    python -m scripts.cleanup_wandb_artifacts ... --dry-run   # preview only
"""
import argparse
import wandb


def _version_int(artifact) -> int:
    return int(artifact.version.lstrip("v"))


def _delete(artifact, dry_run: bool) -> bool:
    """Strip aliases then delete. Returns True on success."""
    try:
        if not dry_run:
            # Aliases block deletion even with delete_aliases=True in some SDK versions
            artifact.aliases = []
            artifact.save()
            artifact.delete(delete_aliases=True)
        return True
    except Exception as e:
        print(f"  WARNING: could not delete {artifact.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True, help="W&B run group to scope cleanup to")
    parser.add_argument("--type", default="model", help="Artifact type (default: model)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    api = wandb.Api()

    # Collect artifact base-names produced by runs in the target group
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
    artifact_names = set()
    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type == args.type:
                artifact_names.add(artifact.name.rsplit(":", 1)[0])

    if not artifact_names:
        print(f"No {args.type} artifacts found for group '{args.group}'.")
        return

    print(f"Found {len(artifact_names)} artifact collection(s) in group '{args.group}'.")

    total_deleted = total_failed = 0
    for name in sorted(artifact_names):
        versions = sorted(
            api.artifact_versions(args.type, f"{args.entity}/{args.project}/{name}"),
            key=_version_int,
        )
        stale = versions[:-1]  # keep only the latest
        for artifact in stale:
            size_mb = artifact.size / 1e6
            tag = "[dry-run] " if args.dry_run else ""
            print(f"{tag}deleting {artifact.name} (aliases={artifact.aliases}, {size_mb:.1f} MB)")
            ok = _delete(artifact, args.dry_run)
            if ok:
                total_deleted += 1
            else:
                total_failed += 1

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {total_deleted} artifact version(s). Failed: {total_failed}.")


if __name__ == "__main__":
    main()
