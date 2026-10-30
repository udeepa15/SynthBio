"""
Cluster the training sequences and reserve the critic pool. Milestone M0.

    python scripts/cluster.py

Clusters the Boel proteins at 30 per cent sequence identity with MMseqs2
`easy-cluster`, holds back a fifth of the clusters as the critic pool, and
writes `reports/clusters.json`, which is committed.

Why this runs once and is then committed. Every gate measured against a given
pool is only comparable to another gate measured against the same pool, so a
later run that drew a different pool would invalidate every recorded number
without anything failing. The assignment is therefore a repository artifact
rather than something each run recomputes, and this script refuses to overwrite
it unless asked.

Why clusters and not organisms. Organism grouping, which the served model uses,
was a real improvement and is not sufficient: paralogues inside one organism and
homologues across two organisms both still leak. The critic is trained on the
specialists' predictions, so leakage there does not merely inflate a score, it
makes the agreement score meaningless in exactly the cases it exists for.

MMseqs2 is a separate binary and is not a Python dependency. Install it, or
point `SYNTHBIO_MMSEQS` at a build, and this script will find it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import clustering  # noqa: E402

logger = logging.getLogger("SynthBio.cluster")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.cluster] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=clustering.CLUSTER_REPORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redraw the assignment even though one is committed. Every gate measured against the "
        "old pool stops being comparable, so rerun scripts/gates.py afterwards.",
    )
    args = parser.parse_args()

    status = clustering.clustering_status()
    if status is not None:
        logger.error(status)
        if clustering.boel_sequence_path() is None:
            logger.error("Recover the sequences with `python scripts/fetch_datasets.py --source boel`.")
        if clustering.mmseqs_path() is None:
            logger.error(
                "MMseqs2 is at https://github.com/soedinglab/MMseqs2/releases. Put it on PATH, or "
                "set SYNTHBIO_MMSEQS to the binary."
            )
        return 1

    if os.path.exists(args.out) and not args.force:
        existing = clustering.load_assignment(args.out)
        logger.error(
            f"'{args.out}' already records {existing.n_clusters:,} clusters and a pool of "
            f"{len(existing.critic_pool):,}. Pass --force to redraw it, which invalidates every "
            "gate measured against the current pool."
        )
        return 1

    assignment = clustering.build_assignment()
    clustering.write_assignment(assignment, args.out)
    logger.info(
        f"{len(assignment.clusters):,} sequences in {assignment.n_clusters:,} clusters; "
        f"{len(assignment.critic_pool):,} clusters reserved as the critic pool, covering "
        f"{len(assignment.critic_ids()):,} sequences. Rerun `python scripts/gates.py`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
