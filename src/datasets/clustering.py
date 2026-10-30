"""
Identity clustering and the reserved critic pool.

This is the guard the whole system rests on, from the plan's section 8. The
critic is trained on the specialists' predictions, so if a specialist's
training rows overlap the critic's training or evaluation rows, the critic
learns that the specialist is more accurate than it really is, and the
agreement score becomes meaningless in exactly the cases it exists for. Such a
system would report high confidence and high agreement and would be wrong,
which is strictly worse than the present model, because the present model at
least claims nothing.

Three overlaps are specific to this project and all three are documented
already. Boel and Price 2011 are both NESG sets from the same pipeline, so they
are siblings rather than independent samples. UESolDS was clustered at 25 per
cent identity with insoluble sequences above 75 per cent identity to soluble
ones removed, so merging it with unclustered Boel rows mixes two sampling
regimes. SoluProt used the NESG data as its own test set and had to remove high
identity sequences to do so.

The rule: cluster the union of every dataset used by any agent, once, assign
every row a cluster id and a source label, reserve the critic pool by cluster,
and never split by row anywhere in the system.

Two things stop that running in this working tree, and both are reported rather
than worked around:

- The Boel training table carries gene ids and the ten features but no
  sequences. `feature_table_v3.csv`, which held the `seq` column, is gitignored
  and absent, so there is nothing to cluster.
- MMseqs2 is not installed, so even with sequences the clustering the plan
  specifies, `easy-cluster` at a minimum sequence identity of 0.3 and a
  coverage of 0.8, cannot be run.

Organism grouping, which the served model already uses, is kept as a second
reported split rather than as a substitute. It was a real improvement and it is
not sufficient: paralogues inside one organism and homologues across two
organisms both still leak.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("SynthBio.datasets.clustering")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.datasets.clustering] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MIN_SEQUENCE_IDENTITY = 0.3
MIN_COVERAGE = 0.8

CLUSTER_REPORT = "reports/clusters.json"

# Where the Boel sequences would be. See the module docstring.
BOEL_SEQUENCE_PATHS = (
    "data/processed/feature_table_v3.csv",
    "data/processed/boel_sequences.csv",
)
SEQUENCE_COLUMNS = ("seq", "sequence", "dna")
ID_COLUMNS = ("name", "gene_id", "id")

# Share of clusters held back as the critic pool. Reserved at the start, before
# any specialist is fitted, and never used for training, tuning or gate
# reporting of any specialist.
CRITIC_POOL_FRACTION = 0.2
CRITIC_POOL_SEED = 42


class ClusteringUnavailable(RuntimeError):
    """Raised when identity clustering cannot be run, with the reason."""


@dataclass
class ClusterAssignment:
    """Cluster ids per row, plus the reserved critic pool."""

    clusters: Dict[str, str]
    critic_pool: Tuple[str, ...]
    method: str
    min_identity: float
    coverage: float
    source_counts: Dict[str, int]

    @property
    def n_clusters(self) -> int:
        return len(set(self.clusters.values()))

    def training_ids(self) -> List[str]:
        pool = set(self.critic_pool)
        return [row for row, cluster in self.clusters.items() if cluster not in pool]

    def critic_ids(self) -> List[str]:
        pool = set(self.critic_pool)
        return [row for row, cluster in self.clusters.items() if cluster in pool]


def mmseqs_path() -> Optional[str]:
    """
    The MMseqs2 binary, or None when it is not installed.

    `SYNTHBIO_MMSEQS` overrides the lookup, so a build that is not on PATH can
    be pointed at without changing the environment for everything else.
    """
    override = os.environ.get("SYNTHBIO_MMSEQS")
    if override:
        return override if os.path.exists(override) else shutil.which(override)
    return shutil.which("mmseqs")


def boel_sequence_path() -> Optional[str]:
    return next((p for p in BOEL_SEQUENCE_PATHS if os.path.exists(p)), None)


def clustering_status() -> Optional[str]:
    """
    None when identity clustering can be run, otherwise the reason it cannot.

    Written as one sentence per obstacle so the gate report can print it
    directly.
    """
    reasons = []
    if boel_sequence_path() is None:
        reasons.append(
            "the training sequences are not in the working tree (looked for "
            + ", ".join(BOEL_SEQUENCE_PATHS)
            + "), so there is nothing to cluster"
        )
    if mmseqs_path() is None:
        reasons.append(
            "MMseqs2 is not installed, so easy-cluster at a minimum sequence identity of "
            f"{MIN_SEQUENCE_IDENTITY} and a coverage of {MIN_COVERAGE} cannot be run"
        )
    if not reasons:
        return None
    return "Identity clustering is blocked: " + "; ".join(reasons) + "."


def load_sequences() -> List[Tuple[str, str]]:
    """
    The `(id, protein)` pairs to cluster, from whichever sequence table exists.

    Raises `ClusteringUnavailable` when no table is present, rather than
    returning an empty list, so a caller cannot mistake absent sequences for a
    dataset with no rows.
    """
    path = boel_sequence_path()
    if path is None:
        raise ClusteringUnavailable(clustering_status() or "No sequence table found.")

    from Bio.Seq import Seq

    frame = pd.read_csv(path)
    sequence_column = next((c for c in SEQUENCE_COLUMNS if c in frame.columns), None)
    id_column = next((c for c in ID_COLUMNS if c in frame.columns), None)
    if sequence_column is None:
        raise ClusteringUnavailable(f"'{path}' carries no sequence column.")

    records = []
    for index, row in frame.iterrows():
        dna = str(row[sequence_column]).upper()
        dna = dna[: len(dna) - len(dna) % 3]
        if not dna:
            continue
        protein = str(Seq(dna).translate(to_stop=True))
        if protein:
            records.append((str(row[id_column]) if id_column else str(index), protein))
    logger.info(f"Loaded {len(records)} sequences to cluster from '{path}'.")
    return records


def cluster_with_mmseqs(
    records: Sequence[Tuple[str, str]],
    min_identity: float = MIN_SEQUENCE_IDENTITY,
    coverage: float = MIN_COVERAGE,
) -> Dict[str, str]:
    """
    Cluster protein sequences with MMseqs2 easy-cluster.

    The representative of each cluster becomes the cluster id, which is what
    `easy-cluster` writes in the first column of its `_cluster.tsv` output.
    """
    binary = mmseqs_path()
    if binary is None:
        raise ClusteringUnavailable(
            "MMseqs2 is not installed, so identity clustering cannot be run. Install it and rerun "
            "the gate harness; nothing else in the system needs to change."
        )

    with tempfile.TemporaryDirectory(prefix="synthbio_mmseqs_") as tmp:
        fasta = os.path.join(tmp, "input.fasta")
        with open(fasta, "w", encoding="utf-8") as handle:
            for row_id, protein in records:
                handle.write(f">{row_id}\n{protein}\n")

        prefix = os.path.join(tmp, "clustered")
        command = [
            binary, "easy-cluster", fasta, prefix, os.path.join(tmp, "work"),
            "--min-seq-id", str(min_identity),
            "-c", str(coverage),
            "--cov-mode", "0",
        ]
        logger.info("Running: " + " ".join(command))
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ClusteringUnavailable(
                f"MMseqs2 easy-cluster failed with exit code {completed.returncode}: "
                + (completed.stderr or completed.stdout or "no output").strip()[-500:]
            )

        tsv = prefix + "_cluster.tsv"
        if not os.path.exists(tsv):
            raise ClusteringUnavailable(f"MMseqs2 produced no cluster table at '{tsv}'.")
        table = pd.read_csv(tsv, sep="\t", header=None, names=["representative", "member"])

    clusters = {str(m): str(r) for r, m in zip(table["representative"], table["member"])}
    logger.info(f"MMseqs2 assigned {len(clusters)} sequences to {len(set(clusters.values()))} clusters.")
    return clusters


def reserve_critic_pool(
    clusters: Dict[str, str],
    fraction: float = CRITIC_POOL_FRACTION,
    seed: int = CRITIC_POOL_SEED,
) -> Tuple[str, ...]:
    """
    Hold back a share of clusters that no specialist may ever see.

    Reserved by cluster and never by row, and reserved once: the ids are written
    to `reports/clusters.json` and committed, so a later run cannot quietly
    draw a different pool and invalidate every gate measured against the old
    one.
    """
    import random

    unique = sorted(set(clusters.values()))
    rng = random.Random(seed)
    count = max(1, int(round(len(unique) * fraction)))
    pool = tuple(sorted(rng.sample(unique, count)))
    logger.info(f"Reserved {len(pool)} of {len(unique)} clusters as the critic pool.")
    return pool


def build_assignment(source: str = "boel2016") -> ClusterAssignment:
    """
    Cluster the sequences and reserve the critic pool, in one call.

    Raises `ClusteringUnavailable` with the reason in the current working tree.
    """
    records = load_sequences()
    clusters = cluster_with_mmseqs(records)
    pool = reserve_critic_pool(clusters)
    return ClusterAssignment(
        clusters=clusters,
        critic_pool=pool,
        method=f"mmseqs2 easy-cluster --min-seq-id {MIN_SEQUENCE_IDENTITY} -c {MIN_COVERAGE}",
        min_identity=MIN_SEQUENCE_IDENTITY,
        coverage=MIN_COVERAGE,
        source_counts={source: len(records)},
    )


def write_assignment(assignment: ClusterAssignment, path: str = CLUSTER_REPORT) -> str:
    """Commit the cluster assignment and the reserved pool to the repository."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "method": assignment.method,
        "min_sequence_identity": assignment.min_identity,
        "coverage": assignment.coverage,
        "n_sequences": len(assignment.clusters),
        "n_clusters": assignment.n_clusters,
        "critic_pool_fraction": CRITIC_POOL_FRACTION,
        "critic_pool_seed": CRITIC_POOL_SEED,
        "critic_pool": list(assignment.critic_pool),
        "source_counts": assignment.source_counts,
        "clusters": assignment.clusters,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info(f"Wrote {path}")
    return path


def load_assignment(path: str = CLUSTER_REPORT) -> Optional[ClusterAssignment]:
    """The committed cluster assignment, or None when it has never been built."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ClusterAssignment(
        clusters={str(k): str(v) for k, v in payload.get("clusters", {}).items()},
        critic_pool=tuple(payload.get("critic_pool", ())),
        method=payload.get("method", "unknown"),
        min_identity=float(payload.get("min_sequence_identity", MIN_SEQUENCE_IDENTITY)),
        coverage=float(payload.get("coverage", MIN_COVERAGE)),
        source_counts=payload.get("source_counts", {}),
    )


def organism_groups(gene_ids: Sequence[str]) -> List[str]:
    """
    The organism grouping the served model already uses, kept for comparison.

    Boel's NESG gene ids carry the source organism as a two or three letter
    prefix, so `AaR17` groups with every other `Aa` gene. This is reported
    beside any identity clustered figure and never instead of one.
    """
    groups = []
    for gene_id in gene_ids:
        text = str(gene_id)
        prefix = "".join(ch for ch in text[:3] if ch.isalpha())
        groups.append(prefix or text)
    return groups
