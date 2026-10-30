"""
Competence domain checks shared by every agent.

Two questions, asked of every input before any agent is believed:

1. Does each feature fall inside the range the agent was fitted on? A gradient
   boosted tree has no slope beyond its outermost split, so a value past the
   training range yields a confident looking answer with nothing behind it.
   `check_feature_ranges` in `src/predict.py` already does this and is reused
   rather than reimplemented.

2. Is there anything in the training set that resembles this sequence? A
   construct inside every feature range can still be unlike every training row,
   which is the case the human genes in `data/testsets` are kept for. This is
   the half of a competence domain the project does not have yet, because the
   training table carries features and gene ids but no sequences.

The second check reports itself as unavailable rather than silently passing. An
input is never called in domain on the strength of a range check alone.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.agents.base import DomainCheck

logger = logging.getLogger("SynthBio.agents.domain")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.domain] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Where a reference set of training sequences would live if the project had one.
# `feature_table_v3.csv` is the table that carried the `seq` column; it is
# gitignored and is not in the working tree, which is why the identity check
# below reports itself unavailable.
TRAINING_SEQUENCE_CANDIDATES = (
    "data/processed/feature_table_v3.csv",
    "data/processed/boel_sequences.csv",
)
SEQUENCE_COLUMN_CANDIDATES = ("seq", "sequence", "orf_dna", "dna")

# Length of the protein k-mer used by the identity approximation below.
KMER = 4

# Containment above this is treated as a near neighbour. It is a k-mer
# approximation and not an alignment, so it is reported as approximate
# everywhere it is used and is never presented as a percentage identity from
# MMseqs2 or BLAST.
NEAR_NEIGHBOUR_CONTAINMENT = 0.30


def _translate(dna: str) -> str:
    """Protein of a coding sequence, stopping at the first stop codon."""
    from Bio.Seq import Seq

    clean = dna.upper().replace(" ", "").replace("\n", "")
    clean = clean[: len(clean) - len(clean) % 3]
    if not clean:
        return ""
    return str(Seq(clean).translate(to_stop=True))


def _kmers(protein: str, k: int = KMER) -> frozenset:
    if len(protein) < k:
        return frozenset({protein}) if protein else frozenset()
    return frozenset(protein[i : i + k] for i in range(len(protein) - k + 1))


@lru_cache(maxsize=1)
def _reference_kmer_sets() -> Optional[Tuple[Tuple[str, frozenset], ...]]:
    """
    Protein k-mer sets for the training rows, or None when no sequence table is
    on disk.

    Cached because the index is read once per process and the application scores
    many sequences against it.
    """
    for path in TRAINING_SEQUENCE_CANDIDATES:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        column = next((c for c in SEQUENCE_COLUMN_CANDIDATES if c in frame.columns), None)
        if column is None:
            logger.warning(f"'{path}' has no sequence column; identity check cannot use it.")
            continue
        index = []
        for row_id, dna in zip(frame.get("name", frame.index), frame[column].astype(str)):
            protein = _translate(dna)
            if protein:
                index.append((str(row_id), _kmers(protein)))
        if index:
            logger.info(f"Identity index built from '{path}': {len(index)} training sequences.")
            return tuple(index)
    logger.info(
        "No training sequence table on disk, so the nearest neighbour identity check is unavailable."
    )
    return None


def nearest_identity(sequence: str) -> Tuple[Optional[float], Optional[str], str]:
    """
    Approximate identity of `sequence` to the closest training row.

    Returns the containment in 0 to 1, the id of the closest row, and a note.
    Containment of protein k-mers is used rather than an alignment, because
    MMseqs2 is not installed and a pure Python alignment over thousands of rows
    is too slow for an interactive application. It overestimates identity for
    short sequences and for repeats, so it is only ever used to decide whether
    anything resembles the input at all.

    Returns `(None, None, reason)` when no training sequences are available.
    """
    index = _reference_kmer_sets()
    if index is None:
        return None, None, (
            "The training sequences are not in the working tree, so how closely this construct "
            "resembles the training set is unknown. Only the feature ranges were checked."
        )

    protein = _translate(sequence)
    query = _kmers(protein)
    if not query:
        return None, None, "The construct does not translate, so no identity comparison was made."

    best, best_id = 0.0, None
    for row_id, reference in index:
        if not reference:
            continue
        shared = len(query & reference)
        if not shared:
            continue
        containment = shared / len(query)
        if containment > best:
            best, best_id = containment, row_id

    note = (
        f"Closest training sequence is {best_id} at approximately {best * 100:.0f} per cent "
        f"shared {KMER}-mers. This is a k-mer containment and not an alignment."
    )
    return best, best_id, note


def describe_domain(
    feature_row: pd.DataFrame,
    ranges: Optional[Dict[str, Dict[str, float]]],
    sequence: Optional[str] = None,
    check_identity: bool = True,
) -> DomainCheck:
    """
    The full competence domain check for one input.

    `feature_row` is a one row frame of the agent's own feature columns and
    `ranges` the agent's training ranges, both in the shape the existing
    metadata sidecar already stores.
    """
    from src.predict import check_feature_ranges

    out_of_range = tuple(check_feature_ranges(feature_row, ranges))
    notes: List[str] = []
    identity: Optional[float] = None
    checked = False

    if sequence and check_identity:
        identity, _, note = nearest_identity(sequence)
        notes.append(note)
        checked = identity is not None

    if not ranges:
        notes.append(
            "No training ranges were recorded for this agent, so no range check was possible."
        )

    return DomainCheck(
        out_of_range=out_of_range,
        nearest_identity=identity,
        identity_checked=checked,
        notes=tuple(notes),
    )


def load_feature_ranges(metadata_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Training ranges from an artifact's metadata sidecar, or None."""
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("feature_ranges")
    except (OSError, json.JSONDecodeError) as err:
        logger.warning(f"Could not read feature ranges from '{metadata_path}': {err}")
        return None
