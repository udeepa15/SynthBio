"""
Goodman 2013 as a within gene source.

Goodman DB, Church GM, Kosuri S. "Causes and effects of N-terminal codon bias in
bacterial genes." Science 342(6157):475-479. doi:10.1126/science.1241934

About 14,000 designed N terminal variants of a fixed reporter, with measured
protein and mRNA levels, so translation efficiency is separable from
transcription. Like Cambray it was ruled out as training data for the tier
model because the protein is held constant, and like Cambray that is precisely
what makes it usable here.

The supplementary table is a modest file rather than a large release, so this is
the cheaper of the two sources to obtain. Drop it at one of `SEARCH_PATHS`.

Column naming varies between the published table and its reposted copies, so
the loader looks for any of several names rather than insisting on one layout.
When both a protein and an mRNA column are present, protein is the ranking
target and the mRNA column is kept out: the question Agent C answers is which
synonymous version expresses more protein, and dividing by mRNA would change the
question to translation efficiency per transcript, which is a different thing
and would need its own gate.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd

from src.datasets.variant_sets import MissingVariantSource, VariantSet, canonicalise

SOURCE = "goodman2013"
CITATION = (
    "Goodman DB, Church GM, Kosuri S. Causes and effects of N-terminal codon bias in bacterial "
    "genes. Science 342:475-479 (2013)."
)
READOUT_KIND = "protein_fluorescence"

SEARCH_PATHS = (
    "data/external/goodman2013/variants.csv",
    "data/external/goodman2013/goodman2013.csv",
    "data/raw/goodman2013.csv",
)

SEQUENCE_COLUMNS = ("sequence", "dna", "variant_sequence", "cds", "seq")
PROTEIN_COLUMNS = ("protein", "protein_level", "prot", "fluorescence", "gfp")
MRNA_COLUMNS = ("rna", "mrna", "rna_level", "transcript")
ID_COLUMNS = ("variant_id", "id", "name", "oligo_id")
GROUP_COLUMNS = ("group_id", "gene", "parent_gene", "scaffold", "promoter")

DEFAULT_GROUP = "goodman_reporter"


def _first(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def path() -> Optional[str]:
    return next((p for p in SEARCH_PATHS if os.path.exists(p)), None)


def load() -> VariantSet:
    """The Goodman library in canonical within gene shape."""
    found = path()
    if found is None:
        raise MissingVariantSource(
            "Goodman 2013 is not in the working tree. Looked for "
            + ", ".join(SEARCH_PATHS)
            + ". The file needs one row per variant with the nucleotide sequence, a variant id and "
            "a measured protein level. Source: " + CITATION
        )

    frame = pd.read_csv(found)
    sequence_column = _first(frame, SEQUENCE_COLUMNS)
    protein_column = _first(frame, PROTEIN_COLUMNS)
    if sequence_column is None or protein_column is None:
        raise ValueError(
            f"'{found}' does not carry both a sequence column and a protein level column. "
            f"Looked for sequences in {SEQUENCE_COLUMNS} and protein in {PROTEIN_COLUMNS}."
        )

    id_column = _first(frame, ID_COLUMNS)
    group_column = _first(frame, GROUP_COLUMNS)
    mrna_column = _first(frame, MRNA_COLUMNS)

    tidy = pd.DataFrame({
        "variant_id": frame[id_column].astype(str) if id_column else [f"{SOURCE}_{i}" for i in range(len(frame))],
        "dna": frame[sequence_column],
        "readout": frame[protein_column],
        "group_id": frame[group_column].astype(str) if group_column else DEFAULT_GROUP,
    })
    notes = [f"Read from '{found}'."]
    if mrna_column:
        notes.append(
            f"An mRNA column ('{mrna_column}') is present and is deliberately unused. Protein level "
            "is the ranking target; protein per transcript is a different question with its own gate."
        )
    return VariantSet(
        name=SOURCE,
        frame=canonicalise(tidy, SOURCE, READOUT_KIND),
        citation=CITATION,
        readout_kind=READOUT_KIND,
        notes=tuple(notes),
    )
