"""
Cambray 2018, about 244,000 synthetic sequences, as a within gene source.

Cambray G, Guimaraes JC, Arkin AP. "Evaluation of 244,000 synthetic sequences
reveals design principles to optimize translation in Escherichia coli."
Nature Biotechnology 36(10):1005-1015. doi:10.1038/nbt.4238

Every variant is an N terminal fusion to one sfGFP scaffold, which is why the
dataset survey ruled it out as training data for the expression tier model: the
reporter's own protein features are frozen across the library and the readout
is translation rate rather than soluble yield. For a within gene ranker both of
those objections become reasons to use it, and the release ships exact
nucleotide sequences, so there is no DNA recovery problem.

One claim about this library that the plan makes is wrong, and the grouping
below is the consequence. The designed 96 nt region is not synonymous across
the library. Mean hydropathy of that region is one of the eight properties the
design of experiments varies deliberately, so the amino acids change along with
the codons: of 244,000 variants there are 157,278 distinct designed peptides,
and even inside one design cell, one seed and one property combination, the
peptide usually differs. A ranker trained on a group whose protein varies would
learn protein differences and attribute them to codon features, which is the
failure this agent exists to fix, in a subtler form.

So the query group here is the translated peptide, not the seed and not the
design cell. Grouped that way the library still yields a large clean training
set: 4,151 peptides carry four or more distinct synonymous nucleotide
sequences, covering roughly 75,000 measured variants in which the protein is
held exactly constant and only the codons differ. That is the controlled
comparison the redesign tab makes.

Two limits travel with the data and belong in the agent's competence domain.
The sequences are 96 nt, so every feature is computed over a 32 codon window
rather than a whole gene. And the readout is protein level from a fixed
reporter, which is translation rate, not soluble yield.

Download it with `python scripts/fetch_datasets.py --source cambray`, which
reduces the authors' 306 MB table to the three columns used here.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd

from src.datasets.variant_sets import MissingVariantSource, VariantSet, canonicalise

SOURCE = "cambray2018"
CITATION = (
    "Cambray G, Guimaraes JC, Arkin AP. Evaluation of 244,000 synthetic sequences reveals design "
    "principles to optimize translation in Escherichia coli. Nat Biotechnol 36:1005-1015 (2018)."
)
READOUT_KIND = "protein_fluorescence"

SEARCH_PATHS = (
    "data/external/cambray2018/variants.csv",
    "data/external/cambray2018/cambray2018.csv",
    "data/raw/cambray2018.csv",
)

SEQUENCE_COLUMNS = ("sequence", "dna", "variant_sequence", "nucleotide_sequence", "seq")
READOUT_COLUMNS = ("protein", "protein_level", "fluorescence", "gfp", "protein_per_mrna", "readout")
ID_COLUMNS = ("variant_id", "id", "name", "design_id", "oligo_id")
GROUP_COLUMNS = ("group_id",)


def _first(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def path() -> Optional[str]:
    """The first search path that exists, or None."""
    return next((p for p in SEARCH_PATHS if os.path.exists(p)), None)


def load() -> VariantSet:
    """
    The Cambray library in canonical within gene shape.

    Raises `MissingVariantSource` when the file is absent, which is the current
    state of the working tree.
    """
    found = path()
    if found is None:
        raise MissingVariantSource(
            "Cambray 2018 is not in the working tree. Looked for "
            + ", ".join(SEARCH_PATHS)
            + ". Download it with `python scripts/fetch_datasets.py --source cambray`. Source: "
            + CITATION
        )

    frame = pd.read_csv(found)
    sequence_column = _first(frame, SEQUENCE_COLUMNS)
    readout_column = _first(frame, READOUT_COLUMNS)
    if sequence_column is None or readout_column is None:
        raise ValueError(
            f"'{found}' does not carry both a sequence column and a readout column. "
            f"Looked for sequences in {SEQUENCE_COLUMNS} and a readout in {READOUT_COLUMNS}."
        )

    id_column = _first(frame, ID_COLUMNS)
    group_column = _first(frame, GROUP_COLUMNS)

    dna = frame[sequence_column].astype(str).str.upper().str.replace(r"\s", "", regex=True)
    tidy = pd.DataFrame({
        "variant_id": frame[id_column].astype(str) if id_column else [f"{SOURCE}_{i}" for i in range(len(frame))],
        "dna": dna,
        "readout": frame[readout_column],
        "group_id": frame[group_column].astype(str) if group_column else group_by_peptide(dna),
    })
    canonical = canonicalise(tidy, SOURCE, READOUT_KIND)
    return VariantSet(
        name=SOURCE,
        frame=canonical,
        citation=CITATION,
        readout_kind=READOUT_KIND,
        notes=(
            f"Read from '{found}'.",
            "Grouped by translated peptide, because the designed region varies amino acids as well "
            "as codons. Groups smaller than the minimum are dropped by `canonicalise`, so what "
            "survives is only the synonymous part of the library.",
            "Sequences are the 96 nt designed region, 32 codons, not whole genes. Every feature is "
            "computed over that window.",
            "The readout is protein level from a fixed sfGFP reporter, so it measures translation "
            "rate rather than soluble yield.",
        ),
    )


def group_by_peptide(dna: pd.Series) -> pd.Series:
    """
    The query group: variants that encode exactly the same peptide.

    This is the only grouping in this library that holds the protein constant,
    and holding the protein constant is the whole point of a within gene
    ranker. See the module docstring for why the design cell will not do.
    """
    from Bio.Seq import Seq

    peptides = []
    for sequence in dna:
        trimmed = sequence[: len(sequence) - len(sequence) % 3]
        peptides.append(str(Seq(trimmed).translate()) if trimmed else "")
    return pd.Series([f"pep_{p}" for p in peptides], index=dna.index)
