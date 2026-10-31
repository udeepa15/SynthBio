"""
Goodman 2013 as a within gene source.

Goodman DB, Church GM, Kosuri S. "Causes and effects of N-terminal codon bias in
bacterial genes." Science 342(6157):475-479. doi:10.1126/science.1241934

14,234 designed variants of the first 11 codons of 137 E. coli genes, each
fused to a fixed superfolder GFP reporter, measured by sorting cells into
fluorescence bins and sequencing. Like Cambray it was ruled out as training data
for the tier model because the protein is held constant, and like Cambray that
is exactly what makes it usable here.

The grouping. The library crosses three things: the parent gene, one of two
promoters, and one of 140 ribosome binding sites. Only the codons are meant to
vary within a design, so the query group is the gene with its promoter and its
ribosome binding site held fixed, which gives 1,096 groups covering every row.
Checked rather than assumed: grouping by gene, promoter and site yields exactly
the same 1,096 groups as grouping by gene, promoter, site and translated
peptide, so the peptide really is constant inside each one.

The readout. The sheet publishes `Trans`, protein per transcript, and `RNA`,
transcript abundance, but no protein column. Protein level is their product,
which is confirmed against the sorted bins: the fluorescence weighted mean of
`Bin.Pct.1` to `Bin.Pct.12` agrees with `Trans` times `RNA` at Spearman 0.999
over the 14,137 rows carrying both. Protein is the ranking target rather than
`Trans` itself, because the question this agent answers is which synonymous
version makes more protein; protein per transcript is a different question and
would need its own gate.

The filter. `Fltr.SetGood` marks the 8,439 variants the authors treat as having
enough reads in every channel to be analysed. It is applied here, because a
ranking built on variants the authors would not analyse is a ranking of noise.

Two limits travel with the data and belong in the agent's competence domain.
The sequences are the 33 nt designed region, 11 codons, shorter even than
Cambray's 96. And the readout is a fluorescence reporter, so it measures
translation rate rather than soluble yield.

Download it by hand: the file is `1241934tables1.csv` in the paper's
supplementary materials, and science.org serves it only to a browser. Put it at
one of `SEARCH_PATHS`.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.datasets.variant_sets import MissingVariantSource, VariantSet, canonicalise

SOURCE = "goodman2013"
CITATION = (
    "Goodman DB, Church GM, Kosuri S. Causes and effects of N-terminal codon bias in bacterial "
    "genes. Science 342:475-479 (2013)."
)
READOUT_KIND = "protein_fluorescence"

SEARCH_PATHS = (
    "data/external/goodman2013/raw/1241934tables1.csv",
    "data/external/goodman2013/variants.csv",
    "data/external/goodman2013/goodman2013.csv",
    "data/raw/goodman2013.csv",
)

# Columns in the authors' sheet.
SEQUENCE_COLUMN = "CDS.seq"
TRANSLATION_COLUMN = "Trans"
RNA_COLUMN = "RNA"
NAME_COLUMN = "Name"
GROUP_COLUMNS = ("Gene", "Promoter", "RBS.seq")
QUALITY_COLUMN = "Fltr.SetGood"
BIN_COLUMNS = tuple(f"Bin.Pct.{i}" for i in range(1, 13))


def path() -> Optional[str]:
    return next((p for p in SEARCH_PATHS if os.path.exists(p)), None)


def bin_weighted_protein(frame: pd.DataFrame) -> pd.Series:
    """
    Protein level read straight off the sorted fluorescence bins.

    Kept as a cross check on `Trans` times `RNA` rather than as the readout
    itself, because the bins are a twelve step ladder while the product is
    continuous.
    """
    present = [c for c in BIN_COLUMNS if c in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index)
    weights = np.arange(1, len(present) + 1, dtype=float)
    percentages = frame[present].to_numpy(dtype=float)
    totals = np.nansum(percentages, axis=1)
    weighted = np.nansum(percentages * weights, axis=1)
    return pd.Series(
        np.where(totals > 0, weighted / np.where(totals == 0, 1.0, totals), np.nan),
        index=frame.index,
    )


def load(apply_quality_filter: bool = True) -> VariantSet:
    """The Goodman library in canonical within gene shape."""
    found = path()
    if found is None:
        raise MissingVariantSource(
            "Goodman 2013 is not in the working tree. Looked for "
            + ", ".join(SEARCH_PATHS)
            + ". The file is `1241934tables1.csv` in the paper's supplementary materials, which "
            "science.org serves only to a browser, so it has to be downloaded by hand. Source: "
            + CITATION
        )

    frame = pd.read_csv(found, low_memory=False)
    missing = [
        c for c in (SEQUENCE_COLUMN, TRANSLATION_COLUMN, RNA_COLUMN, *GROUP_COLUMNS)
        if c not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"'{found}' is missing {', '.join(missing)}. The sheet layout has changed and the "
            "column mapping in src/datasets/goodman.py must be revisited."
        )

    notes = [f"Read from '{found}'."]
    if apply_quality_filter and QUALITY_COLUMN in frame.columns:
        good = frame[QUALITY_COLUMN].astype(str).str.lower().eq("true")
        notes.append(
            f"Kept the {int(good.sum()):,} of {len(frame):,} variants the authors mark "
            f"`{QUALITY_COLUMN}`, which is their own read depth filter."
        )
        frame = frame[good]

    protein = frame[TRANSLATION_COLUMN] * frame[RNA_COLUMN]
    check = bin_weighted_protein(frame)
    both = protein.notna() & check.notna()
    if both.sum() > 100:
        from scipy.stats import spearmanr

        agreement = float(spearmanr(protein[both], check[both]).statistic)
        notes.append(
            f"Protein level is `{TRANSLATION_COLUMN}` times `{RNA_COLUMN}`, which agrees with the "
            f"fluorescence weighted mean of the sorted bins at Spearman {agreement:.3f}."
        )

    tidy = pd.DataFrame({
        "variant_id": (
            frame[NAME_COLUMN].astype(str)
            if NAME_COLUMN in frame.columns
            else [f"{SOURCE}_{i}" for i in range(len(frame))]
        ),
        "dna": frame[SEQUENCE_COLUMN],
        "readout": protein,
        "group_id": frame[list(GROUP_COLUMNS)].astype(str).agg("|".join, axis=1),
    })

    notes.append(
        "Grouped by gene, promoter and ribosome binding site, which holds the peptide constant "
        "inside every group; only the codons vary."
    )
    notes.append(
        "Sequences are the 33 nt designed region, 11 codons, not whole genes. Every feature is "
        "computed over that window."
    )
    return VariantSet(
        name=SOURCE,
        frame=canonicalise(tidy, SOURCE, READOUT_KIND),
        citation=CITATION,
        readout_kind=READOUT_KIND,
        notes=tuple(notes),
    )
