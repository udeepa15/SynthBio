"""
The one shape every within gene variant source is read into.

A within gene source is a set of query groups. Inside a group every sequence
encodes the same protein and only the codons differ, and each sequence carries a
measured readout that orders the group. That is the whole contract: Agent C
never sees anything else, and nothing about a particular publication's file
layout reaches the ranker.

Canonical columns:

    source_dataset  which publication the row came from, never dropped
    group_id        the query group, the parent gene or scaffold
    variant_id      unique within the source
    dna             the exact coding sequence, A, C, G and T only
    readout         the measured quantity that orders the group, higher better
    readout_kind    what was measured, for instance 'protein_fluorescence'

A source is not loadable until its file is in the working tree. Each loader
raises `MissingVariantSource` naming the path it looked for and what the file
should contain, so the trainer can report what is missing rather than failing
with a parse error on a file that is not there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("SynthBio.datasets")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.datasets] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

VARIANT_SET_COLUMNS = ("source_dataset", "group_id", "variant_id", "dna", "readout", "readout_kind")

# A query group below this size cannot train a within group ranker and cannot be
# scored by a rank correlation either.
MIN_GROUP_SIZE = 4


class MissingVariantSource(FileNotFoundError):
    """Raised when a source's file is not in the working tree."""


@dataclass
class VariantSet:
    """One ingested within gene source, already in canonical shape."""

    name: str
    frame: pd.DataFrame
    citation: str
    readout_kind: str
    notes: Tuple[str, ...] = ()

    @property
    def n_groups(self) -> int:
        return int(self.frame["group_id"].nunique())

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.frame):,} sequences in {self.n_groups:,} query group(s), "
            f"readout '{self.readout_kind}'."
        )


def canonicalise(
    frame: pd.DataFrame,
    source: str,
    readout_kind: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Validate and tidy a loader's output into the canonical frame.

    Drops rows with a non ACGT sequence, a sequence whose length is not a
    multiple of three, or a missing readout, and then drops groups left too
    small to rank. Every drop is logged with a count, because a loader that
    silently discards most of a library is the kind of thing that only shows up
    as an unexplained gate result weeks later.
    """
    missing = [c for c in ("group_id", "variant_id", "dna", "readout") if c not in frame.columns]
    if missing:
        raise ValueError(f"{source}: loader returned no {', '.join(missing)} column.")

    out = frame.copy()
    out["dna"] = out["dna"].astype(str).str.upper().str.replace(r"\s", "", regex=True)
    before = len(out)

    bad_alphabet = ~out["dna"].str.fullmatch(r"[ACGT]+")
    if bad_alphabet.any():
        logger.warning(f"{source}: dropping {int(bad_alphabet.sum())} row(s) with a non ACGT sequence.")
        out = out[~bad_alphabet]

    bad_frame = out["dna"].str.len() % 3 != 0
    if bad_frame.any():
        logger.warning(f"{source}: dropping {int(bad_frame.sum())} row(s) whose length is not a multiple of three.")
        out = out[~bad_frame]

    out["readout"] = pd.to_numeric(out["readout"], errors="coerce")
    no_readout = out["readout"].isna()
    if no_readout.any():
        logger.warning(f"{source}: dropping {int(no_readout.sum())} row(s) with no usable readout.")
        out = out[~no_readout]

    sizes = out.groupby("group_id")["variant_id"].transform("size")
    too_small = sizes < min_group_size
    if too_small.any():
        logger.warning(
            f"{source}: dropping {int(too_small.sum())} row(s) in groups smaller than {min_group_size}."
        )
        out = out[~too_small]

    out["source_dataset"] = source
    out["readout_kind"] = readout_kind
    out = out[list(VARIANT_SET_COLUMNS)].reset_index(drop=True)
    logger.info(f"{source}: kept {len(out):,} of {before:,} rows in {out['group_id'].nunique():,} group(s).")
    return out


def available_variant_sets() -> Dict[str, Optional[str]]:
    """
    Each known within gene source against the reason it cannot be loaded.

    None means the source loaded. A string is the reason it did not, suitable
    for printing in the gate report without further handling.
    """
    from src.datasets import cambray, goodman

    status: Dict[str, Optional[str]] = {}
    for module in (cambray, goodman):
        try:
            module.load()
            status[module.SOURCE] = None
        except MissingVariantSource as err:
            status[module.SOURCE] = str(err)
        except Exception as err:  # a malformed file, which is different from an absent one
            status[module.SOURCE] = f"present but unreadable: {err}"
    return status


def load_variant_sets(sources: Sequence[str] = ()) -> List[VariantSet]:
    """
    Every within gene source that is on disk, or the named subset of them.

    Returns an empty list when none is available, which is the current state of
    the working tree and is why Agent C abstains.
    """
    from src.datasets import cambray, goodman

    modules = {cambray.SOURCE: cambray, goodman.SOURCE: goodman}
    wanted = list(sources) if sources else list(modules)
    loaded: List[VariantSet] = []
    for name in wanted:
        module = modules.get(name)
        if module is None:
            raise KeyError(f"Unknown within gene source '{name}'. Known sources: {', '.join(modules)}.")
        try:
            loaded.append(module.load())
        except MissingVariantSource as err:
            logger.info(str(err))
    return loaded
