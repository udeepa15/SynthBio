"""
Expression vector 5' leaders, and the transcript they make with an insert.

This module exists because of a specific gap. `feature_table_v4.csv` carries
Boel's own RNAstructure energies, Gz48v and Gave48, computed over the
expression vector's 5' untranslated region. They are the strongest features in
that table and the pipeline deliberately excludes them from the served model,
because the application cannot compute them from a pasted coding sequence: it
does not know which vector the sequence is going into. A model fitted on them
would load and then never run.

The information is not missing from biology. It is missing from the input. Ask
which vector the construct is going into and the leader becomes known, the
transcript's true initiation window can be assembled from leader plus insert,
and the strongest feature in the Boel table becomes servable.

On the leaders below. A ribosome does not read the plasmid, it reads the
transcript, so what matters is the sequence from the transcription start to the
start codon: promoter remnant, ribosome binding site and spacer. The pET entry
is the 71 nt T7 leader published in Welch et al. 2009, Supporting Table 1, as
the sequence shared by all 81 of their constructs. It is used here as the
default because it is a measured sequence from this project's own validation
data rather than one copied from a vector map.

A user whose vector is not listed can supply the leader themselves, which is
the honest option: a wrong leader produces a confident folding number with
nothing behind it, exactly the failure this module is meant to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# The leader of a pET style T7 expression construct, from the transcription
# start to the base before the start codon. Source: Welch M et al., PLoS ONE
# 4(9):e7002, Supporting Table 1, row "5'-term RNA", the single leader shared by
# all 81 constructs in that study.
PET_T7_LEADER = (
    "GGGGAATTGTGAGCGGATAACAATTCCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGATATTACC"
)

NO_VECTOR = "none"
CUSTOM = "custom"
DEFAULT_VECTOR = "pET"


@dataclass(frozen=True)
class Vector:
    """One expression vector, described by the leader it puts on the transcript."""

    key: str
    label: str
    leader: str
    source: str

    @property
    def leader_length(self) -> int:
        return len(self.leader)


VECTORS: Dict[str, Vector] = {
    DEFAULT_VECTOR: Vector(
        key=DEFAULT_VECTOR,
        label="pET, T7 promoter and lac operator",
        leader=PET_T7_LEADER,
        source=(
            "Welch et al. 2009, PLoS ONE 4(9):e7002, Supporting Table 1, the 71 nt leader shared "
            "by all 81 constructs."
        ),
    ),
    NO_VECTOR: Vector(
        key=NO_VECTOR,
        label="Not declared",
        leader="",
        source="No vector declared, so no leader is assumed.",
    ),
}


def get(key: Optional[str]) -> Vector:
    """The named vector, or the undeclared one."""
    return VECTORS.get(key or NO_VECTOR, VECTORS[NO_VECTOR])


def choices() -> List[str]:
    """Vector keys for the interface, the default first and 'not declared' last."""
    ordered = [DEFAULT_VECTOR] + [k for k in VECTORS if k not in (DEFAULT_VECTOR, NO_VECTOR)]
    return ordered + [NO_VECTOR]


def resolve_leader(vector: Optional[str], custom_leader: Optional[str] = None) -> str:
    """
    The 5' leader to use, from a named vector or a leader the user supplied.

    A supplied leader wins over a named vector, because someone who types a
    leader has a reason to. It is cleaned to ACGU/ACGT and uppercased; anything
    else would reach ViennaRNA and fail there instead of here.
    """
    if custom_leader:
        cleaned = "".join(c for c in custom_leader.upper() if c in "ACGTU").replace("U", "T")
        if cleaned:
            return cleaned
    return get(vector).leader
