"""
Dataset ingestion, one module per source.

Every source arrives in a different shape and every one of them needs a
decision recorded about what its label means. The rules that apply to all of
them, from the plan's section 8:

- Add a `source_dataset` column before merging anything. Boel and Price 2011
  are both NESG sets from the same pipeline, so they are siblings rather than
  independent samples, and a merged table that cannot tell them apart cannot be
  reasoned about later.
- Cluster the union of every dataset used by any agent, once, at the start, and
  assign every row a cluster id. Never split by row anywhere in the system.
- Decide once whether the modelled object is the tagged construct or the native
  protein, then apply the same rule to every source including Boel. A tag
  changes GRAVY, pI and the 5' folding.

The within gene sources, Cambray and Goodman, are the ones Agent C needs, and
they are the two that invert the dataset survey's earlier verdict. Both were
ruled out as training data for the tier model because every variant is a fusion
to a single sfGFP scaffold, so GRAVY and pI are frozen across the library and
the readout is translation rate rather than soluble yield. For a within gene
ranker both objections become the reason to use them: the protein is held
constant and only the codons vary, which is exactly the comparison the redesign
tab makes, and translation rate is the mechanism a synonymous swap acts on.
"""

from src.datasets.variant_sets import (  # noqa: F401
    VARIANT_SET_COLUMNS,
    VariantSet,
    available_variant_sets,
    load_variant_sets,
)

__all__ = [
    "VARIANT_SET_COLUMNS",
    "VariantSet",
    "available_variant_sets",
    "load_variant_sets",
]
