"""
Build the modelling feature table from the Boel sequence table.

Reads `data/processed/feature_table_v3.csv`, which carries one row per gene with
its expression score, Boel's own folding columns and the expressed DNA sequence,
and writes `data/processed/feature_table_v4.csv` with every feature the
application can compute from a sequence.

Both files are gitignored, so this script is how v4 is reproduced:

    python -m scripts.build_feature_table

Takes roughly fifteen minutes: the folding average evaluates a ViennaRNA fold
for each 48 nt window of each of the 6,348 genes.
"""

import sys
import time
from typing import Dict

import pandas as pd

from src.features import (
    calculate_gc_content,
    calculate_gc3,
    calculate_cai,
    calculate_rare_codon_fraction,
    calculate_folding_energy,
    calculate_genewide_folding_avg,
    calculate_protein_features,
    FOLDING_WINDOW_NT,
)

INPUT_CSV = "data/processed/feature_table_v3.csv"
OUTPUT_CSV = "data/processed/feature_table_v4.csv"

# Every Boel construct ends in a vector-derived C-terminal His-tag,
# CTCGAGCACCACCACCACCACCACTGA. It is stripped before computing composition and
# protein features so that they describe the native gene, matching how the v2
# table was built. The 5' head folding deliberately keeps the sequence as
# expressed, since that is the region the ribosome meets.
HIS_TAG_NT = 27


def features_for(seq: str) -> Dict[str, float]:
    """Every servable feature for one expressed construct sequence."""
    native = seq[:-HIS_TAG_NT] if len(seq) > HIS_TAG_NT else seq
    protein = calculate_protein_features(native)
    return {
        "GC_percent": calculate_gc_content(native),
        "GC3": calculate_gc3(native),
        "CAI": calculate_cai(native),
        "rare_codon_fraction": calculate_rare_codon_fraction(native),
        "rare_codons_head30": calculate_rare_codon_fraction(native, head_codons=30),
        "length_codons": len(native) // 3,
        "head_folding_48": calculate_folding_energy(seq[:FOLDING_WINDOW_NT]),
        "genewide_folding_avg": calculate_genewide_folding_avg(native),
        "gravy_score": protein["gravy_score"],
        "pI": protein["pI"],
    }


def main() -> int:
    source = pd.read_csv(INPUT_CSV)
    if "seq" not in source.columns:
        print(f"'{INPUT_CSV}' has no 'seq' column; nothing to compute from.", file=sys.stderr)
        return 1

    started = time.time()
    computed = pd.DataFrame([features_for(s) for s in source.seq.astype(str)])
    elapsed = time.time() - started

    # Boel's own folding columns are carried through for comparison. They fold
    # the expression vector's 5' UTR, so they are not servable and the deployed
    # model does not use them.
    carried = source[["name", "E", "Gz48v", "Gave48"]].rename(
        columns={"name": "gene_id", "E": "expression_score"}
    )
    table = pd.concat([carried.reset_index(drop=True), computed], axis=1)

    missing = table.isna().sum()
    if missing.any():
        print("Columns with missing values:", missing[missing > 0].to_dict(), file=sys.stderr)
        if table["genewide_folding_avg"].isna().any():
            print("Folding is empty: ViennaRNA failed to load. Install it with "
                  "`pip install ViennaRNA`.", file=sys.stderr)
            return 1

    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}: {len(table)} rows, {table.shape[1]} columns, "
          f"computed in {elapsed / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
