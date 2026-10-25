"""
Does the redesign engine climb a hill that corresponds to measured yield?

The engine optimises the model's expected tier over synonymous codon swaps.
Welch et al. 2009 measured yield for exactly that kind of variation: series of
synonymous variants of one protein, so the protein, the vector and the host are
held fixed and only codon choice differs. That makes it the one dataset in the
project that can tell whether the objective points the right way.

Two checks are reported per series.

1. Rank correlation between the objective and measured yield. A positive
   figure means a candidate the engine prefers did yield more at the bench. A
   negative figure means the engine is climbing away from yield.
2. What a redesign of the worst variant in the series actually does to GC3,
   which is the strongest single sequence predictor of yield here. If the
   objective is unreliable, GC3 is the corroborating signal to watch.

Run: python scripts/redesign_sanity_check.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.stats import spearmanr

from src.model import load_model
from src.redesign import _score_batch, optimize_sequence

MODEL_PATH = os.path.join("models", "baseline_model.joblib")
WELCH_PATH = os.path.join("data", "validation", "welch2009.csv")
MIN_SERIES_SIZE = 5


def main() -> None:
    logging.disable(logging.INFO)

    model = load_model(MODEL_PATH)
    welch = pd.read_csv(WELCH_PATH)
    welch = welch[welch["yield_ug_per_ml"].notna()]

    scored = _score_batch(list(welch["orf_dna"]), model)
    welch = welch.assign(
        expected_tier=[s.expected_tier for s in scored],
        argmax_tier=[s.tier_class for s in scored],
        gc3=[s.features["gc3_content"] for s in scored],
        gc=[s.features["gc_content"] for s in scored],
    )

    print(f"Welch 2009: {len(welch)} variants with measured yield across "
          f"{welch['series'].nunique()} series.\n")
    print("Rank correlation against measured yield, within series")
    print(f"{'series':12s} {'n':>4s} {'objective':>10s} {'argmax tier':>12s} {'GC3':>8s} {'GC%':>8s}")

    for name, group in welch.groupby("series"):
        if len(group) < MIN_SERIES_SIZE:
            continue
        yields = group["yield_ug_per_ml"]
        print(
            f"{name:12s} {len(group):4d} "
            f"{spearmanr(group.expected_tier, yields).statistic:+10.3f} "
            f"{spearmanr(group.argmax_tier, yields).statistic:+12.3f} "
            f"{spearmanr(group.gc3, yields).statistic:+8.3f} "
            f"{spearmanr(group.gc, yields).statistic:+8.3f}"
        )

    print("\nRedesigning the lowest-yielding variant of each series")
    for name, group in welch.groupby("series"):
        if len(group) < MIN_SERIES_SIZE:
            continue
        worst = group.loc[group["yield_ug_per_ml"].idxmin()]
        best_measured = group.loc[group["yield_ug_per_ml"].idxmax()]
        result = optimize_sequence(worst["orf_dna"], model, rounds=6, candidates_per_round=24)

        gc3_before = result.original.features["gc3_content"]
        gc3_after = result.best.features["gc3_content"]
        print(
            f"\n{name}: worst variant {worst['variant_id']} "
            f"({worst['yield_ug_per_ml']:.3g} ug/ml), best measured {best_measured['variant_id']} "
            f"({best_measured['yield_ug_per_ml']:.3g} ug/ml)"
        )
        print(
            f"  objective {result.original.expected_tier:.3f} -> {result.best.expected_tier:.3f}, "
            f"tier {result.original.tier_class} -> {result.best.tier_class}, "
            f"{result.codons_changed} codons changed"
        )
        print(
            f"  GC3 {gc3_before:.2f} -> {gc3_after:.2f} "
            f"({gc3_after - gc3_before:+.2f}); best measured variant sits at "
            f"{_gc3_of(best_measured['orf_dna'], model):.2f}"
        )


def _gc3_of(sequence: str, model) -> float:
    return _score_batch([sequence], model)[0].features["gc3_content"]


if __name__ == "__main__":
    main()
