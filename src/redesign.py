"""
Synonymous-codon redesign: search for a DNA sequence that encodes the same
protein and scores better under the trained expression model.

The model is used as the scoring function and is not retrained. The search is
an adaptive (1 + lambda) hill climb over synonymous codon swaps: each round
mutates the current best sequence in several places, scores every candidate
through the same feature pipeline the predictor uses, and keeps the best if it
improved. That is a local search, not a genetic algorithm with crossover,
because crossover between synonymous variants of one protein recombines
independent codon choices and buys little over mutation alone here.

Two things this engine deliberately does not do.

It does not change the protein. Every swap is synonymous, the start codon and
any terminal stop codon are locked, and the translated product of the result is
checked against the original before it is returned. A violation raises rather
than being reported, because a redesign that quietly alters the protein is
worse than no redesign.

It does not touch GRAVY or the isoelectric point. Those are properties of the
amino acid sequence, so no synonymous change can move them. When the construct
fails on either, the engine says so and recommends a fusion tag instead of
presenting a codon redesign as a remedy.

On what the score means: the objective is the model's expected tier, the
probability-weighted mean over the six tiers, rather than the single most
likely tier. With six classes the most likely tier is a step function, so most
single-codon swaps leave it unchanged and a hill climb on it stalls
immediately; the expected tier moves continuously and gives the search a
gradient. The reported before-and-after tiers are still the most likely tier,
which is what the predictor shows.

The model this scores against currently performs at the majority-class
baseline, so a gain in predicted tier is not evidence of a gain in measured
yield. The caller is expected to surface that, and the interface does.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.features import ECOLI_CODON_W, extract_features
from src.predict import TIER_MAPPING, build_feature_row

logger = logging.getLogger("SynthBio.redesign")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.redesign] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# Thresholds for the two amino-acid-level failure modes, matching the
# diagnostics the interface already applies to the original construct.
GRAVY_INSOLUBLE_ABOVE = 0.4
PI_AGGREGATION_WINDOW = (7.2, 7.8)

# Search budget. Each candidate costs one full feature extraction, which is
# dominated by ViennaRNA folding and so scales with sequence length: around
# 0.15 s for an 840 nt construct. The defaults keep a typical run inside about
# half a minute, which is what an interactive dashboard can carry.
DEFAULT_CANDIDATES_PER_ROUND = 30
DEFAULT_ROUNDS = 8
DEFAULT_PATIENCE = 3

# Share of mutable codons swapped per candidate at the start of the search,
# and the floor it decays to. A large first step explores, and the step halves
# whenever a round fails to improve, so the search ends by trying single swaps.
INITIAL_SWAP_FRACTION = 0.08
MIN_SWAPS = 1


def _codon_families() -> Dict[str, List[str]]:
    """
    Amino acid letter to the list of codons encoding it, standard table.

    Built from Biopython rather than written out, so it cannot drift from the
    table used to translate.
    """
    from Bio.Data.CodonTable import standard_dna_table

    families: Dict[str, List[str]] = {}
    for codon, amino_acid in standard_dna_table.forward_table.items():
        families.setdefault(amino_acid, []).append(codon)
    return families


def _synonym_map() -> Dict[str, List[str]]:
    """Codon to every codon encoding the same amino acid, itself included."""
    synonyms: Dict[str, List[str]] = {}
    for codons in _codon_families().values():
        for codon in codons:
            synonyms[codon] = sorted(codons)
    return synonyms


# Codon to its synonymous alternatives, and the E. coli usage weight each one
# is sampled by. Sampling by usage rather than uniformly keeps the search in
# the region of codon choices E. coli actually makes, instead of wandering into
# combinations no highly expressed gene uses.
SYNONYMS: Dict[str, List[str]] = _synonym_map()
SAMPLING_WEIGHT: Dict[str, float] = {codon: max(usage, 0.01) for codon, usage in ECOLI_CODON_W.items()}


def split_codons(sequence: str) -> List[str]:
    """Split a sequence into in-frame codons, dropping a trailing partial codon."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def translate_codons(codons: Sequence[str]) -> str:
    """
    Amino acid string for a codon list, stop codons rendered as '*'.

    Used as the identity check on a redesign, so an unrecognised codon becomes
    'X' rather than being skipped: a sequence with ambiguity codes must compare
    unequal to a different one, not silently equal.
    """
    from Bio.Data.CodonTable import standard_dna_table

    forward = standard_dna_table.forward_table
    out = []
    for codon in codons:
        if codon in STOP_CODONS:
            out.append("*")
        else:
            out.append(forward.get(codon, "X"))
    return "".join(out)


@dataclass
class ConservedRegionSpec:
    """A parsed conserved-region request: which codons are locked, and why."""

    locked: Set[int]
    regions: List[Tuple[int, int]]
    problems: List[str] = field(default_factory=list)

    @property
    def described(self) -> str:
        """The accepted regions in the user's own 1-based amino acid numbering."""
        if not self.regions:
            return "none"
        return ", ".join(f"{start}-{end}" if start != end else str(start) for start, end in self.regions)


def parse_conserved_regions(text: str, n_codons: int) -> ConservedRegionSpec:
    """
    Parse amino acid indices to hold fixed, as entered in the interface.

    Accepts a comma or semicolon separated list of 1-based inclusive ranges and
    single positions, for example "45-60, 102, 210-215". Positions are amino
    acid numbers, counting the initial methionine as 1, which is how a
    catalytic site is reported in the literature.

    Anything unusable is reported in `problems` rather than raised: a typo in
    one range should not discard the rest, and the interface shows the list.
    """
    spec = ConservedRegionSpec(locked=set(), regions=[])
    if not text or not text.strip():
        return spec

    for chunk in (part.strip() for part in text.replace(";", ",").split(",")):
        if not chunk:
            continue
        try:
            if "-" in chunk.lstrip("-"):
                low_text, _, high_text = chunk.partition("-")
                start, end = int(low_text.strip()), int(high_text.strip())
            else:
                start = end = int(chunk)
        except ValueError:
            spec.problems.append(f"'{chunk}' is not a position or a range like 45-60, so it was ignored.")
            continue

        if start > end:
            start, end = end, start
        if end < 1:
            spec.problems.append(f"'{chunk}' is below the first amino acid, so it was ignored.")
            continue
        if start > n_codons:
            spec.problems.append(
                f"'{chunk}' lies past the end of the protein, which is {n_codons} amino acids long, "
                "so it was ignored."
            )
            continue

        clipped_start, clipped_end = max(start, 1), min(end, n_codons)
        if (clipped_start, clipped_end) != (start, end):
            spec.problems.append(
                f"'{chunk}' was clipped to {clipped_start}-{clipped_end}, the extent of the protein."
            )
        spec.regions.append((clipped_start, clipped_end))
        spec.locked.update(range(clipped_start - 1, clipped_end))

    spec.regions.sort()
    return spec


def mutable_positions(codons: Sequence[str], conserved: Set[int]) -> List[int]:
    """
    Codon indices the search may change.

    Excluded: the start codon, any terminal stop codon, the conserved region,
    codons with no synonymous alternative (methionine and tryptophan) and
    codons the table does not recognise, which includes anything carrying an
    ambiguity code.
    """
    positions = []
    last = len(codons) - 1
    for index, codon in enumerate(codons):
        if index == 0:
            continue
        if index == last and codon in STOP_CODONS:
            continue
        if index in conserved:
            continue
        if codon in STOP_CODONS:
            # An internal stop codon is a defect in the input, flagged by
            # validation. Leave it exactly as supplied.
            continue
        if len(SYNONYMS.get(codon, [])) < 2:
            continue
        positions.append(index)
    return positions


def _swap_codon(codon: str, rng: random.Random) -> str:
    """One synonymous replacement, sampled by E. coli usage, never the same codon."""
    alternatives = [c for c in SYNONYMS.get(codon, []) if c != codon]
    if not alternatives:
        return codon
    weights = [SAMPLING_WEIGHT.get(c, 0.01) for c in alternatives]
    return rng.choices(alternatives, weights=weights, k=1)[0]


def mutate(
    codons: Sequence[str],
    positions: Sequence[int],
    n_swaps: int,
    rng: random.Random,
) -> List[str]:
    """Copy the codon list with `n_swaps` randomly chosen positions swapped."""
    out = list(codons)
    if not positions:
        return out
    chosen = rng.sample(list(positions), k=min(n_swaps, len(positions)))
    for index in chosen:
        out[index] = _swap_codon(out[index], rng)
    return out


def generate_variants(
    sequence: str,
    n_variants: int = 100,
    conserved_text: str = "",
    n_swaps: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    A library of synonymous variants of one construct, for inspection or testing.

    This is the mutation generator on its own, without the scoring loop: it is
    what Step 1 of the brief asks for and is useful for checking that the
    conserved-region lock holds before any model is involved.
    """
    codons = split_codons(sequence)
    spec = parse_conserved_regions(conserved_text, len(codons))
    positions = mutable_positions(codons, spec.locked)
    rng = random.Random(seed)
    swaps = n_swaps if n_swaps is not None else max(MIN_SWAPS, int(len(positions) * INITIAL_SWAP_FRACTION))
    return ["".join(mutate(codons, positions, swaps, rng)) for _ in range(n_variants)]


@dataclass
class TagRecommendation:
    """A fusion-tag suggestion for a failure synonymous changes cannot address."""

    driver: str
    detail: str
    recommendation: str


def recommend_fusion_tags(features: Dict[str, float]) -> List[TagRecommendation]:
    """
    Fusion tags for the amino-acid-level failure modes, where they apply.

    GRAVY and pI are computed from the translated product, so a synonymous
    redesign cannot move either by construction. Where one of them is the
    problem, the remedy is to change the protein, which in practice means
    appending a tag, and the engine says so rather than offering codon swaps.
    """
    out: List[TagRecommendation] = []

    gravy = features.get("gravy_score")
    if gravy is not None and not _is_missing(gravy) and gravy > GRAVY_INSOLUBLE_ABOVE:
        out.append(TagRecommendation(
            driver="GRAVY (hydrophobicity)",
            detail=(
                f"GRAVY is {gravy}, above the {GRAVY_INSOLUBLE_ABOVE} threshold, so the product is "
                "hydrophobic enough to partition into inclusion bodies. Hydropathy is a property of "
                "the amino acids, so no synonymous codon change can alter it."
            ),
            recommendation=(
                "Append an N-terminal MBP (maltose binding protein) or GST solubility tag, and keep a "
                "protease site so the tag can be removed after purification."
            ),
        ))

    pi_value = features.get("pI")
    low, high = PI_AGGREGATION_WINDOW
    if pi_value is not None and not _is_missing(pi_value) and low <= pi_value <= high:
        out.append(TagRecommendation(
            driver="Isoelectric point (pI)",
            detail=(
                f"The pI of {pi_value} falls inside the E. coli cytoplasmic pH range ({low} to {high}), "
                "so the product carries close to no net charge under expression conditions and is prone "
                "to aggregation. The pI follows from the amino acid composition, so a synonymous "
                "redesign cannot shift it."
            ),
            recommendation=(
                "Append a charged fusion tag to move the pI out of this window. An acidic tag such as "
                "MBP shifts it down; a polyhistidine or other basic tag shifts it up."
            ),
        ))

    return out


def _is_missing(value: Any) -> bool:
    """True for None and NaN, which both reach here from the feature extractor."""
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


@dataclass
class ScoredSequence:
    """One candidate: its sequence, its features and what the model made of it."""

    sequence: str
    features: Dict[str, float]
    expected_tier: float
    tier_class: int
    tier_label: str
    confidence: float
    probabilities: Dict[int, float]


@dataclass
class RedesignResult:
    """The outcome of a redesign run, with everything the interface reports."""

    original: ScoredSequence
    best: ScoredSequence
    history: List[float]
    candidates_scored: int
    codons_changed: int
    codons_mutable: int
    conserved: ConservedRegionSpec
    tag_recommendations: List[TagRecommendation]
    improved: bool
    protein: str

    def corroboration(self) -> "Corroboration":
        """Whether an independent sequence measure agrees with the model's gain."""
        return corroborate(self)

    @property
    def expected_tier_gain(self) -> float:
        return round(self.best.expected_tier - self.original.expected_tier, 3)

    def feature_comparison(self) -> pd.DataFrame:
        """Original against redesigned, for the features a codon swap can move."""
        rows = [
            ("GC content (%)", "gc_content"),
            ("GC3, wobble position (%)", "gc3_content"),
            ("Codon adaptation index", "codon_adaptation_index"),
            ("Rare codon fraction", "rare_codon_fraction"),
            ("Rare codons, first 30", "rare_codons_head30"),
            ("5' folding, 48 nt (kcal/mol)", "head_folding_48"),
            ("Folding energy, gene-wide (kcal/mol)", "genewide_folding_avg"),
        ]
        data = []
        for label, key in rows:
            before = self.original.features.get(key)
            after = self.best.features.get(key)
            if _is_missing(before) or _is_missing(after):
                change = None
            else:
                change = round(float(after) - float(before), 3)
            data.append({
                "Feature": label,
                "Original": before,
                "Redesigned": after,
                "Change": change,
            })
        return pd.DataFrame(data)

    def changed_codon_table(self) -> pd.DataFrame:
        """Every codon the redesign changed, in amino acid numbering."""
        before = split_codons(self.original.sequence)
        after = split_codons(self.best.sequence)
        rows = []
        for index, (old, new) in enumerate(zip(before, after)):
            if old != new:
                rows.append({
                    "Amino acid position": index + 1,
                    "Residue": translate_codons([old]),
                    "Original codon": old,
                    "Redesigned codon": new,
                })
        return pd.DataFrame(rows)


# How the objective behaved on the one dataset that can test it: Welch et al.
# 2009 series of synonymous variants of one protein, with measured yield. Rank
# correlation of the objective against yield, within series, from
# scripts/redesign_sanity_check.py. GC3 is given alongside because it is
# positive in both series while the objective is not.
WELCH_CALIBRATION = [
    {"series": "polymerase", "n": 30, "objective": -0.408, "gc3": 0.643},
    {"series": "scFv", "n": 32, "objective": 0.469, "gc3": 0.394},
]

# Verdicts for the corroboration check below.
CORROBORATED = "corroborated"
CONTRADICTED = "contradicted"
INCONCLUSIVE = "inconclusive"


@dataclass
class Corroboration:
    """
    Whether an independent sequence measure agrees with the model's verdict.

    The objective is the model's expected tier, and on the Welch series that
    objective tracks measured yield in one series and runs against it in the
    other. GC3 tracks yield positively in both. So when a redesign raises the
    model's score while lowering GC3, the two signals disagree and the result
    should not be presented as an improvement. This is reported rather than
    used to override the objective, because the brief asks for the model to be
    the judge.
    """

    verdict: str
    gc3_change: Optional[float]
    head_folding_change: Optional[float]
    cai_change: Optional[float]
    summary: str


def corroborate(result: "RedesignResult") -> Corroboration:
    """Check the model's gain against GC3, the strongest single yield predictor here."""

    def delta(key: str) -> Optional[float]:
        before, after = result.original.features.get(key), result.best.features.get(key)
        if _is_missing(before) or _is_missing(after):
            return None
        return round(float(after) - float(before), 3)

    gc3 = delta("gc3_content")
    head = delta("head_folding_48")
    cai = delta("codon_adaptation_index")

    if not result.improved:
        return Corroboration(
            INCONCLUSIVE, gc3, head, cai,
            "The search did not raise the model's score, so there is nothing to corroborate.",
        )
    if gc3 is None:
        return Corroboration(
            INCONCLUSIVE, gc3, head, cai,
            "GC3 could not be computed for one of the two sequences, so the check did not run.",
        )
    if gc3 > 0:
        return Corroboration(
            CORROBORATED, gc3, head, cai,
            f"GC3 rose by {gc3:+.2f} points alongside the model's score. On the Welch synonymous "
            "variants GC3 tracks measured yield positively in both series, so an independent "
            "sequence measure agrees with the model here.",
        )
    if gc3 < 0:
        return Corroboration(
            CONTRADICTED, gc3, head, cai,
            f"GC3 fell by {gc3:.2f} points while the model's score rose. On the Welch synonymous "
            "variants GC3 tracks measured yield positively in both series, so the two signals "
            "disagree and this redesign is not supported by the sequence evidence.",
        )
    return Corroboration(
        INCONCLUSIVE, gc3, head, cai,
        "GC3 did not move, so it says nothing either way about this redesign.",
    )


def _score_batch(sequences: Sequence[str], model: Any) -> List[ScoredSequence]:
    """
    Score candidates through the predictor's own feature path and the model.

    One `predict_proba` call covers the whole batch, so the cost of a round is
    the feature extractions, not the model.
    """
    feature_dicts = [extract_features(seq) for seq in sequences]
    frame = pd.concat([build_feature_row(f) for f in feature_dicts], ignore_index=True)

    model_cols = list(getattr(model, "feature_names_in_", frame.columns))
    missing = [c for c in model_cols if c not in frame.columns]
    if missing:
        raise ValueError(
            "The loaded model needs features this pipeline cannot compute from a sequence: "
            + ", ".join(missing)
        )

    probabilities = model.predict_proba(frame[model_cols])
    classes = [int(c) for c in getattr(model, "classes_", range(probabilities.shape[1]))]

    scored = []
    for seq, feats, row in zip(sequences, feature_dicts, probabilities):
        per_class = {c: float(p) for c, p in zip(classes, row)}
        expected = float(sum(c * p for c, p in per_class.items()))
        tier_class = max(per_class, key=per_class.get)
        scored.append(ScoredSequence(
            sequence=seq,
            features=feats,
            expected_tier=expected,
            tier_class=tier_class,
            tier_label=TIER_MAPPING.get(tier_class, f"Tier {tier_class}"),
            confidence=per_class[tier_class],
            probabilities=per_class,
        ))
    return scored


def optimize_sequence(
    sequence: str,
    model: Any,
    conserved_text: str = "",
    candidates_per_round: int = DEFAULT_CANDIDATES_PER_ROUND,
    rounds: int = DEFAULT_ROUNDS,
    patience: int = DEFAULT_PATIENCE,
    seed: Optional[int] = 42,
    progress: Optional[Any] = None,
) -> RedesignResult:
    """
    Search for a synonymous redesign of `sequence` that the model scores higher.

    `model` must be a fitted classifier over the servable feature columns; the
    heuristic fallback has no tier to optimise against, so the caller is
    expected to refuse the run without a trained artifact.

    `progress` is an optional callable taking (round_index, rounds, best score),
    which the interface uses to drive a progress bar.

    Raises ValueError for a sequence that cannot be redesigned, and
    RuntimeError if a candidate ever translates differently from the original,
    which would be a defect in the mutation operator.
    """
    codons = split_codons(sequence)
    if len(codons) < 2:
        raise ValueError("A redesign needs a coding sequence of at least two codons.")

    protein = translate_codons(codons)
    spec = parse_conserved_regions(conserved_text, len(codons))
    positions = mutable_positions(codons, spec.locked)

    if not positions:
        raise ValueError(
            "No codon in this construct can be changed synonymously. Every position is either "
            "locked, a single-codon amino acid, or outside the standard codon table."
        )

    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError(
            "A redesign scores candidates with the trained model, so it cannot run in heuristic mode."
        )

    logger.info(
        f"Redesign starting: {len(codons)} codons, {len(positions)} mutable, "
        f"{len(spec.locked)} conserved, budget {rounds} rounds x {candidates_per_round} candidates."
    )

    rng = random.Random(seed)
    original = _score_batch([sequence], model)[0]
    best = original
    history = [original.expected_tier]
    candidates_scored = 1
    n_swaps = max(MIN_SWAPS, int(len(positions) * INITIAL_SWAP_FRACTION))
    stalled = 0

    for round_index in range(rounds):
        best_codons = split_codons(best.sequence)
        batch = ["".join(mutate(best_codons, positions, n_swaps, rng)) for _ in range(candidates_per_round)]
        scored = _score_batch(batch, model)
        candidates_scored += len(scored)

        for candidate in scored:
            # The invariant the whole engine rests on. Checked on every
            # candidate rather than only on the winner, so a defect in the
            # mutation operator surfaces immediately.
            if translate_codons(split_codons(candidate.sequence)) != protein:
                raise RuntimeError(
                    "A candidate did not encode the original protein. The redesign was abandoned."
                )

        round_best = max(scored, key=lambda s: s.expected_tier)
        if round_best.expected_tier > best.expected_tier:
            logger.info(
                f"Round {round_index + 1}: expected tier {best.expected_tier:.3f} -> "
                f"{round_best.expected_tier:.3f} with {n_swaps} swaps per candidate."
            )
            best = round_best
            stalled = 0
        else:
            stalled += 1
            # A failed round means the step was too large for the local
            # landscape, so halve it and try a finer search.
            n_swaps = max(MIN_SWAPS, n_swaps // 2)
            logger.info(
                f"Round {round_index + 1}: no improvement on {best.expected_tier:.3f}; "
                f"step reduced to {n_swaps} swaps."
            )

        history.append(best.expected_tier)
        if progress is not None:
            progress(round_index + 1, rounds, best.expected_tier)

        if stalled >= patience and n_swaps <= MIN_SWAPS:
            logger.info(f"Redesign stopping early after {stalled} rounds without improvement.")
            break

    changed = sum(1 for a, b in zip(split_codons(original.sequence), split_codons(best.sequence)) if a != b)
    result = RedesignResult(
        original=original,
        best=best,
        history=history,
        candidates_scored=candidates_scored,
        codons_changed=changed,
        codons_mutable=len(positions),
        conserved=spec,
        tag_recommendations=recommend_fusion_tags(original.features),
        improved=best.expected_tier > original.expected_tier,
        protein=protein,
    )
    logger.info(
        f"Redesign finished: {candidates_scored} candidates scored, {changed} codons changed, "
        f"expected tier {original.expected_tier:.3f} -> {best.expected_tier:.3f} "
        f"(tier {original.tier_class} -> {best.tier_class})."
    )
    return result
