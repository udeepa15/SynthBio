"""
Input sequence validation for SynthBio.

Checks a candidate DNA construct before any feature is computed, so that
malformed input produces an explicit, actionable message rather than silently
reaching the feature extractor and returning NaN for the protein features.

Each problem is reported as a SequenceIssue carrying a one-sentence
explanation and a one-sentence remedy. Issues are either blocking (scoring is
refused) or advisory (scoring proceeds, with the caveat shown).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

from src.features import FOLDING_WINDOW_NT

logger = logging.getLogger("SynthBio.validation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.validation] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

UNAMBIGUOUS_BASES = set("ACGT")
# IUPAC nucleotide ambiguity codes, accepted but flagged: the codon table and
# the translation step cannot resolve them.
AMBIGUITY_CODES = set("RYSWKMBDHVN")
START_CODON = "ATG"
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Above this length the input is a genome, plasmid or contig rather than one
# coding sequence. The longest E. coli CDS is around 10 kb, so 30 kb leaves
# generous room for an unusual construct while still catching a whole-genome
# accession, which otherwise arrives as a frame error, a missing start codon and
# tens of thousands of internal stops, none of which name the actual problem.
MAX_CDS_NT = 30_000

# FOLDING_WINDOW_NT is imported from features: below that length the folding
# proxy is computed over the whole construct rather than the 5' window it is
# defined over, so the feature describes a different region than it claims to.


@dataclass
class SequenceIssue:
    """One validation finding: what is wrong, and what to do about it."""

    code: str
    title: str
    detail: str
    remedy: str
    blocking: bool


def normalise_sequence(raw: str) -> str:
    """Strip whitespace, FASTA header lines and digits; upper-case the rest."""
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")]
    joined = "".join(lines)
    return "".join(ch for ch in joined.upper() if not ch.isspace() and not ch.isdigit())


def _internal_stop_positions(seq: str) -> List[int]:
    """Codon indices (1-based) of in-frame stop codons before the final codon."""
    codon_count = len(seq) // 3
    positions = []
    for i in range(codon_count - 1):
        if seq[i * 3:i * 3 + 3] in STOP_CODONS:
            positions.append(i + 1)
    return positions


def validate_sequence(raw: str) -> Tuple[str, List[SequenceIssue]]:
    """
    Validate a raw sequence string.

    Returns the normalised sequence and the list of issues found. Scoring
    should proceed only when no issue has blocking=True.
    """
    seq = normalise_sequence(raw)
    issues: List[SequenceIssue] = []

    if not seq:
        issues.append(SequenceIssue(
            code="empty",
            title="No sequence provided",
            detail="The input is empty once whitespace and FASTA header lines are removed.",
            remedy="Paste a coding DNA sequence, or fetch one by NCBI accession on the other tab.",
            blocking=True,
        ))
        return seq, issues

    # Far too long to be one coding sequence. Reported alone, because every
    # later check would also fire and bury the one finding that explains them.
    if len(seq) > MAX_CDS_NT:
        issues.append(SequenceIssue(
            code="not_a_coding_sequence",
            title="This looks like a genome or plasmid, not a single gene",
            detail=(
                f"The input is {len(seq):,} nt. A coding sequence for E. coli expression is "
                f"typically a few hundred to a few thousand, and the longest known is around "
                f"10,000. A whole-genome or plasmid accession is the usual cause."
            ),
            remedy=(
                "Supply one gene's coding sequence, from its ATG to its stop codon. On NCBI, "
                "open the genome record's annotation and use the CDS entry for the gene you "
                "want, rather than the genome accession itself."
            ),
            blocking=True,
        ))
        return seq, issues

    # Characters outside the nucleotide alphabet.
    invalid = sorted({ch for ch in seq if ch not in UNAMBIGUOUS_BASES and ch not in AMBIGUITY_CODES})
    if invalid:
        shown = ", ".join(f"'{ch}'" for ch in invalid[:6])
        issues.append(SequenceIssue(
            code="invalid_characters",
            title="Sequence contains non-nucleotide characters",
            detail=f"Found {shown} in the input, which is not a DNA base or an IUPAC ambiguity code.",
            remedy="Remove the offending characters; a protein sequence or a spreadsheet cell pasted by mistake is the usual cause.",
            blocking=True,
        ))

    # Ambiguity codes: tolerated, but they distort CAI and block translation.
    ambiguous = sorted({ch for ch in seq if ch in AMBIGUITY_CODES})
    if ambiguous:
        issues.append(SequenceIssue(
            code="ambiguity_codes",
            title="Sequence contains ambiguity codes",
            detail=f"Found {', '.join(ambiguous)}, which no codon in the E. coli adaptiveness table matches, so those codons are skipped by CAI and stop translation.",
            remedy="Resolve the ambiguous positions to a single base before relying on CAI, GRAVY or pI.",
            blocking=False,
        ))

    # Reading frame.
    if len(seq) % 3 != 0:
        issues.append(SequenceIssue(
            code="frame",
            title="Length is not a multiple of three",
            detail=f"The sequence is {len(seq)} nt, which leaves {len(seq) % 3} nt outside the final codon and shifts the reading frame.",
            remedy="Trim to a whole number of codons, or check that you pasted the coding region rather than a genomic fragment.",
            blocking=True,
        ))

    # Start codon.
    if not seq.startswith(START_CODON):
        issues.append(SequenceIssue(
            code="start_codon",
            title="Sequence does not begin with ATG",
            detail=f"The first codon is '{seq[:3]}', so the translated protein and therefore GRAVY and pI will not correspond to the intended product.",
            remedy="Supply the coding sequence from its start codon, excluding any 5' UTR or promoter region.",
            blocking=True,
        ))

    # Internal stop codons.
    stops = _internal_stop_positions(seq)
    if stops:
        shown = ", ".join(str(p) for p in stops[:5])
        more = f" and {len(stops) - 5} more" if len(stops) > 5 else ""
        issues.append(SequenceIssue(
            code="internal_stop",
            title="In-frame stop codon before the end of the sequence",
            detail=f"A stop codon occurs at codon {shown}{more}, so translation terminates early and the protein features describe only the truncated product.",
            remedy="Check the reading frame, or remove the premature stop if it came from a sequencing error.",
            blocking=True,
        ))

    # Folding window.
    if len(seq) < FOLDING_WINDOW_NT:
        issues.append(SequenceIssue(
            code="short_for_folding",
            title=f"Sequence is shorter than the {FOLDING_WINDOW_NT} nt folding window",
            detail=f"The sequence is {len(seq)} nt, so the 5' folding proxy is computed over the whole construct rather than the first {FOLDING_WINDOW_NT} nt it is defined for.",
            remedy=f"Supply at least {FOLDING_WINDOW_NT} nt from the start codon for the folding figure to mean what it says.",
            blocking=True,
        ))

    logger.info(
        f"Validated {len(seq)} nt sequence: "
        f"{sum(1 for i in issues if i.blocking)} blocking, {sum(1 for i in issues if not i.blocking)} advisory."
    )
    return seq, issues


def blocking_issues(issues: List[SequenceIssue]) -> List[SequenceIssue]:
    return [i for i in issues if i.blocking]


def advisory_issues(issues: List[SequenceIssue]) -> List[SequenceIssue]:
    return [i for i in issues if not i.blocking]


def is_scorable(issues: List[SequenceIssue]) -> bool:
    return not blocking_issues(issues)


def issue_summary(issues: List[SequenceIssue]) -> Dict[str, int]:
    return {
        "blocking": len(blocking_issues(issues)),
        "advisory": len(advisory_issues(issues)),
    }
