"""
Feature engineering module for computing construct biophysical & protein features:
1. GC Content (%)
2. Codon Adaptation Index (CAI) for E. coli
3. mRNA folding stability score (MFE delta G)
4. GRAVY Score (Hydrophobicity)
5. Isoelectric Point (pI)
"""

import math
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

# Configure logger
logger = logging.getLogger("SynthBio.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.features] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# E. coli K-12 codon usage relative adaptiveness (simplified representative table)
ECOLI_CODON_W: Dict[str, float] = {
    'TTT': 0.58, 'TTC': 1.00, 'TTA': 0.14, 'TTG': 0.13,
    'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 1.00,
    'ATT': 0.49, 'ATC': 1.00, 'ATA': 0.08, 'ATG': 1.00,
    'GTT': 0.54, 'GTC': 0.38, 'GTA': 0.23, 'GTG': 1.00,
    'TCT': 0.86, 'TCC': 0.74, 'TCA': 0.14, 'TCG': 0.14,
    'CCT': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 1.00,
    'ACT': 0.89, 'ACC': 1.00, 'ACA': 0.14, 'ACG': 0.25,
    'GCT': 1.00, 'GCC': 0.26, 'GCA': 0.58, 'GCG': 0.38,
    'TAT': 0.43, 'TAC': 1.00, 'TAA': 1.00, 'TAG': 0.08,
    'CAT': 0.57, 'CAC': 1.00, 'CAA': 0.34, 'CAG': 1.00,
    'AAT': 0.45, 'AAC': 1.00, 'AAA': 1.00, 'AAG': 0.25,
    'GAT': 0.63, 'GAC': 1.00, 'GAA': 1.00, 'GAG': 0.31,
    'TGT': 0.46, 'TGC': 1.00, 'TGA': 0.30, 'TGG': 1.00,
    'CGT': 1.00, 'CGC': 0.64, 'CGA': 0.06, 'CGG': 0.08,
    'AGT': 0.16, 'AGC': 0.41, 'AGA': 0.04, 'AGG': 0.02,
    'GGT': 1.00, 'GGC': 0.72, 'GGA': 0.13, 'GGG': 0.15,
}

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC percentage of a DNA/RNA sequence."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if not seq:
        logger.warning("Empty sequence provided for GC calculation.")
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    gc = round((gc_count / len(seq)) * 100.0, 2)
    return gc


def _relative_adaptiveness(table: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise a codon usage table into Sharp & Li relative adaptiveness weights.

    Each codon's weight is its usage divided by that of the most-used synonymous
    codon for the same amino acid, so every family's best codon is 1.0. The
    table above is almost in that form already, but the serine family peaks at
    0.86, which depressed CAI for every serine-containing construct.

    Stop codons and the two single-codon families (Met, Trp) are dropped: they
    have no synonymous alternative, so they carry no information about codon
    choice and Sharp & Li exclude them.
    """
    try:
        from Bio.Data.CodonTable import standard_dna_table
        amino_acid_of = standard_dna_table.forward_table
    except ImportError:  # pragma: no cover - Biopython is a hard dependency
        logger.warning("Biopython unavailable; CAI weights used without family normalisation.")
        return {c: w for c, w in table.items() if w > 0}

    families: Dict[str, Dict[str, float]] = {}
    for codon, usage in table.items():
        amino_acid = amino_acid_of.get(codon)
        if amino_acid is None:  # stop codon
            continue
        families.setdefault(amino_acid, {})[codon] = usage

    weights: Dict[str, float] = {}
    for amino_acid, codons in families.items():
        if len(codons) < 2:  # Met (ATG), Trp (TGG): no synonymous choice
            continue
        peak = max(codons.values())
        if peak <= 0:
            continue
        for codon, usage in codons.items():
            if usage > 0:
                weights[codon] = usage / peak
    return weights


# Relative adaptiveness weights actually used for CAI, derived from the usage
# table above. Synonymous families only, each normalised to a peak of 1.0.
ECOLI_CODON_WEIGHTS: Dict[str, float] = _relative_adaptiveness(ECOLI_CODON_W)


def calculate_cai(sequence: str, codon_table: Dict[str, float] = None) -> float:
    """
    Calculate Codon Adaptation Index (CAI) for an open reading frame sequence.

    The geometric mean of each codon's relative adaptiveness weight, per Sharp &
    Li (1987). Codons with no synonymous alternative and stop codons do not
    enter the mean.
    """
    codon_table = ECOLI_CODON_WEIGHTS if codon_table is None else codon_table
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if len(seq) < 3:
        logger.warning(f"Sequence length ({len(seq)} bp) too short for CAI codon analysis.")
        return 0.0

    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    valid_w_values = []

    for codon in codons:
        if codon in codon_table and codon_table[codon] > 0:
            valid_w_values.append(math.log(codon_table[codon]))

    if not valid_w_values:
        logger.warning("No valid codons found matching E. coli CAI adaptiveness table.")
        return 0.0

    cai = round(math.exp(sum(valid_w_values) / len(valid_w_values)), 3)
    return cai


# Length of the 5' region the folding proxy is defined over. Below this length
# the proxy is computed over whatever is present, which is why validation
# refuses shorter sequences.
FOLDING_WINDOW_NT = 48


def estimate_mrna_folding_stability(sequence: str) -> float:
    """
    GC-derived proxy for 5' mRNA secondary structure, reported in kcal/mol.

    This is not a thermodynamic minimum free energy: no folding model is
    evaluated. The value scales with the GC fraction of the first
    FOLDING_WINDOW_NT nucleotides, so a more negative figure means a GC-richer
    5' region, taken to imply tighter folding and obstructed ribosome binding.
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if not seq:
        logger.warning("Empty sequence provided for MFE folding stability calculation.")
        return 0.0
    
    region_5prime = seq[:FOLDING_WINDOW_NT]
    gc_fraction = (region_5prime.count("G") + region_5prime.count("C")) / max(len(region_5prime), 1)
    mfe = -1.5 * len(region_5prime) * (gc_fraction ** 1.5)
    return round(mfe, 2)


def calculate_gc3(sequence: str) -> float:
    """
    GC fraction at the third, wobble position of each codon, as a percentage.

    The third position is where synonymous choice mostly lives, so GC3 measures
    codon composition while being almost independent of the protein. On Welch's
    synonymous variant series, where the protein is held fixed, it is the
    strongest single predictor of measured yield that this module computes
    (Spearman 0.52 within series, against 0.46 for overall GC%).
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    third = seq[2::3]
    if not third:
        logger.warning("Sequence too short for GC3.")
        return 0.0
    return round(100.0 * sum(1 for base in third if base in "GC") / len(third), 2)


# Codons read slowly in E. coli, by low cognate tRNA abundance. A moderate
# share of these is associated with *higher* yield, not lower: pausing is
# thought to give the nascent chain time to fold. Hence it is reported as a
# feature rather than folded into a penalty.
RARE_CODONS = frozenset({
    "AGG", "AGA", "CGA", "CGG", "ATA", "CTA", "CCC", "TCG", "ACG", "GGA", "GGG",
})


def calculate_rare_codon_fraction(sequence: str, head_codons: int = 0) -> float:
    """
    Fraction of codons drawn from the slowly-read set.

    With `head_codons` set, only that many codons from the 5' end are counted,
    which is the region where ribosome pausing matters most.
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    codons = [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    if head_codons:
        codons = codons[:head_codons]
    if not codons:
        return 0.0
    return round(sum(1 for c in codons if c in RARE_CODONS) / len(codons), 4)


# Window and step for the gene-wide folding average, matching the 48 nt windows
# Boel reports as Gave48.
FOLDING_AVG_WINDOW_NT = 48
FOLDING_AVG_STEP_NT = 24


@lru_cache(maxsize=1)
def _rna_module():
    """
    Import ViennaRNA, working around its missing runtime dependency on Windows.

    The `_RNA` extension links an OpenMP runtime that the wheel does not ship,
    so a bare `import RNA` fails with "DLL load failed" in a process that has
    loaded nothing else. SciPy and XGBoost both pull that runtime in, which is
    why folding worked inside the application, where the model loads first, and
    silently returned nothing in a standalone script. Importing SciPy first makes
    the import deterministic rather than dependent on what ran before.

    Returns the module, or None when ViennaRNA is genuinely unavailable.
    """
    try:
        import RNA
        return RNA
    except ImportError:
        pass

    # Put the OpenMP runtime on the DLL search path. scikit-learn ships
    # vcomp140.dll in its .libs directory; on Windows that is enough for the
    # ViennaRNA extension to resolve.
    if hasattr(os, "add_dll_directory"):
        for root in site_packages_dirs():
            for candidate in (os.path.join(root, "sklearn", ".libs"),):
                if os.path.isdir(candidate):
                    try:
                        os.add_dll_directory(candidate)
                        logger.info(f"Added '{candidate}' to the DLL search path for ViennaRNA.")
                    except OSError:
                        pass
    try:
        import RNA
        return RNA
    except ImportError:
        pass

    # Last resort: importing scikit-learn loads the same runtime as a side
    # effect, which works even where add_dll_directory does not apply.
    try:
        import sklearn  # noqa: F401
        import RNA
        logger.info("ViennaRNA loaded after preloading the scikit-learn runtime.")
        return RNA
    except ImportError as err:
        logger.warning(f"ViennaRNA unavailable ({err}); real folding energy will not be computed.")
        return None


def site_packages_dirs() -> List[str]:
    """Directories that may hold installed packages, for DLL path lookup."""
    import sysconfig

    paths = {sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")}
    try:
        import sklearn

        paths.add(os.path.dirname(os.path.dirname(sklearn.__file__)))
    except ImportError:
        pass
    return [p for p in paths if p and os.path.isdir(p)]


def calculate_folding_energy(sequence: str) -> Optional[float]:
    """
    Minimum free energy of the sequence, in kcal/mol, via ViennaRNA.

    A real thermodynamic fold, unlike `estimate_mrna_folding_stability`, which
    is a GC-derived proxy. Returns None when ViennaRNA is not installed, so the
    caller can fall back rather than fail.
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "").replace("T", "U")
    if not seq:
        return None
    rna = _rna_module()
    if rna is None:
        return None
    try:
        _, mfe = rna.fold(seq)
        return round(float(mfe), 2)
    except Exception as err:
        logger.error(f"ViennaRNA folding failed: {err}")
        return None


def calculate_genewide_folding_avg(
    sequence: str,
    window: int = FOLDING_AVG_WINDOW_NT,
    step: int = FOLDING_AVG_STEP_NT,
) -> Optional[float]:
    """
    Folding energy averaged over sliding windows across the coding sequence.

    This is the servable counterpart of Boel's Gave48: it needs only the
    construct, with no expression vector. On Welch's measured yields it reaches
    Spearman -0.45 within series, so a more negative average, meaning tighter
    structure, goes with lower yield.
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if len(seq) < window:
        return calculate_folding_energy(seq)

    energies = []
    for start in range(0, len(seq) - window + 1, step):
        mfe = calculate_folding_energy(seq[start:start + window])
        if mfe is None:
            return None
        energies.append(mfe)
    if not energies:
        return None
    return round(sum(energies) / len(energies), 2)


def calculate_protein_features(sequence: str) -> Dict[str, float]:
    """
    Translate DNA sequence to amino acid sequence (stopping at first stop codon)
    and compute GRAVY score (hydrophobicity) and Isoelectric Point (pI).
    """
    if not isinstance(sequence, str) or not sequence.strip():
        logger.warning("Invalid or empty string passed for protein feature extraction.")
        return {"gravy_score": float("nan"), "pI": float("nan")}

    try:
        from Bio.Seq import Seq
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        seq = sequence.upper().replace(" ", "").replace("\n", "")
        seq_obj = Seq(seq)
        protein_seq = str(seq_obj.translate(to_stop=True))

        if not protein_seq or any(c in protein_seq for c in ("*", "X", "J", "O", "U", "B", "Z")):
            clean_protein = "".join([c for c in protein_seq if c in "ACDEFGHIKLMNPQRSTVWY"])
            if not clean_protein:
                logger.warning(f"Could not translate DNA to standard amino acids. Sequence length: {len(seq)} bp.")
                return {"gravy_score": float("nan"), "pI": float("nan")}
            protein_seq = clean_protein

        analyzer = ProteinAnalysis(protein_seq)
        gravy = round(analyzer.gravy(), 4)
        pi = round(analyzer.isoelectric_point(), 4)
        return {"gravy_score": gravy, "pI": pi}
    except Exception as err:
        logger.error(f"Protein feature extraction failed: {str(err)}", exc_info=False)
        return {"gravy_score": float("nan"), "pI": float("nan")}


def extract_features(sequence: str) -> Dict[str, float]:
    """
    Extract every computed feature for a given sequence.

    `mrna_folding_stability` is the historical GC-derived proxy, kept so that
    artifacts fitted on it still load. `genewide_folding_avg` is the real
    ViennaRNA energy and is what new models should use; it is NaN when
    ViennaRNA is unavailable.
    """
    logger.info(f"Extracting features for sequence length {len(sequence)} bp...")
    folding_avg = calculate_genewide_folding_avg(sequence)
    head_folding = calculate_folding_energy(sequence[:FOLDING_WINDOW_NT])
    clean = sequence.upper().replace(" ", "").replace("\n", "")
    features = {
        "gc_content": calculate_gc_content(sequence),
        "gc3_content": calculate_gc3(sequence),
        "codon_adaptation_index": calculate_cai(sequence),
        "rare_codon_fraction": calculate_rare_codon_fraction(sequence),
        "rare_codons_head30": calculate_rare_codon_fraction(sequence, head_codons=30),
        "length_codons": len(clean) // 3,
        "mrna_folding_stability": estimate_mrna_folding_stability(sequence),
        # Folding of the construct's own 5' head. The servable counterpart of
        # Boel's Gz48v, which additionally folds the expression vector's UTR.
        "head_folding_48": float("nan") if head_folding is None else head_folding,
        "genewide_folding_avg": float("nan") if folding_avg is None else folding_avg,
    }
    protein_feats = calculate_protein_features(sequence)
    features.update(protein_feats)
    logger.info(
        f"Feature Extraction Summary -> GC: {features['gc_content']}%, GC3: {features['gc3_content']}%, "
        f"CAI: {features['codon_adaptation_index']}, rare codons: {features['rare_codon_fraction']}, "
        f"folding avg: {features['genewide_folding_avg']} kcal/mol, "
        f"GRAVY: {features.get('gravy_score')}, pI: {features.get('pI')}"
    )
    return features


def extract_features_df(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """Batch extract core features for a pandas DataFrame of sequences."""
    logger.info(f"Batch extracting features for DataFrame of {len(df)} constructs...")
    features_list = df[sequence_col].apply(extract_features).tolist()
    return pd.DataFrame(features_list)


# Above this many annotated coding sequences the record is a genome or a
# multi-gene contig, not one gene.
MAX_ANNOTATED_CDS = 3


def _parse_fasta_records(text: str) -> List[Tuple[str, str]]:
    """Split FASTA text into (header, sequence) pairs."""
    records, header, chunks = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def fetch_ncbi_cds(accession_id: str) -> Tuple[str, str]:
    """
    Fetch the coding sequence annotated on an NCBI record.

    A RefSeq mRNA record carries the untranslated regions as well as the CDS, so
    fetching the record whole yields a sequence that does not begin at the start
    codon: NM_000207 returns 465 nt around a 333 nt CDS. NCBI can return the
    annotated CDS directly, which is what this asks for.

    Returns the longest CDS and a note describing what was found. Raises
    LookupError when the record carries no CDS annotation, which is the case for
    a genome or a non-coding record.
    """
    acc = accession_id.strip().upper()
    if not acc:
        raise ValueError("NCBI Accession ID cannot be empty.")

    import urllib.request

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={acc}&rettype=fasta_cds_na&retmode=text"
    )
    logger.info(f"Requesting annotated CDS for '{acc}' from NCBI...")
    req = urllib.request.Request(url, headers={"User-Agent": "SynthBio/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")

    records = [(h, s.upper()) for h, s in _parse_fasta_records(text) if s]

    # A genome annotates thousands of genes. Picking the longest would return
    # some arbitrary gene under the user's accession, which is worse than
    # refusing: they would score a sequence they never asked for.
    if len(records) > MAX_ANNOTATED_CDS:
        logger.warning(f"'{acc}' annotates {len(records)} CDS records; refusing to choose one.")
        raise LookupError(
            f"Record '{acc}' annotates {len(records):,} coding sequences, so it is a genome or "
            "multi-gene record rather than one gene."
        )

    if not records:
        # Older GenBank entries for cDNA clones often carry no CDS annotation
        # and are themselves close to a bare coding sequence, so the record is
        # used as it stands and validation decides whether it is usable.
        logger.warning(f"No CDS annotation on '{acc}'; falling back to the record as supplied.")
        seq = fetch_ncbi_sequence(acc)
        return seq, (
            f"{acc} carries no CDS annotation, so the record is used as supplied ({len(seq)} nt). "
            "If it includes untranslated regions, trim to the start codon yourself."
        )

    header, seq = max(records, key=lambda r: len(r[1]))
    if len(records) == 1:
        note = f"Using the coding sequence annotated on {acc} ({len(seq)} nt)."
    else:
        note = (
            f"{acc} annotates {len(records)} coding sequences; using the longest "
            f"({len(seq)} nt). Paste another one directly if you need it."
        )
    logger.info(f"Fetched CDS for '{acc}': {len(seq)} nt from {len(records)} annotated CDS record(s).")
    return seq, note


def fetch_ncbi_sequence(accession_id: str, email: str = "synthbio@example.com") -> str:
    """Fetch nucleotide FASTA sequence from NCBI Nucleotide database using accession number."""
    acc = accession_id.strip().upper()
    if not acc:
        logger.error("Empty NCBI Accession ID provided.")
        raise ValueError("NCBI Accession ID cannot be empty.")

    logger.info(f"Attempting NCBI Entrez fetch for Accession '{acc}'...")

    # Try BioPython Entrez first
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = email
        handle = Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        seq = str(record.seq).upper()
        if seq:
            logger.info(f"Successfully fetched {len(seq)} bp from NCBI via Bio.Entrez for '{acc}'.")
            return seq
    except Exception as err:
        logger.warning(f"Bio.Entrez fetch failed for '{acc}': {str(err)}. Attempting HTTP eutils fallback...")

    # Fallback to direct HTTP request to NCBI eutils API
    try:
        import urllib.request
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={acc}&rettype=fasta&retmode=text"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            fasta_data = response.read().decode('utf-8')
            lines = fasta_data.strip().split('\n')
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
            if not seq:
                raise ValueError(f"No sequence data returned from NCBI for accession '{acc}'.")
            logger.info(f"Successfully fetched {len(seq)} bp from NCBI via HTTP eutils for '{acc}'.")
            return seq.upper()
    except Exception as e:
        logger.error(f"Failed to fetch sequence from NCBI for accession '{acc}': {str(e)}")
        raise ValueError(f"Failed to fetch sequence from NCBI for accession '{acc}': {str(e)}")
