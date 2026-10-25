# 🧪 Data Pipeline & Feature Engineering Specifications

## 📂 Data Directory Layout
- **`data/raw/`**: Holds raw, untouched sequence and experimental expression datasets ingested from public repositories (e.g., NCBI, ENA) or lab measurements.
- **`data/processed/`**: Contains cleaned sequence feature tables (`feature_table.csv`) ready for ML training and validation.

---

## 📋 Feature Table Schema (`data/processed/feature_table.csv`)

| Column | Data Type | Description | Range / Units |
| :--- | :--- | :--- | :--- |
> **Which table to use.** `feature_table_v2.csv` supersedes `feature_table.csv`
> and is what `resolve_dataset_path()` picks by default. The v1 table's `CAI` was
> computed from weights derived over the highest-expressing genes, which leaked
> the target label into a feature, and its single `folding_energy` column held
> the weaker gene-wide average. v2 also strips the vector-derived His-tag (the
> final 27 nt, `CTCGAGCACCACCACCACCACTGA`) before computing GC and CAI.
>
> **v2 columns:** `gene_id`, `GC_percent`, `CAI`, `head_folding_energy`,
> `genewide_folding_avg48`, `expression_score`.
>
> | Feature | Boël column | Spearman vs expression |
> | :--- | :--- | ---: |
> | `head_folding_energy` | `Gz48v`: vector 5' UTR + first 48 nt of the gene | 0.177 |
> | `genewide_folding_avg48` | `Gave48`: sliding 48 nt windows over the rest of the gene | 0.077 |
> | `GC_percent` | — | −0.086 |
> | `CAI` | — | 0.048 |
>
> The mapping of those two column names onto the Boël supplement is inferred from
> its figure legends rather than stated in it. The correlation gap is consistent
> with the inference.

| Column | Data Type | Description | Observed range in the Boël set |
| :--- | :--- | :--- | :--- |
| `gene_id` | String | Construct identifier. The letter prefix is the source organism (`AaR17`, `BtR40`), and folds are grouped by it during validation. | 170 distinct prefixes |
| `GC_percent` | Float | GC nucleotide content | 24.9 – 78.1 |
| `CAI` | Float | Codon Adaptation Index against *E. coli* K-12 | 0.614 – 0.917 |
| `folding_energy` | Float | Estimated 5' mRNA folding MFE (kcal/mol) | −24.4 – −2.8 |
| `expression_score`| Integer | Expression performance target tier | `0` – `5`, see below |

**Target tiers.** The supplied `feature_table.csv` carries **six** tiers, not the
three an earlier revision of this document described. The counts are uneven,
which is why macro F1 rather than accuracy is the headline metric:

| `expression_score` | Label shown in the app | Rows |
| :--- | :--- | ---: |
| 0 | Very Low | 1,754 |
| 1 | Low | 131 |
| 2 | Medium-Low | 423 |
| 3 | Medium | 896 |
| 4 | High | 1,171 |
| 5 | Very High | 1,973 |

> **Open item.** How these six tiers were derived from the original Boël et al.
> measurements is not recorded anywhere in this repository, and the script that
> produced `feature_table.csv` is not committed. Until that provenance is
> established, the mapping from a tier to a physical expression level is unknown,
> and the `10 + 17 x tier` figure the app reports as a yield is a label, not a
> measurement. See step 2 of [`accuracy_roadmap.md`](accuracy_roadmap.md).

> **Servable features.** `src/features.py` computes GC and CAI on a scale
> compatible with v2: GFP's computed CAI of 0.443 sits inside the table's
> 0.33–0.82 range. The two folding columns are **not** servable. They are
> RNAstructure energies that include the expression vector's 5' UTR, so they
> cannot be derived from a pasted construct at all. `run_pipeline` excludes them
> by default and reports that it did; see step 0 of
> [`accuracy_roadmap.md`](accuracy_roadmap.md).

---

## 🧬 Feature Engineering Calculations ([`src/features.py`](file:///d:/UDEEPA/Projects/SynthBio/src/features.py))

### 1. GC Content (%)
$$\text{GC\%} = \left( \frac{\text{Count}(\text{'G'}) + \text{Count}(\text{'C'})}{\text{Length}(\text{Sequence})} \right) \times 100$$

### 2. Codon Adaptation Index (CAI)
Calculates geometric mean of relative adaptiveness values ($w_i$) for codons in the open reading frame against the *E. coli* K-12 codon usage table (`ECOLI_CODON_W`):

$$\text{CAI} = \exp \left( \frac{1}{L} \sum_{i=1}^{L} \ln(w_i) \right)$$

where $L$ is the number of valid codons in the sequence.

### 3. 5' mRNA Folding Stability (MFE)
Estimates secondary structure minimum free energy (MFE) for the translation initiation region ($\text{seq}[0:48]$):

$$\text{MFE} = -1.5 \times N_{5'} \times (\text{GC}_{\text{fraction}})^{1.5} \quad (\text{kcal/mol})$$

where $N_{5'}$ is the length of the 5' region (up to 48 bases).

---

## 🏷️ Expression Tier Categorization ([`src/data_processing.py`](file:///d:/UDEEPA/Projects/SynthBio/src/data_processing.py))

`categorize_expression()` maps a continuous yield in arbitrary units onto three
labels:

| Expression Yield ($\text{AU}$) | Tier Label |
| :--- | :--- |
| $\ge 70.0$ | **High** |
| $30.0 - 69.9$ | **Medium** |
| $< 30.0$ | **Low** |

This function is **not** what produced the `expression_score` column in the
supplied feature table, which holds six tiers rather than three. It applies only
to a raw table that carries an `expression_yield` column, which no committed
dataset does.
