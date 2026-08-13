# 🧪 Data Pipeline & Feature Engineering Specifications

## 📂 Data Directory Layout
- **`data/raw/`**: Holds raw, untouched sequence and experimental expression datasets ingested from public repositories (e.g., NCBI, ENA) or lab measurements.
- **`data/processed/`**: Contains cleaned sequence feature tables (`feature_table.csv`) ready for ML training and validation.

---

## 📋 Feature Table Schema (`data/processed/feature_table.csv`)

| Column | Data Type | Description | Range / Units |
| :--- | :--- | :--- | :--- |
| `gene_id` | String | Unique construct identifier | e.g., `Synth_001` |
| `GC_percent` | Float | Calculated GC nucleotide content | $0.0\% - 100.0\%$ (Optimal: $45\%-55\%$) |
| `CAI` | Float | Codon Adaptation Index against *E. coli* K-12 | $0.0 - 1.0$ (Optimal: $> 0.65$) |
| `folding_energy` | Float | Estimated 5' mRNA folding MFE | $\text{kcal/mol}$ (Optimal: $> -15.0\text{ kcal/mol}$) |
| `expression_score`| Integer | Expression performance target tier | `0` (Low), `1` (Medium), `2` (High) |

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
Continuous laboratory expression yield values ($\text{AU}$) are categorized into discrete tiers for classification:

| Expression Yield ($\text{AU}$) | Tier Label | Target Class (`expression_score`) |
| :--- | :--- | :--- |
| $\ge 70.0$ | **High** | `2` |
| $30.0 - 69.9$ | **Medium** | `1` |
| $< 30.0$ | **Low** | `0` |
