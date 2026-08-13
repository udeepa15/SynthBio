# 🧬 SynthBio: New Features Documentation

This document summarizes the new feature engineering routines, Explainable AI (XAI) modules, and smart logic layer enhancements implemented in the **SynthBio Construct Failure Predictor**.

---

## 📌 Summary of New Capabilities

1. **Protein-Level Feature Extraction** (`gravy_score` & `pI`)
2. **5-Feature Processed Dataset Generation** (`feature_table_boel_5features.csv`)
3. **Explainable AI (XAI) SHAP Waterfall Plot Visualization**
4. **Rule-Based Smart Logic Layer**: *Over-Optimization Trap Detection*

---

## 1. 🧪 Protein-Level Feature Extraction

SynthBio now translates DNA coding sequences into amino acid sequences to extract protein-level biophysical properties using Biopython (`Bio.Seq` and `Bio.SeqUtils.ProtParam`).

### Feature Definitions

| Feature | Column Name | Data Type | Module / Tool | Description |
| :--- | :--- | :--- | :--- | :--- |
| **GRAVY Score** | `gravy_score` | Float | `Bio.SeqUtils.ProtParam` | Grand Average of Hydropathicity. Positive values indicate hydrophobic proteins; negative values indicate hydrophilic proteins. |
| **Isoelectric Point** | `pI` | Float | `Bio.SeqUtils.ProtParam` | The pH value at which the net electrical charge of the translated protein is zero. |

### Implementation Details ([`src/features.py`](file:///d:/UDEEPA/Projects/SynthBio/src/features.py))

```python
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_protein_features(sequence: str) -> Dict[str, float]:
    """
    Translates DNA sequence into amino acids (stopping at first stop codon)
    and computes GRAVY score and Isoelectric Point (pI).
    Handles translation errors or malformed sequences gracefully by returning NaN.
    """
    if not isinstance(sequence, str) or not sequence.strip():
        return {"gravy_score": float("nan"), "pI": float("nan")}
    try:
        seq_obj = Seq(sequence.upper().replace(" ", "").replace("\n", ""))
        protein_seq = str(seq_obj.translate(to_stop=True))
        
        # Clean non-standard amino acid characters if present
        clean_protein = "".join([c for c in protein_seq if c in "ACDEFGHIKLMNPQRSTVWY"])
        if not clean_protein:
            return {"gravy_score": float("nan"), "pI": float("nan")}

        analyzer = ProteinAnalysis(clean_protein)
        return {
            "gravy_score": round(analyzer.gravy(), 4),
            "pI": round(analyzer.isoelectric_point(), 4)
        }
    except Exception:
        return {"gravy_score": float("nan"), "pI": float("nan")}
```

---

## 2. 📂 5-Feature Processed Dataset

Dataset generation routines in [`src/data_processing.py`](file:///d:/UDEEPA/Projects/SynthBio/src/data_processing.py) append `gravy_score` and `pI` to raw sequence tables and export `data/processed/feature_table_boel_5features.csv`.

### Command Execution
```powershell
python -m src.data_processing --input data/raw/sample_sequences.csv --output data/processed/feature_table_boel_5features.csv
```

---

## 3. 🔍 Explainable AI (XAI) — SHAP Waterfall Plot

The XAI module ([`src/xai.py`](file:///d:/UDEEPA/Projects/SynthBio/src/xai.py)) wraps trained gradient boosted models using `shap.TreeExplainer` and renders single-prediction waterfall plots inside Streamlit.

### Key Visual & Analytical Features
- **Baseline Expected Value**: Displays the base expected risk/score before feature contributions.
- **Red Bars (+SHAP)**: Feature values that push failure risk or prediction score higher.
- **Blue Bars (-SHAP)**: Feature values that pull failure risk or prediction score lower.
- **Dynamic Text Summary**: Generates natural language summaries explaining the top positive and negative feature contributors.

```python
from src.xai import render_shap_waterfall

# Call inside Streamlit app:
render_shap_waterfall(model, feat_df)
```

---

## 4. ⚠️ Rule-Based Smart Logic Layer: Over-Optimization Trap

To prevent classic synthetic biology failure cases (e.g., Dong et al. 1995), a rule-based smart logic layer checks for extreme codon optimization that creates translation-blocking mRNA hairpins.

### Trigger Condition
$$\text{CAI} > 0.85 \quad \text{AND} \quad \text{predicted\_score} \le 2 \quad \text{AND} \quad \text{folding\_energy} < -15.0 \text{ kcal/mol}$$

### UI Error Banner & Recommendation ([`app/main.py`](file:///d:/UDEEPA/Projects/SynthBio/app/main.py))
```python
if cai_val > 0.85 and pred_score <= 2 and mfe_val < -15.0:
    st.error("⚠️ OVER-OPTIMIZATION TRAP DETECTED!")
    st.info(
        "Your Codon Adaptation Index is excellent, but the model predicts failure. "
        "This is likely because the aggressive codon optimization resulted in high GC content, "
        "creating tight mRNA hairpins that block the ribosome. We recommend relaxing the CAI "
        "to improve folding energy."
    )
```

---

## 🧪 Automated Testing Suite

All new features include unit tests in [`tests/`](file:///d:/UDEEPA/Projects/SynthBio/tests/):
- [`tests/test_protein_features.py`](file:///d:/UDEEPA/Projects/SynthBio/tests/test_protein_features.py): Tests sequence translation, GRAVY/pI calculations, stop codons, and NaN handling.
- [`tests/test_xai.py`](file:///d:/UDEEPA/Projects/SynthBio/tests/test_xai.py): Tests SHAP TreeExplainer calculation and plot rendering.
- [`tests/test_over_optimization.py`](file:///d:/UDEEPA/Projects/SynthBio/tests/test_over_optimization.py): Tests Over-Optimization Trap trigger conditions.
