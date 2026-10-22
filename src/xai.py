"""
Explainable AI (XAI) Module for SynthBio Streamlit Application.
Provides SHAP TreeExplainer integration and Waterfall Plot visualization
with dynamic natural language summaries and biological explanations for single construct predictions.
"""

from typing import Any, Optional, Dict
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure logger
logger = logging.getLogger("SynthBio.xai")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.xai] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    import shap
except ImportError:
    shap = None

# Feature description mapping for biological interpretation
FEATURE_BIO_MAP = {
    "GC_percent": {
        "name": "GC Content (%)",
        "high_desc": "High GC content promotes tight mRNA secondary structure folding and slows down translation initiation in E. coli.",
        "low_desc": "Low GC content reduces overall mRNA stability and can lead to mRNA degradation."
    },
    "CAI": {
        "name": "Codon Adaptation Index (CAI)",
        "high_desc": "High CAI aligns with preferred E. coli tRNAs, accelerating translation elongation speed.",
        "low_desc": "Low CAI indicates rare codons that stall translation ribosomes, leading to premature termination or low yield."
    },
    "folding_energy": {
        "name": "mRNA Folding Stability (MFE)",
        "high_desc": "Strong negative MFE indicates tight 5' hairpins near the Shine-Dalgarno / start codon site that physically block ribosome binding.",
        "low_desc": "Relaxed MFE allows open 5' mRNA conformation for efficient ribosome loading."
    },
    "gravy_score": {
        "name": "GRAVY Score (Hydrophobicity)",
        "high_desc": "Hydrophobic protein sequences (positive GRAVY) increase inclusion body formation and protein aggregation in host cytoplasm.",
        "low_desc": "Hydrophilic protein sequences (negative GRAVY) remain soluble in host cytoplasm."
    },
    "pI": {
        "name": "Isoelectric Point (pI)",
        "high_desc": "Basic pI relative to E. coli cytoplasmic pH (~7.2-7.6) influences net charge and protein-RNA electrostatic interaction.",
        "low_desc": "Acidic pI relative to cytoplasmic pH influences protein solubility and stability."
    }
}


def render_shap_waterfall(model: Any, input_df: pd.DataFrame) -> None:
    """
    Render a SHAP Waterfall Plot inside a large expander for a single construct prediction in Streamlit using matplotlib,
    accompanied by sub-columns with white background chart and a 1-sentence dynamic summary.
    """
    with st.expander("🧬 Explainable AI (XAI) — SHAP Feature Attribution", expanded=False):
        if shap is None:
            logger.warning("SHAP rendering requested but `shap` package is not installed.")
            st.warning("⚠️ SHAP library is not installed. Please run `python -m pip install shap` to enable interactive XAI waterfall plots.")
            return

        if model is None or not hasattr(model, "predict"):
            logger.info("SHAP waterfall rendering skipped: Running in heuristic mode without a trained model artifact.")
            st.info("ℹ️ Running in heuristic mode without a loaded model artifact. Train or load a model artifact to generate SHAP Waterfall plots.")
            return

        try:
            logger.info(f"Preparing input features for SHAP Waterfall plot: {list(input_df.columns)}...")
            if hasattr(model, "feature_names_in_"):
                cols = [c for c in list(model.feature_names_in_) if c in input_df.columns]
                if cols:
                    eval_df = input_df[cols]
                else:
                    eval_df = input_df
            else:
                eval_df = input_df

            explainer = shap.TreeExplainer(model)
            explanation = explainer(eval_df)
            sample_exp = explanation[0]

            pred_class = 0
            if hasattr(model, "predict"):
                try:
                    pred_class = int(model.predict(eval_df)[0])
                except Exception:
                    pred_class = 0

            if len(sample_exp.shape) > 1:
                class_idx = min(pred_class, sample_exp.shape[1] - 1)
                sample_exp = sample_exp[:, class_idx]

            # Sub-columns: Left 1.5 width (Chart), Right 1 width (Analysis)
            sub_col1, sub_col2 = st.columns([1.5, 1])

            with sub_col1:
                st.markdown("### 📈 Local Feature Waterfall Chart")
                fig, ax = plt.subplots(figsize=(7.5, 4.2))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                shap.plots.waterfall(sample_exp, show=False)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                logger.info("SHAP Waterfall plot rendered successfully with white background contrast.")

            # Extract values for dynamic summary
            base_value = float(sample_exp.base_values) if hasattr(sample_exp, "base_values") else 0.0
            shap_values = sample_exp.values
            feature_names = sample_exp.feature_names if hasattr(sample_exp, "feature_names") else eval_df.columns.tolist()

            max_pos_idx = int(np.argmax(shap_values))
            max_neg_idx = int(np.argmin(shap_values))

            top_increasing_feature = feature_names[max_pos_idx] if shap_values[max_pos_idx] > 0 else None
            top_decreasing_feature = feature_names[max_neg_idx] if shap_values[max_neg_idx] < 0 else None

            with sub_col2:
                st.markdown("### 💡 Detailed Result & Biological Analysis")
                st.markdown(f"• **Base Expected Score ($E[f(X)]$):** `{base_value:.2f}` (Baseline expectation across training constructs).")

                if top_increasing_feature:
                    feat_meta = FEATURE_BIO_MAP.get(top_increasing_feature, {"name": top_increasing_feature, "high_desc": "Pushes failure risk higher."})
                    st.markdown(
                        f"🔴 **Primary Failure Risk Driver**: <span style='color:#FF4B4B; font-weight:bold;'>{feat_meta['name']}</span> (`+{shap_values[max_pos_idx]:.3f} SHAP`)\n\n"
                        f"*{feat_meta['high_desc']}*",
                        unsafe_allow_html=True
                    )

                if top_decreasing_feature:
                    feat_meta_dec = FEATURE_BIO_MAP.get(top_decreasing_feature, {"name": top_decreasing_feature, "low_desc": "Helps lower failure risk."})
                    st.markdown(
                        f"🔵 **Primary Favorable Factor**: <span style='color:#0068C9; font-weight:bold;'>{feat_meta_dec['name']}</span> (`{shap_values[max_neg_idx]:.3f} SHAP`)\n\n"
                        f"*{feat_meta_dec['low_desc']}*",
                        unsafe_allow_html=True
                    )

                with st.expander("📖 How to Read This SHAP Chart", expanded=False):
                    st.markdown(
                        """
                        - **Base Value ($E[f(X)]$)**: Average model output score across all baseline training constructs.
                        - **Red Bars ($+\\text{SHAP}$)**: Feature values that push prediction score toward lower expression / higher risk.
                        - **Blue Bars ($-\\text{SHAP}$)**: Feature values that pull prediction score toward higher expression / lower risk.
                        - **Final Output ($f(x)$)**: Final calculated model score for this specific construct.
                        """
                    )

            # Single-sentence blue st.info box below sub-columns
            if top_increasing_feature and top_decreasing_feature:
                summary_sentence = (
                    f"The feature that increased construct failure risk the most was **{FEATURE_BIO_MAP.get(top_increasing_feature, {}).get('name', top_increasing_feature)}** "
                    f"(+{shap_values[max_pos_idx]:.3f}), while **{FEATURE_BIO_MAP.get(top_decreasing_feature, {}).get('name', top_decreasing_feature)}** "
                    f"helped lower the risk ({shap_values[max_neg_idx]:.3f})."
                )
            elif top_increasing_feature:
                summary_sentence = (
                    f"The feature that increased construct failure risk the most was **{FEATURE_BIO_MAP.get(top_increasing_feature, {}).get('name', top_increasing_feature)}** "
                    f"(+{shap_values[max_pos_idx]:.3f})."
                )
            elif top_decreasing_feature:
                summary_sentence = (
                    f"The feature that helped lower construct failure risk the most was **{FEATURE_BIO_MAP.get(top_decreasing_feature, {}).get('name', top_decreasing_feature)}** "
                    f"({shap_values[max_neg_idx]:.3f})."
                )
            else:
                summary_sentence = "All biophysical features contributed balanced baseline values to the final prediction output."

            st.info(f"💡 **SHAP Summary**: {summary_sentence}")
            logger.info(f"XAI 1-sentence summary: {summary_sentence}")

        except Exception as err:
            logger.error(f"Error rendering SHAP Waterfall plot: {str(err)}", exc_info=True)
            st.error(f"Error rendering SHAP Waterfall plot: {str(err)}")
