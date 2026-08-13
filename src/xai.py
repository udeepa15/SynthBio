"""
Explainable AI (XAI) Module for SynthBio Streamlit Application.
Provides SHAP TreeExplainer integration and Waterfall Plot visualization
with dynamic natural language summaries for single construct predictions.
"""

from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

try:
    import shap
except ImportError:
    shap = None


def render_shap_waterfall(model: Any, input_df: pd.DataFrame) -> None:
    """
    Render a SHAP Waterfall Plot for a single construct prediction in Streamlit using matplotlib,
    accompanied by a dynamic text summary explaining the feature impacts.
    
    Parameters:
    -----------
    model : Any
        Trained machine learning model artifact (e.g., XGBoost, LightGBM, or RandomForest).
    input_df : pd.DataFrame
        Single-row pandas DataFrame containing the 5 feature values for the user construct:
        ["GC_percent", "CAI", "folding_energy", "gravy_score", "pI"]
    """
    st.subheader("🧬 Explainable AI (XAI) — SHAP Waterfall Explanation")

    if shap is None:
        st.warning("⚠️ SHAP library is not installed. Please install `shap` to enable interactive XAI waterfall plots.")
        return

    if model is None or not hasattr(model, "predict"):
        st.info("ℹ️ Running in heuristic mode without a loaded model artifact. Train or load a model artifact to generate SHAP Waterfall plots.")
        return

    try:
        # Align input_df columns with model features if feature_names_in_ is present
        if hasattr(model, "feature_names_in_"):
            cols = [c for c in list(model.feature_names_in_) if c in input_df.columns]
            if cols:
                eval_df = input_df[cols]
            else:
                eval_df = input_df
        else:
            eval_df = input_df

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        explanation = explainer(eval_df)

        # Extract explanation for the single sample row
        sample_exp = explanation[0]

        # Handle multi-class model outputs (e.g. XGBClassifier with 6 tier classes)
        pred_class = 0
        if hasattr(model, "predict"):
            try:
                pred_class = int(model.predict(eval_df)[0])
            except Exception:
                pred_class = 0

        # If sample_exp has multiple dimensions (for multi-class output)
        if len(sample_exp.shape) > 1:
            class_idx = min(pred_class, sample_exp.shape[1] - 1)
            sample_exp = sample_exp[:, class_idx]

        # Render Waterfall plot using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(sample_exp, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # Extract values for dynamic natural language summary
        base_value = float(sample_exp.base_values) if hasattr(sample_exp, "base_values") else 0.0
        shap_values = sample_exp.values
        feature_names = sample_exp.feature_names if hasattr(sample_exp, "feature_names") else eval_df.columns.tolist()

        # Identify top risk-increasing (+SHAP) and risk-decreasing (-SHAP) features
        max_pos_idx = int(np.argmax(shap_values))
        max_neg_idx = int(np.argmin(shap_values))

        top_increasing_feature = feature_names[max_pos_idx] if shap_values[max_pos_idx] > 0 else None
        top_decreasing_feature = feature_names[max_neg_idx] if shap_values[max_neg_idx] < 0 else None

        # Build dynamic summary text
        summary_lines = []
        summary_lines.append(f"The baseline expected risk score is **{base_value:.2f}**.")
        
        if top_increasing_feature and top_decreasing_feature:
            summary_lines.append(
                f"The feature that increased failure risk the most was **{top_increasing_feature}** "
                f"(+{shap_values[max_pos_idx]:.3f}), while **{top_decreasing_feature}** "
                f"helped lower the risk ({shap_values[max_neg_idx]:.3f})."
            )
        elif top_increasing_feature:
            summary_lines.append(
                f"The feature that increased failure risk the most was **{top_increasing_feature}** "
                f"(+{shap_values[max_pos_idx]:.3f})."
            )
        elif top_decreasing_feature:
            summary_lines.append(
                f"The feature that helped lower failure risk the most was **{top_decreasing_feature}** "
                f"({shap_values[max_neg_idx]:.3f})."
            )
        else:
            summary_lines.append("All features contributed balanced baseline values to the final prediction score.")

        summary_text = " ".join(summary_lines)
        st.info(f"💡 **SHAP Interpretation**: {summary_text}")

    except Exception as err:
        st.error(f"Error rendering SHAP Waterfall plot: {str(err)}")
