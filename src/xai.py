"""
Explainable AI (XAI) Module for SynthBio Streamlit Application.
Provides SHAP TreeExplainer integration and Waterfall Plot visualization
with dynamic natural language summaries and biological explanations for single construct predictions.

The model is a six-class tier classifier, so a SHAP value is an attribution
toward the predicted tier, not toward failure. A positive value therefore means
"more of this tier", which is unfavourable only when the predicted tier is
itself low. Labels below are derived from the predicted tier for that reason.
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
    "GC3": {
        "name": "GC3 (Wobble Position)",
        "high_desc": "High GC at codon wobble positions reflects synonymous codon choice and correlates with higher yield in E. coli.",
        "low_desc": "Low GC3 reflects AT-rich codon endings."
    },
    "CAI": {
        "name": "Codon Adaptation Index (CAI)",
        "high_desc": "High CAI aligns with preferred E. coli tRNAs, accelerating translation elongation speed.",
        "low_desc": "Low CAI indicates rare codons that stall translation ribosomes, leading to premature termination or low yield."
    },
    "rare_codon_fraction": {
        "name": "Rare Codon Fraction",
        "high_desc": "Slowly read codons pause ribosomes, which can aid cotranslational folding.",
        "low_desc": "Low rare codon fraction means translation proceeds rapidly with abundant tRNAs."
    },
    "rare_codons_head30": {
        "name": "Rare Codons (5' Head 30)",
        "high_desc": "Rare codons in the first 30 codons pause translation initiation to prevent ribosomal jamming.",
        "low_desc": "Few rare codons in the initial 30 codons."
    },
    "head_folding_48": {
        "name": "5' Head Folding Energy (kcal/mol)",
        "high_desc": "Tightly structured 5' mRNA secondary hairpins impede ribosome entry.",
        "low_desc": "Open 5' mRNA conformation facilitates ribosome loading."
    },
    "genewide_folding_avg": {
        "name": "Gene-Wide Folding Energy (avg kcal/mol)",
        "high_desc": "Tight gene-wide mRNA structures can slow down ribosomal transit.",
        "low_desc": "Relaxed gene-wide folding allows smoother translation elongation."
    },
    "folding_energy": {
        "name": "5' folding proxy (GC-derived, not thermodynamic)",
        "high_desc": "A strongly negative proxy value indicates a GC-rich 5' region, taken to imply tight hairpins near the start codon that obstruct ribosome binding.",
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
    },
    "length_codons": {
        "name": "Transcript Length (Codons)",
        "high_desc": "Longer proteins impose greater metabolic load on host translational machinery.",
        "low_desc": "Shorter peptide chains express more efficiently in E. coli."
    }
}


LOW_TIER_CEILING = 2  # Tiers 0-2 are low-expression outcomes.

# Reference points separating a "high" from a "low" value for each feature.
# The biological blurbs describe what a high or low VALUE does, so they must be
# selected from the value, never from the sign of the attribution: a feature can
# be strong evidence for a low tier while its own value is favourable.
FEATURE_VALUE_MIDPOINT = {
    "GC_percent": 55.0,
    "GC3": 50.0,
    "CAI": 0.70,
    "rare_codon_fraction": 0.05,
    "rare_codons_head30": 0.05,
    "head_folding_48": -15.0,
    "genewide_folding_avg": -15.0,
    "folding_energy": -20.0,
    "gravy_score": 0.0,
    "pI": 7.4,
    "length_codons": 300,
}


def describe_feature_value(feature: str, value: Optional[float]) -> str:
    """Biological note for a feature, chosen from its value rather than its SHAP sign."""
    meta = FEATURE_BIO_MAP.get(feature)
    if meta is None:
        return ""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    midpoint = FEATURE_VALUE_MIDPOINT.get(feature)
    if midpoint is None:
        return ""
    # Folding energies run negative: a value below the midpoint is the tighter "high" case.
    is_high = value < midpoint if feature in ("folding_energy", "head_folding_48", "genewide_folding_avg") else value > midpoint
    return meta["high_desc"] if is_high else meta["low_desc"]


def tier_is_unfavourable(predicted_class: Optional[int]) -> Optional[bool]:
    """
    Whether the predicted tier is itself a poor outcome.

    Returns None when the tier is unknown, in which case attribution is
    described neutrally rather than as a failure or a success driver.
    """
    if predicted_class is None:
        return None
    return predicted_class <= LOW_TIER_CEILING


def render_shap_waterfall(model: Any, input_df: pd.DataFrame, predicted_class: Optional[int] = None) -> None:
    """
    Render a SHAP Waterfall Plot for a single construct prediction.

    `predicted_class` is the tier the model assigned. It decides whether a
    positive attribution is described as driving a poor outcome or a good one;
    without it the attribution is described without a verdict attached.
    """
    with st.expander("Feature attribution (SHAP)", expanded=False):
        if shap is None:
            logger.warning("SHAP rendering requested but `shap` package is not installed.")
            st.warning("SHAP is not installed, so attribution is unavailable. Install it with `python -m pip install shap`.")
            return

        if model is None or not hasattr(model, "predict"):
            logger.info("SHAP waterfall rendering skipped: Running in heuristic mode without a trained model artifact.")
            st.info("Attribution requires a trained model. The figures above came from biophysical rules, which have no attribution to show.")
            return

        try:
            logger.info(f"Preparing input features for SHAP Waterfall plot: {list(input_df.columns)}...")
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
                eval_df = input_df.reindex(columns=expected_cols, fill_value=np.nan)
            else:
                eval_df = input_df

            explainer = shap.TreeExplainer(model)
            explanation = explainer(eval_df)
            sample_exp = explanation[0]

            pred_class = predicted_class
            if pred_class is None:
                try:
                    pred_class = int(model.predict(eval_df)[0])
                except Exception:
                    logger.warning("Could not determine the predicted tier for attribution labelling.")
                    pred_class = None

            if len(sample_exp.shape) > 1:
                class_idx = min(pred_class if pred_class is not None else 0, sample_exp.shape[1] - 1)
                sample_exp = sample_exp[:, class_idx]

            # Sub-columns: Left 1.5 width (Chart), Right 1 width (Analysis)
            sub_col1, sub_col2 = st.columns([1.5, 1])

            with sub_col1:
                tier_note = f" for Tier {pred_class}" if pred_class is not None else ""
                st.markdown(f"**Attribution for this construct{tier_note}**")
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

            # A positive SHAP value is evidence FOR the predicted tier. Whether
            # that is bad news depends on which tier was predicted, so the
            # labels are chosen from the tier rather than from the sign alone.
            unfavourable = tier_is_unfavourable(pred_class)
            tier_phrase = f"Tier {pred_class}" if pred_class is not None else "the predicted tier"

            if unfavourable is True:
                toward_label = f"Strongest evidence for {tier_phrase}, a low-expression outcome"
            elif unfavourable is False:
                toward_label = f"Strongest evidence for {tier_phrase}, an acceptable-expression outcome"
            else:
                toward_label = "Strongest evidence for the predicted tier"
            away_label = f"Strongest evidence against {tier_phrase}"

            # Feature values, used for the biological notes.
            values = {c: float(eval_df.iloc[0][c]) for c in eval_df.columns}

            with sub_col2:
                st.markdown("**Reading the attribution**")
                st.markdown(
                    f"Baseline model output across the training constructs: `{base_value:.2f}`. "
                    f"The bars show how this construct's features moved the output away from that baseline "
                    f"for {tier_phrase}."
                )

                if top_increasing_feature:
                    name = FEATURE_BIO_MAP.get(top_increasing_feature, {}).get("name", top_increasing_feature)
                    note = describe_feature_value(top_increasing_feature, values.get(top_increasing_feature))
                    st.markdown(
                        f"▲ **{toward_label}**: {name} (`+{shap_values[max_pos_idx]:.3f} SHAP`)"
                        + (f"\n\n*{note}*" if note else "")
                    )

                if top_decreasing_feature:
                    name_dec = FEATURE_BIO_MAP.get(top_decreasing_feature, {}).get("name", top_decreasing_feature)
                    note_dec = describe_feature_value(top_decreasing_feature, values.get(top_decreasing_feature))
                    st.markdown(
                        f"▼ **{away_label}**: {name_dec} (`{shap_values[max_neg_idx]:.3f} SHAP`)"
                        + (f"\n\n*{note_dec}*" if note_dec else "")
                    )

                with st.expander("How to read this chart", expanded=False):
                    st.markdown(
                        f"""
                        - **Baseline ($E[f(X)]$)**: the model's average output for {tier_phrase} across the training constructs.
                        - **Bars pointing right ($+$SHAP)**: this feature's value is evidence *for* {tier_phrase}.
                        - **Bars pointing left ($-$SHAP)**: this feature's value is evidence *against* {tier_phrase}.
                        - **Final output ($f(x)$)**: the model's output for {tier_phrase} for this construct.

                        A positive bar is not in itself a warning. It means more of the tier that was
                        predicted, which is unwelcome only when that tier is a low one.
                        """
                    )

            top_name = FEATURE_BIO_MAP.get(top_increasing_feature, {}).get("name", top_increasing_feature)
            bottom_name = FEATURE_BIO_MAP.get(top_decreasing_feature, {}).get("name", top_decreasing_feature)

            if top_increasing_feature and top_decreasing_feature:
                summary_sentence = (
                    f"**{top_name}** was the strongest evidence for {tier_phrase} "
                    f"(+{shap_values[max_pos_idx]:.3f}), and **{bottom_name}** the strongest evidence against it "
                    f"({shap_values[max_neg_idx]:.3f})."
                )
            elif top_increasing_feature:
                summary_sentence = (
                    f"**{top_name}** was the strongest evidence for {tier_phrase} "
                    f"(+{shap_values[max_pos_idx]:.3f})."
                )
            elif top_decreasing_feature:
                summary_sentence = (
                    f"**{bottom_name}** was the strongest evidence against {tier_phrase} "
                    f"({shap_values[max_neg_idx]:.3f})."
                )
            else:
                summary_sentence = (
                    f"No feature moved the output appreciably away from the baseline for {tier_phrase}."
                )

            st.info(summary_sentence)
            logger.info(f"XAI summary (tier={pred_class}, unfavourable={unfavourable}): {summary_sentence}")

        except Exception as err:
            logger.error(f"Error rendering SHAP Waterfall plot: {str(err)}", exc_info=True)
            st.error(f"Error rendering SHAP Waterfall plot: {str(err)}")
