"""
Streamlit Web Interface for SynthBio: Construct Failure Predictor.
Provides a modern, dark-mode dashboard layout for evaluating gene construct failure risk,
predicting expression tiers, extracting 5 biophysical features, and visualizing SHAP attributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging
import traceback

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logger
logger = logging.getLogger("SynthBio.app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.app] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from src.predict import predict_construct
from src.model import load_model, get_shap_explainer
from src.features import fetch_ncbi_sequence
from src.xai import render_shap_waterfall

@st.cache_resource
def load_trained_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.joblib")
    logger.info(f"Checking for trained model artifact at '{model_path}'...")
    if os.path.exists(model_path):
        try:
            logger.info("Attempting to unpickle and load model artifact...")
            model = load_model(model_path)
            logger.info("Model loaded successfully. Initializing SHAP TreeExplainer...")
            explainer = get_shap_explainer(model)
            logger.info("Artifact loading pipeline complete.")
            return model, explainer
        except Exception as err:
            logger.error(f"Failed to load model artifact: {str(err)}", exc_info=True)
            logger.warning("Application will fall back to biophysical heuristic prediction mode.")
            return None, None
    else:
        logger.warning(f"No model artifact found at '{model_path}'. Running in heuristic prediction mode.")
        return None, None

def inject_custom_css():
    """Inject modern dark-mode glassmorphism CSS styling for UI polish."""
    st.markdown(
        """
        <style>
        /* Modern Dark Theme Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: #00E5FF !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #A0AAB8 !important;
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 14px 18px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #00E5FF;
        }
        /* Tab Styling */
        button[data-baseweb="tab"] {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }
        /* Expander Header Polish */
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    logger.info("Initializing Streamlit Application Page Configuration...")
    st.set_page_config(
        page_title="SynthBio | Construct Failure Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # 1. Sidebar Configuration
    st.sidebar.header("⚙️ Settings & Model Config")
    target_host = st.sidebar.selectbox(
        "Select Target Host",
        ["Escherichia coli (K-12)", "Escherichia coli (BL21)"],
        index=0
    )
    model_arch = st.sidebar.selectbox(
        "Model Architecture",
        ["XGBoost Baseline", "LightGBM Baseline"],
        index=0
    )
    
    model, explainer = load_trained_artifacts()
    if model is not None:
        st.sidebar.success("✅ Trained Model Loaded (models/baseline_model.joblib)")
    else:
        st.sidebar.info("ℹ️ Running in Heuristic Mode (Train model via pipeline)")

    # 2. Main Header & Input Section
    st.title("🧬 SynthBio: Construct Failure Predictor")
    st.markdown(
        "Predict *E. coli* expression yield and evaluate construct failure risk "
        "using 5 computed biophysical and protein features: **GC%**, **Codon Adaptation Index (CAI)**, "
        "**mRNA Folding Stability (MFE)**, **GRAVY (Hydrophobicity)**, and **Isoelectric Point (pI)**."
    )

    st.subheader("🧪 Input Gene Sequence")
    tab_raw, tab_ncbi = st.tabs(["📄 Raw DNA Sequence", "🌐 NCBI Accession ID"])
    
    sequence_input = ""
    with tab_raw:
        sample_seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
        raw_seq = st.text_area(
            "Paste DNA Sequence (5' to 3'):",
            value=sample_seq,
            height=130
        ).strip().upper()
        sequence_input = raw_seq

    with tab_ncbi:
        accession_id = st.text_input(
            "Enter NCBI Accession Number:",
            value="",
            placeholder="e.g., M19447, NM_001101, NC_000913, M15077.1"
        ).strip()
        
        if accession_id:
            logger.info(f"User requested NCBI Accession fetch for ID: '{accession_id}'")
            try:
                with st.spinner(f"Fetching FASTA sequence from NCBI for accession '{accession_id}'..."):
                    fetched_seq = fetch_ncbi_sequence(accession_id)
                    if fetched_seq:
                        st.success(f"Successfully fetched {len(fetched_seq)} bp sequence from NCBI!")
                        sequence_input = fetched_seq
            except Exception as err:
                logger.error(f"NCBI accession lookup failed for '{accession_id}': {str(err)}", exc_info=True)
                st.error(f"Error fetching NCBI accession: {str(err)}")

    # Sequence preview wrapped inside an expander
    if len(sequence_input) > 0:
        with st.expander("View Sequence Preview", expanded=False):
            st.text_area(
                f"Full Sequence ({len(sequence_input)} bp):",
                value=sequence_input,
                height=120,
                disabled=True
            )

    run_btn = st.button("🚀 Analyze & Predict Expression", type="primary")

    # 3. Results & XAI Section (Only visible after clicking Analyze or having a sequence input)
    if (run_btn or sequence_input) and len(sequence_input) > 0:
        logger.info(f"Triggering prediction workflow for input sequence length {len(sequence_input)} bp...")
        try:
            res = predict_construct(sequence_input, model=model, explainer=explainer)
            logger.info(
                f"Prediction Complete -> Risk: {res['risk_score']}%, Tier: {res['predicted_tier']}, "
                f"Yield: {res['predicted_yield']} AU"
            )

            st.markdown("---")

            # 50/50 Split Layout for Results
            left_col, right_col = st.columns([1, 1])

            # Left Column: Extracted Features
            with left_col:
                st.markdown("### 📊 Extracted Sequence & Protein Features")
                
                # Grid display of 5 features using st.metric
                f1, f2, f3 = st.columns(3)
                f1.metric("1. GC Content", f"{res['features']['gc_content']}%")
                f2.metric("2. CAI", f"{res['features']['codon_adaptation_index']}")
                f3.metric("3. MFE Folding", f"{res['features']['mrna_folding_stability']} kcal/mol")
                
                f4, f5, f_blank = st.columns(3)
                gravy_score = res['features'].get('gravy_score')
                pi_value = res['features'].get('pI')
                f4.metric("4. GRAVY Score", f"{gravy_score}" if gravy_score is not None and not np.isnan(gravy_score) else "N/A")
                f5.metric("5. Isoelectric Point", f"{pi_value}" if pi_value is not None and not np.isnan(pi_value) else "N/A")

                # Biological Construct Diagnostics Logic Layer
                with st.expander("🩺 Biological Construct Diagnostics", expanded=True):
                    has_warning = False

                    # 2. GRAVY Score Logic (Hydrophobicity)
                    if gravy_score is not None and not np.isnan(gravy_score) and gravy_score > 0.4:
                        has_warning = True
                        st.warning("⚠️ Inclusion Body Risk: This construct has a high GRAVY score. The protein is highly hydrophobic and is likely to aggregate into insoluble inclusion bodies rather than remain in the soluble fraction.")

                    # 3. Isoelectric Point (pI) Logic
                    if pi_value is not None and not np.isnan(pi_value) and (7.2 <= pi_value <= 7.8):
                        has_warning = True
                        st.warning("⚠️ Aggregation Warning: The pI of this protein falls within the internal pH of the E. coli cytoplasm (7.2 - 7.8). The protein will carry a neutral charge and is highly prone to aggregation. Consider adding a charged fusion tag to shift the pI.")

                    # 4. Success State
                    if not has_warning:
                        st.success("✅ Diagnostics Clear: No major solubility or aggregation red flags detected for E. coli expression based on GRAVY and pI thresholds.")

            # Right Column: Prediction Output & Risk
            with right_col:
                st.markdown("### 🎯 Prediction Output & Risk")
                
                # Dynamic colored box for failure risk
                risk = res["risk_score"]
                if risk < 30.0:
                    st.success(f"🟢 **Low Risk of Construct Failure ({risk}%)**")
                elif risk < 60.0:
                    st.warning(f"🟡 **Moderate Construct Failure Risk ({risk}%)**")
                else:
                    st.error(f"🔴 **High Construct Failure Risk ({risk}%)**")

                p1, p2 = st.columns(2)
                p1.metric("Predicted Expression Tier", res["predicted_tier"])
                p2.metric("Estimated Yield", f"{res['predicted_yield']} AU")

                # Smart Logic Layer: Over-Optimization Trap check
                cai_val = res["features"]["codon_adaptation_index"]
                mfe_val = res["features"]["mrna_folding_stability"]
                pred_score = res.get("predicted_score")
                if pred_score is None:
                    pred_score = 0 if "Low" in str(res["predicted_tier"]) else (5 if "High" in str(res["predicted_tier"]) else 3)

                if cai_val > 0.85 and pred_score <= 2 and mfe_val < -15.0:
                    logger.warning("Over-Optimization Trap detected for user construct!")
                    st.error("⚠️ OVER-OPTIMIZATION TRAP DETECTED!")
                    st.info(
                        "Your Codon Adaptation Index is excellent, but the model predicts failure. "
                        "Aggressive codon optimization resulted in high GC content, creating tight mRNA hairpins "
                        "that block ribosome entry. We recommend relaxing the CAI to improve 5' folding energy."
                    )

                # Feature Importance Summary inside an expander
                with st.expander("🔍 View Global Feature Importance", expanded=False):
                    fi_df = pd.DataFrame(
                        list(res["feature_importances"].items()),
                        columns=["Feature", "Importance Score"]
                    ).sort_values("Importance Score", ascending=True)
                    st.bar_chart(fi_df.set_index("Feature"))

            # 4. Explainable AI (XAI) Section inside large expander
            st.markdown("---")
            feat_df = pd.DataFrame([{
                "GC_percent": res["features"]["gc_content"],
                "CAI": res["features"]["codon_adaptation_index"],
                "folding_energy": res["features"]["mrna_folding_stability"],
                "gravy_score": res["features"].get("gravy_score", np.nan),
                "pI": res["features"].get("pI", np.nan)
            }])
            render_shap_waterfall(model, feat_df)

        except Exception as pipeline_err:
            logger.error(f"Unhandled exception during prediction execution: {str(pipeline_err)}", exc_info=True)
            st.error(f"An unexpected error occurred during prediction: {str(pipeline_err)}")

if __name__ == "__main__":
    main()
