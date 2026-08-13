"""
Streamlit Web Interface for SynthBio: Construct Failure Predictor.
Allows users to enter a synthetic gene sequence, run predictions,
view failure risk scores, computed features, and feature importance visual output.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_construct
from src.model import load_model, get_shap_explainer
from src.features import fetch_ncbi_sequence
from src.xai import render_shap_waterfall

@st.cache_resource
def load_trained_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.joblib")
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            explainer = get_shap_explainer(model)
            return model, explainer
        except Exception:
            return None, None
    return None, None

def main():
    st.set_page_config(
        page_title="SynthBio | Construct Failure Predictor",
        page_icon="🧬",
        layout="wide"
    )

    st.title("🧬 SynthBio: Construct Failure Predictor")
    st.markdown(
        "Predict *E. coli* expression yield and evaluate construct failure risk "
        "using 5 computed features: **GC%**, **Codon Adaptation Index (CAI)**, **mRNA Folding Stability (MFE)**, **GRAVY (Hydrophobicity)**, and **Isoelectric Point (pI)**."
    )

    model, explainer = load_trained_artifacts()

    st.sidebar.header("⚙️ Settings & Model Config")
    st.sidebar.selectbox("Select Target Host", ["Escherichia coli (K-12)", "Escherichia coli (BL21)"])
    st.sidebar.selectbox("Model Architecture", ["XGBoost Baseline", "LightGBM Baseline"])
    if model is not None:
        st.sidebar.success("✅ Trained Model Loaded (models/baseline_model.joblib)")
    else:
        st.sidebar.info("ℹ️ Running in Heuristic Mode (Train model via pipeline)")

    col1, col2 = st.columns([3, 2])

    sequence_input = ""
    with col1:
        st.subheader("📥 Input Gene Sequence")
        tab_raw, tab_ncbi = st.tabs(["📝 Raw DNA Sequence", "🌐 NCBI Accession ID"])
        
        with tab_raw:
            sample_seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
            raw_seq = st.text_area(
                "Paste DNA Sequence (5' to 3'):",
                value=sample_seq,
                height=140
            ).strip().upper()
            sequence_input = raw_seq

        with tab_ncbi:
            accession_id = st.text_input(
                "Enter NCBI Accession Number:",
                value="",
                placeholder="e.g., M19447, NM_001101, NC_000913"
            ).strip()
            
            if accession_id:
                try:
                    with st.spinner(f"Fetching FASTA sequence from NCBI for accession '{accession_id}'..."):
                        fetched_seq = fetch_ncbi_sequence(accession_id)
                        if fetched_seq:
                            st.success(f"Successfully fetched {len(fetched_seq)} bp sequence from NCBI!")
                            st.text_area("Fetched Sequence Preview:", value=fetched_seq[:300] + ("..." if len(fetched_seq) > 300 else ""), height=80, disabled=True)
                            sequence_input = fetched_seq
                except Exception as err:
                    st.error(f"Error fetching NCBI accession: {str(err)}")

        run_btn = st.button("🚀 Analyze & Predict Expression", type="primary")

    if (run_btn or sequence_input) and len(sequence_input) > 0:
        res = predict_construct(sequence_input, model=model, explainer=explainer)

        with col1:
            st.markdown("### 📊 Extracted Sequence & Protein Features")
            f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
            f_col1.metric("GC Content", f"{res['features']['gc_content']}%")
            f_col2.metric("CAI", f"{res['features']['codon_adaptation_index']}")
            f_col3.metric("MFE Stability", f"{res['features']['mrna_folding_stability']}")
            f_col4.metric("GRAVY", f"{res['features'].get('gravy_score', 'N/A')}")
            f_col5.metric("pI", f"{res['features'].get('pI', 'N/A')}")

        with col2:
            st.subheader("🎯 Prediction Output & Risk")
            
            risk = res["risk_score"]
            if risk < 30.0:
                st.success(f"**Low Risk of Failure ({risk}%)**")
            elif risk < 60.0:
                st.warning(f"**Moderate Failure Risk ({risk}%)**")
            else:
                st.error(f"**High Failure Risk ({risk}%)**")

            st.metric("Predicted Expression Tier", res["predicted_tier"])
            st.metric("Estimated Yield", f"{res['predicted_yield']} AU")

            # Smart Logic Layer: Check for Over-Optimization Trap (Dong et al. 1995 failure case)
            cai_val = res["features"]["codon_adaptation_index"]
            mfe_val = res["features"]["mrna_folding_stability"]
            pred_score = res.get("predicted_score")
            if pred_score is None:
                pred_score = 0 if "Low" in str(res["predicted_tier"]) else (5 if "High" in str(res["predicted_tier"]) else 3)

            if cai_val > 0.85 and pred_score <= 2 and mfe_val < -15.0:
                st.error("⚠️ OVER-OPTIMIZATION TRAP DETECTED!")
                st.info(
                    "Your Codon Adaptation Index is excellent, but the model predicts failure. "
                    "This is likely because the aggressive codon optimization resulted in high GC content, "
                    "creating tight mRNA hairpins that block the ribosome. We recommend relaxing the CAI "
                    "to improve folding energy."
                )

            st.markdown("### 🔍 Feature Importance Summary")
            fi_df = pd.DataFrame(
                list(res["feature_importances"].items()),
                columns=["Feature", "Importance Score"]
            ).sort_values("Importance Score", ascending=True)
            
            st.bar_chart(fi_df.set_index("Feature"))

        # Render SHAP Waterfall Plot
        st.markdown("---")
        feat_df = pd.DataFrame([{
            "GC_percent": res["features"]["gc_content"],
            "CAI": res["features"]["codon_adaptation_index"],
            "folding_energy": res["features"]["mrna_folding_stability"],
            "gravy_score": res["features"].get("gravy_score", np.nan),
            "pI": res["features"].get("pI", np.nan)
        }])
        render_shap_waterfall(model, feat_df)

if __name__ == "__main__":
    main()

