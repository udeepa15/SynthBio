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

def main():
    st.set_page_config(
        page_title="SynthBio | Construct Failure Predictor",
        page_icon="🧬",
        layout="wide"
    )

    st.title("🧬 SynthBio: Construct Failure Predictor")
    st.markdown(
        "Predict *E. coli* expression yield and evaluate construct failure risk "
        "using 3 core computed features: **GC%**, **Codon Adaptation Index (CAI)**, and **mRNA Folding Stability (MFE)**."
    )

    st.sidebar.header("⚙️ Settings & Model Config")
    st.sidebar.selectbox("Select Target Host", ["Escherichia coli (K-12)", "Escherichia coli (BL21)"])
    st.sidebar.selectbox("Model Architecture", ["XGBoost Baseline", "LightGBM Baseline"])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📥 Input Gene Sequence")
        sample_seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
        sequence_input = st.text_area(
            "Paste DNA Sequence (5' to 3'):",
            value=sample_seq,
            height=160
        ).strip().upper()

        run_btn = st.button("🚀 Analyze & Predict Expression", type="primary")

    if run_btn or sequence_input:
        res = predict_construct(sequence_input)

        with col1:
            st.markdown("### 📊 Extracted Sequence Features")
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("GC Content", f"{res['features']['gc_content']}%")
            f_col2.metric("Codon Adapt. Index (CAI)", f"{res['features']['codon_adaptation_index']}")
            f_col3.metric("mRNA Stability (MFE)", f"{res['features']['mrna_folding_stability']} kcal/mol")

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

            st.markdown("### 🔍 Feature Importance Output")
            fi_df = pd.DataFrame(
                list(res["feature_importances"].items()),
                columns=["Feature", "Importance Score"]
            ).sort_values("Importance Score", ascending=True)
            
            st.bar_chart(fi_df.set_index("Feature"))

if __name__ == "__main__":
    main()
