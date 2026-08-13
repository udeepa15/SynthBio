# 🧬 SynthBio: Construct Failure Predictor for *E. coli*

## 📌 Project Overview
**SynthBio** is a machine learning pipeline and interactive Streamlit web dashboard engineered to evaluate synthetic gene constructs before laboratory synthesis. By analyzing computed sequence-level biophysical features—**GC Content (%)**, **Codon Adaptation Index (CAI)**, and **mRNA Folding Stability (MFE)**—SynthBio forecasts protein expression yield tiers (**High**, **Medium**, **Low**) and calculates a quantitative **Construct Failure Risk Score ($0\% - 100\%$)**.

In synthetic biology, expressing recombinant proteins in *Escherichia coli* often fails due to suboptimal codon usage, extreme GC content causing secondary mRNA hairpins, or translation initiation blockage. SynthBio allows synthetic biologists to screen constructs in silico, saving time and laboratory resources.

---

## 🎯 Key Objectives
- **Early Construct Screening**: Identify high-risk gene constructs prior to gene synthesis and cloning.
- **Biophysical Feature Extraction**: Compute codon adaptation index against *E. coli* K-12 usage tables, 5' mRNA secondary structure free energy, and GC content dynamically.
- **Explainable Machine Learning**: Train gradient-boosted decision trees (XGBoost / LightGBM) with Stratified K-Fold Cross-Validation and extract SHAP (SHapley Additive exPlanations) values to explain feature contributions.
- **Interactive Web App**: Provide an intuitive Streamlit interface for pasting DNA sequences, viewing risk metrics, and inspecting feature importance visual outputs.

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Machine Learning**: `xgboost`, `lightgbm`, `scikit-learn`
- **Feature Computation**: `biopython`, `numpy`, `pandas`
- **Explainability & Viz**: `shap`, `matplotlib`, `seaborn`
- **Web Interface**: `streamlit`
- **Configuration & Testing**: `pyyaml`, `pytest`

---

## 📅 4-Week Development Roadmap

```
Week 1: Data Collection & Feature Engineering
├── Setup directory structure and virtual environment
├── Implement GC%, CAI, and MFE folding calculations in src/features.py
└── Prototype pipeline skeleton in src/pipeline.py

Week 2: Baseline Model Development & Cross Validation
├── Build data cleaning and dataset splitting in src/data_processing.py
├── Train baseline XGBoost/RandomForest classifiers in src/model.py
├── Implement Stratified K-Fold Cross-Validation for small dataset reliability
└── Build accuracy, F1-score, and precision/recall evaluation metrics

Week 3: SHAP Integration & Inference Engine
├── Integrate SHAP TreeExplainer for feature importance interpretation
├── Build risk scoring heuristic logic in src/predict.py
├── Export serialized model artifacts (.joblib) to models/
└── Write automated unit test suite in tests/

Week 4: Streamlit Dashboard & Deployment
├── Build interactive UI in app/main.py
├── Integrate cached model loading and live SHAP visualizations
└── Finalize project documentation and handoff guidelines
```
