# SynthBio: Construct Failure Predictor for *E. coli*

SynthBio is a machine learning pipeline and Streamlit application designed to predict the expression performance of synthetic gene constructs in *Escherichia coli*. Using computed sequence features—GC content (GC%), Codon Adaptation Index (CAI), and minimum free energy (MFE) mRNA folding stability score—SynthBio trains gradient boosted models (XGBoost/LightGBM) to forecast numeric expression yield and categorical expression tiers (High/Medium/Low), helping synthetic biologists flag high-risk constructs prior to laboratory synthesis.

---

## 📅 4-Week Project Roadmap

- **Week 1: Data Collection & Feature Engineering**
  - Ingest public gene expression datasets into `data/raw/`.
  - Build sequence analysis scripts in `src/features.py` (GC%, CAI, MFE folding stability).
  - Perform exploratory data analysis in `notebooks/01_eda.ipynb`.
- **Week 2: Baseline Model Development**
  - Implement data preprocessing & splitting pipeline in `src/data_processing.py`.
  - Train baseline XGBoost / LightGBM regressors and classifiers in `src/model.py`.
  - Prototype and tune hyperparameters in `notebooks/02_model_baseline.ipynb`.
- **Week 3: Feature Importance & Inference Module**
  - Integrate SHAP (SHapley Additive exPlanations) for global and local feature importance in `src/predict.py`.
  - Export serialized model artifacts to `models/`.
  - Implement automated unit tests in `tests/`.
- **Week 4: Streamlit Web Application & Finalization**
  - Build interactive UI in `app/main.py` to accept user gene sequences.
  - Display construct failure risk score, predicted yield/tier, and feature importance visuals.
  - Finalize documentation and verify full pipeline integration.

---

## 📂 Directory Structure Overview

```text
SynthBio/
├── data/              # Stores raw input datasets and finalized, preprocessed feature tables.
│   ├── raw/           # Holds untouched public and experimental sequence/expression datasets.
│   └── processed/     # Contains cleaned sequence features (GC%, CAI, MFE) ready for training.
├── notebooks/         # Contains Jupyter notebooks for exploratory analysis and model prototyping.
│   ├── 01_eda.ipynb   # Exploratory Data Analysis notebook.
│   └── 02_model_baseline.ipynb # Baseline model prototyping notebook.
├── src/               # Contains modular Python source code for data processing, modeling, and inference.
│   ├── __init__.py
│   ├── data_processing.py # Sequence parsing, cleaning, and dataset splitting.
│   ├── features.py    # GC%, Codon Adaptation Index (CAI), and MFE calculation routines.
│   ├── model.py       # Model training, cross-validation, and hyperparameter optimization routines.
│   └── predict.py     # Inference pipeline and SHAP feature importance extraction.
├── app/               # Houses the Streamlit web application and UI resources.
│   ├── main.py        # Entry point for the Streamlit dashboard.
│   └── assets/        # Visual assets and custom CSS styles for the web app.
├── models/            # Stores trained gradient boosting model artifacts and scalers.
├── config/            # Holds configuration settings for features and hyperparameters.
│   └── config.yaml    # Project setup and hyperparameter parameters.
├── tests/             # Contains automated unit tests for quality assurance.
│   ├── test_features.py # Unit tests for sequence feature calculations.
│   └── test_model.py    # Unit tests for model inference and risk scoring.
├── .gitignore         # Specifies files ignored by Git version control.
├── README.md          # Project overview, architectural setup, and documentation.
└── requirements.txt   # Dependencies and package requirements.
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/SynthBio.git
   cd SynthBio
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Streamlit App

Launch the interactive web application with:
```bash
streamlit run app/main.py
```

---

## 🧬 Synonymous Codon Redesign

The **Redesign** tab searches for a DNA sequence encoding the same protein that the
trained model scores higher, by swapping codons for synonymous alternatives. The model
is used as the scoring function and is not retrained. Amino acid positions entered in
"indices to conserve" are held fixed, as are the start codon and any terminal stop
codon, and every candidate is checked against the original translation before it is
accepted, so the protein never changes.

GRAVY and the isoelectric point follow from the amino acid sequence, so no synonymous
change can move them. Where either drives the failure, the tab recommends a fusion tag
instead of offering a codon redesign.

**Read the output with the calibration in mind.** The model performs at the majority
class baseline, and on the Welch 2009 synonymous variant series the objective tracks
measured yield in one series (+0.47) and runs against it in the other (-0.41), while
GC3 tracks it positively in both (+0.39, +0.64). The tab therefore reports a
corroboration check alongside every result, flagging a redesign whose model gain comes
with a fall in GC3. Reproduce these figures with:

```bash
python scripts/redesign_sanity_check.py
```

---

## 🧪 Running Tests

Execute unit tests to verify feature calculations and model outputs:
```bash
pytest tests/
```
