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

## 🤖 Multi agent predictor and the gates

The single model is being replaced by four narrow agents and a critic, following
`docs/synthbio-multi-agent-plan.pdf`. Each agent answers one question, carries the result
of its own named gate, and abstains when it has not passed it. An abstention is an
answer here, not a failure.

| Agent | Question | Status |
| :--- | :--- | :--- |
| A, expression | Will this protein express, and in which tier | Wraps the served artifact. Gate blocked: the clustered split needs the training sequences, which are not in the tree |
| B, solubility | Will it be soluble rather than inclusion bodies | Declared, not built. No solubility dataset on disk |
| C, codon variant | Of two synonymous versions of one gene, which expresses better | Built and wired in, abstaining. No within gene training data on disk, so Gate C has not been run |
| D, structure | Does the 5' region fold so as to block initiation | Declared, not built. Blocked by a missing input, the vector's 5' leader, rather than by a missing dataset |

`reports/gates.md` is the single source of truth for every numeric claim in this project.
Regenerate it, along with every baseline and every gate that can be run:

```bash
python scripts/gates.py
```

### The external datasets

Both downloads come from the authors' own deposits, are reduced on the way in to the
columns this project uses, and are gitignored because they are regenerable.

```bash
python scripts/fetch_datasets.py --source cambray
```

```bash
python scripts/fetch_datasets.py --source boel
```

Cambray 2018 is Agent C's training data: 244,000 designed sequences, of which 74,900 in
4,126 groups hold the protein exactly constant and vary only the codons. Boel 2016 rebuilds
`data/processed/feature_table_v3.csv`, the sequence bearing table the deployed model was
built from, which is what identity clustering and the reserved critic pool need:

```bash
python scripts/cluster.py
```

Agent C is the agent the plan puts first, because the within gene question is the one the
tier model demonstrably gets wrong. Train and gate it once a source is on disk; it writes
the gate result beside the artifact, and the agent reads that sidecar and stays quiet
unless it records a pass:

```bash
python scripts/train_agent_c.py
```

While Agent C abstains, the redesign search keeps ranking on the tier model with the GC3
corroboration check as its guard. Hand it a gated agent and it ranks on that instead,
which is the fix for the backwards ranking described above.

---

## 🧪 Running Tests

Execute unit tests to verify feature calculations and model outputs:
```bash
pytest tests/
```
