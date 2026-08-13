# 🤖 Machine Learning Model Architecture & Inference

## 📐 Model Selection & Algorithms
SynthBio utilizes gradient boosted decision trees as its core predictive architecture, with automatic fallback support for Random Forest ensembles when optional libraries are unavailable.

- **Primary Classifier**: `XGBoost` (`XGBClassifier`)
- **Fallback Classifier**: `RandomForestClassifier` (`sklearn.ensemble`)
- **Baseline Dummy Classifier**: `DummyClassifier` (`strategy="most_frequent"`) for pipeline sanity checks

---

## ⚙️ Model Hyperparameters & Configuration
Default model parameters are managed via [`config/config.yaml`](file:///d:/UDEEPA/Projects/SynthBio/config/config.yaml):

```yaml
model:
  type: "xgboost"
  task: "classification"
  target_column: "expression_tier" # High (2), Medium (1), Low (0)
  params:
    n_estimators: 100
    learning_rate: 0.05
    max_depth: 4
    random_state: 42
```

---

## 🔄 Cross-Validation Strategy
Given that biological construct datasets are often small and costly to produce in early development stages, SynthBio employs **Stratified $K$-Fold Cross-Validation** ($K=5$ folds default) in [`src/model.py`](file:///d:/UDEEPA/Projects/SynthBio/src/model.py):

- **Stratification**: Preserves class distribution proportions across every training/validation fold.
- **Metrics Tracked**:
  - Mean & Standard Deviation of **Accuracy**
  - **Macro F1-Score** ($F_1$)
  - **Macro Precision & Recall**

---

## 📊 Construct Failure Risk Scoring Heuristic
In addition to model tier predictions, SynthBio computes a continuous **Construct Failure Risk Score ($0.0\% - 99.9\%$)** in [`src/predict.py`](file:///d:/UDEEPA/Projects/SynthBio/src/predict.py) based on biophysical boundary violations:

$$\text{Risk} = \text{Penalty}_{\text{GC}} + \text{Penalty}_{\text{CAI}} + \text{Penalty}_{\text{MFE}}$$

- **GC Content Penalty**:
  - If $\text{GC} < 40\%$: $\text{Penalty} = (40.0 - \text{GC}) \times 1.5$
  - If $\text{GC} > 60\%$: $\text{Penalty} = (\text{GC} - 60.0) \times 1.5$
- **CAI Penalty**:
  - If $\text{CAI} < 0.70$: $\text{Penalty} = (0.70 - \text{CAI}) \times 60.0$
- **MFE Secondary Structure Penalty**:
  - If $\text{MFE} < -12.0\text{ kcal/mol}$: $\text{Penalty} = |\text{MFE} + 12.0| \times 2.0$

---

## 🔍 SHAP (SHapley Additive exPlanations) Integration
To ensure model predictions are interpretable for synthetic biologists, SynthBio integrates `shap.TreeExplainer`:

- **Global Importance**: Calculates mean absolute SHAP values across feature columns (`GC_percent`, `CAI`, `folding_energy`).
- **Local Explanation**: Explains individual construct predictions by measuring how much each sequence feature pushes the prediction toward High, Medium, or Low expression.
- **Fallback**: If SHAP is unavailable, a biological weighted heuristic is dynamically returned for seamless UI rendering.
