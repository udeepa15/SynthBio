# 📊 SynthBio Architecture & Flow Mermaid Diagrams

This document contains visual Mermaid architectural diagrams for all core subsystems of the **SynthBio** application, including end-to-end data flow, feature engineering routines, ML training & cross-validation, inference risk scoring, Streamlit UI serving, and Partner handoff workflows.

---

## 1. 🌐 Overall System High-Level Data Architecture

```mermaid
flowchart TD
    A["Raw DNA Sequences / FASTA / CSV"] --> B["Data Processing & Cleaning (src/data_processing.py)"]
    B --> C["Biophysical Feature Extraction Engine (src/features.py)"]
    
    subgraph Features ["Computed Core Features"]
        C --> C1["GC Content (%)"]
        C --> C2["Codon Adaptation Index (CAI)"]
        C --> C3["mRNA Folding Stability (MFE)"]
    end
    
    C1 & C2 & C3 --> D["Processed Feature Matrix & Target Tiers"]
    
    D --> E["Stratified 5-Fold Cross-Validation (src/model.py)"]
    E --> F["Train XGBoost / Random Forest Classifier"]
    F --> G["Serialize Model Artifact (models/baseline_model.joblib)"]
    
    G --> H["Streamlit Web Application (app/main.py)"]
    C1 & C2 & C3 --> I["Construct Risk Scoring Engine (src/predict.py)"]
    F --> J["SHAP TreeExplainer Local Interpretability"]
    
    H & I & J --> K["User Dashboard Output"]
    K --> K1["Construct Failure Risk Badge (0-100%)"]
    K --> K2["Predicted Expression Tier & Yield"]
    K --> K3["SHAP Feature Importance Bar Chart"]
```

---

## 2. 🧬 Feature Engineering Pipeline (`src/features.py`)

```mermaid
flowchart LR
    Seq["Input Sequence String (5' to 3')"] --> Clean["Upper Case & Whitespace Removal"]
    
    Clean --> GC["calculate_gc_content"]
    Clean --> CAI["calculate_cai"]
    Clean --> MFE["estimate_mrna_folding_stability"]
    
    GC --> GC_Logic["Count (G + C) / Total Length * 100"]
    CAI --> CAI_Logic["Match Codons against E. coli K-12 Table (w_i)<br/>Compute Geometric Mean: exp(sum(ln(w_i))/L)"]
    MFE --> MFE_Logic["Extract 5' Region (First 48 bases)<br/>Compute MFE: -1.5 * L * (GC_fraction^1.5)"]
    
    GC_Logic --> Out["Dict: {gc_content, codon_adaptation_index, mrna_folding_stability}"]
    CAI_Logic --> Out
    MFE_Logic --> Out
```

---

## 3. 🔄 ML Training & Validation Flow (`src/pipeline.py`)

```mermaid
flowchart TD
    Start["Launch Pipeline (python -m src.pipeline)"] --> Load["Stage 1: Load CSV (data/processed/feature_table.csv)"]
    Load --> Check{CSV Exists?}
    Check -- Yes --> Validate["Validate Schema, Nulls & Dtypes"]
    Check -- No --> Fake["Generate Synthetic Fallback Dataset"]
    
    Validate --> Split["Stage 2: Perform 80/20 Stratified Train/Test Split"]
    Fake --> Split
    
    Split --> CV["Stage 3: Perform Stratified 5-Fold Cross-Validation"]
    CV --> Metrics["Calculate Mean Accuracy & Macro F1 Score"]
    
    Metrics --> Train["Stage 4: Fit Final XGBoost / RandomForest Classifier"]
    Train --> Eval["Evaluate Predictions on 20% Test Set"]
    
    Eval --> Save["Stage 5: Export Artifact to models/baseline_model.joblib"]
    Save --> End["Pipeline Execution Complete (Exit Code 0)"]
```

---

## 4. 🎯 Inference Engine & Failure Risk Calculation (`src/predict.py`)

```mermaid
flowchart TD
    InputSeq["User Gene Sequence Input"] --> Ext["Extract 3 Core Features"]
    
    subgraph Penalties ["Failure Risk Scoring Heuristic"]
        Ext --> P1["GC Penalty: 1.5 * deviation outside [40%, 60%]"]
        Ext --> P2["CAI Penalty: 60.0 * (0.70 - CAI) if CAI < 0.70"]
        Ext --> P3["MFE Penalty: 2.0 * |MFE + 12.0| if MFE < -12.0 kcal/mol"]
    end
    
    P1 & P2 & P3 --> RiskScore["Sum Penalties -> Failure Risk Score (0.0% - 99.9%)"]
    
    Ext --> ModelCheck{Trained Model Loaded?}
    ModelCheck -- Yes --> MLPred["Run XGBoost Model Prediction"]
    ModelCheck -- No --> RulePred["Fallback Heuristic Tier Rule"]
    
    MLPred & RulePred --> TierOutput["Predicted Tier: High / Medium / Low"]
    
    Ext --> SHAPCheck{SHAP Explainer Available?}
    SHAPCheck -- Yes --> SHAPCalc["Calculate Local TreeExplainer SHAP Values"]
    SHAPCheck -- No --> HeuristicFI["Return Biological Heuristic Importances"]
    
    RiskScore & TierOutput & SHAPCalc & HeuristicFI --> ResultDict["Return Result Dictionary to UI"]
```

---

## 5. 🖥️ Streamlit Web Application Interactive Architecture (`app/main.py`)

```mermaid
flowchart TD
    User["User opens Streamlit App (streamlit run app/main.py)"] --> Init["Initialize Page Config & Layout"]
    Init --> Cache["Load Trained Model & SHAP Explainer (@st.cache_resource)"]
    
    Cache --> Sidebar["Render Sidebar Options & Model Status Badge"]
    Init --> InputArea["Render Text Area for 5' to 3' DNA Sequence"]
    
    InputArea --> Trigger{Button Clicked OR Sequence Changed?}
    Trigger -- Yes --> Predict["Call predict_construct(sequence, model, explainer)"]
    
    Predict --> MetricCards["Display Metrics: GC%, CAI, MFE (kcal/mol)"]
    Predict --> RiskBadge["Display Failure Risk Badge (Success / Warning / Error)"]
    Predict --> TierCard["Display Predicted Tier & Estimated Yield (AU)"]
    Predict --> BarChart["Render SHAP Feature Importance Bar Chart"]
```

---

## 6. 🤝 Partner 1 & Partner 2 Data Handoff Sync Workflow

```mermaid
flowchart LR
    subgraph Partner1 ["Partner 1 (Bio Track - Chamodya)"]
        P1_1["Ingest Raw Sequences"] --> P1_2["Run Sequence Feature Calculation"]
        P1_2 --> P1_3["Preserve 0-5 Target Scores"]
        P1_3 --> P1_4["Deliver CSV: data/processed/feature_table.csv"]
    end
    
    P1_4 ==> SyncPoint(("🔄 Week 2 Sync Decision"))
    
    subgraph Partner2 ["Partner 2 (AI/ML Track)"]
        SyncPoint -- Option A: Category --> ClassModel["XGBClassifier (objective='multi:softprob')<br/>Report Accuracy & Macro F1"]
        SyncPoint -- Option B: Number --> RegModel["XGBRegressor (objective='reg:squarederror')<br/>Report R^2 & MAE"]
        
        ClassModel & RegModel --> TrainPipe["Run K-Fold Cross Validation & Training"]
        TrainPipe --> Deploy["Export Model Artifact to models/ for Streamlit Dashboard"]
    end
```
