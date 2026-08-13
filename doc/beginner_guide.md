# 🐣 SynthBio Explained for Beginners (Zero-Domain-Knowledge Guide)

Welcome! If you have **no background in biology or machine learning**, this document is written specifically for you. It explains what **SynthBio** is, why it exists, and how the code works from start to finish using simple everyday analogies.

---

## 💡 The Big Picture in Plain English

### The Problem
Scientists often want to instruct bacteria (like *E. coli*) to produce useful proteins—such as insulin for medicine, enzymes for detergents, or eco-friendly materials. To do this, they write instructions using DNA sequences and order them from a DNA synthesis laboratory. 

However, making custom DNA costs real money and takes weeks. If the DNA sequence is poorly designed, the bacteria will fail to read it, producing zero protein and wasting the scientist's budget and time.

### The Solution (What SynthBio Does)
**SynthBio** acts like a **smart spell-checker and performance predictor for DNA**. 

Before spending money to manufacture DNA in a real lab, a scientist pastes their DNA sequence into SynthBio. Our software analyzes the sequence, runs it through a trained Machine Learning model, and predicts:
1. **Predicted Expression Yield**: Will it produce a **High**, **Medium**, or **Low** amount of protein?
2. **Failure Risk Score**: A percentage ($0\%$ to $100\%$) indicating how likely the sequence is to fail.
3. **Why it might fail**: A breakdown of which parts of the sequence are problematic.

---

## 🥑 Everyday Analogies That Make It Click

To understand how our code analyzes DNA, think of these 3 simple analogies:

| Biological Term | What It Means | Everyday Analogy |
| :--- | :--- | :--- |
| **DNA Sequence** | A string made of 4 letters: `A`, `T`, `C`, `G` | A **recipe book** written in a 4-letter alphabet. |
| **Bacteria (*E. coli*)** | The organism reading the DNA | The **chef** reading the recipe book to cook a dish. |
| **1. GC Content (%)** | The ratio of `G` & `C` letters to total letters | **How sticky the pages are.** If the pages are too sticky ($>60\%$) or too slippery ($<40\%$), the chef struggles to read them. |
| **2. Codon Adaptation Index (CAI)** | How familiar the words are to the bacteria | **Local slang vs. foreign words.** If the recipe uses words the chef knows well, cooking is fast. If it uses obscure foreign dialect words, cooking slows down or stops. |
| **3. mRNA Folding Energy (MFE)** | Secondary structure knots at the start of the sequence | **A tangled knot on the front cover.** If the very first page is tangled up (tight negative energy), the chef can't open the book to start cooking. |

---

## 🏃 The Complete Journey of Code (From Beginning to End)

Here is exactly what happens inside the codebase step-by-step when the program runs:

```
[User Pastes Sequence] 
       │
       ▼
[1. Feature Extraction]  ──► Counts letters: calculates GC%, CAI (slang score), and MFE (knot score).
       │
       ▼
[2. ML Brain & Pipeline] ──► Passes numbers into XGBoost model trained on historical construct results.
       │
       ▼
[3. Risk Engine]        ──► Calculates penalty points and generates a 0-100% Failure Risk Score.
       │
       ▼
[4. Web Dashboard]      ──► Displays colorful Green/Yellow/Red badges and bar charts on the screen.
```

---

### Step 1: Input Sequence Processing ([`src/features.py`](file:///d:/UDEEPA/Projects/SynthBio/src/features.py))
When a user inputs a raw sequence string (e.g., `"ATGAAACGCATTAGC..."`):
1. **Cleaning**: The code strips away spaces, newlines, and converts all letters to uppercase.
2. **GC Calculation**: It counts `G` and `C` bases and divides by total sequence length.
3. **CAI Calculation**: It breaks the sequence into 3-letter words ("codons") and checks how frequently *E. coli* uses each word.
4. **MFE Calculation**: It inspects the first 48 letters (the 5' region) to estimate if a physical knot will block translation.

---

### Step 2: Training the Machine Learning Model ([`src/model.py`](file:///d:/UDEEPA/Projects/SynthBio/src/model.py) & [`src/pipeline.py`](file:///d:/UDEEPA/Projects/SynthBio/src/pipeline.py))
Before making live predictions, the system needs to learn what successful constructs look like:
1. **Data Loading**: Reads historical feature data from [`data/processed/feature_table.csv`](file:///d:/UDEEPA/Projects/SynthBio/data/processed/feature_table.csv).
2. **Train/Test Split**: Holds out $20\%$ of the data as a secret test to verify the model later.
3. **5-Fold Cross-Validation**: Splits the training data into 5 chunks, testing the model 5 times to ensure it doesn't just memorize the data, but truly learns the patterns.
4. **Model Fitting**: Trains an **XGBoost Classifier** (a decision tree model) to map sequence features to High, Medium, or Low expression tiers.
5. **Saving Memory**: Saves the trained model to disk at [`models/baseline_model.joblib`](file:///d:/UDEEPA/Projects/SynthBio/models/baseline_model.joblib).

---

### Step 3: Making Predictions & Calculating Failure Risk ([`src/predict.py`](file:///d:/UDEEPA/Projects/SynthBio/src/predict.py))
When a user wants to test a new DNA sequence:
1. **Model Prediction**: The saved XGBoost model evaluates the 3 computed features and assigns a predicted tier (**High**, **Medium**, or **Low**).
2. **Risk Heuristic**: The code checks for danger zones:
   - Is GC content $<40\%$ or $>60\%$? Add penalty.
   - Is CAI score $<0.70$? Add penalty.
   - Is MFE folding energy $<-12.0\text{ kcal/mol}$? Add penalty.
   - The total penalty is capped at $99.9\%$ to give a clear **Failure Risk Score**.
3. **SHAP Feature Importance**: Evaluates how much each specific feature contributed to the final score.

---

### Step 4: Displaying Results on the Screen ([`app/main.py`](file:///d:/UDEEPA/Projects/SynthBio/app/main.py))
The user interacts with a clean web interface powered by **Streamlit**:
- **Green Badge (0-30% Risk)**: "Low Risk of Failure" - Safe to synthesize!
- **Yellow Badge (30-60% Risk)**: "Moderate Failure Risk" - Proceed with caution.
- **Red Badge (>60% Risk)**: "High Failure Risk" - Redesign construct before spending money!
- **Bar Chart**: Shows which feature (GC%, CAI, or MFE) needs optimization.

---

## 🗺️ Project File Directory Tour

If you open the project folder, here is what each directory does in one sentence:

- **[`app/`](file:///d:/UDEEPA/Projects/SynthBio/app)**: Contains `main.py` which builds the user-friendly web interface dashboard.
- **[`src/`](file:///d:/UDEEPA/Projects/SynthBio/src)**: The core engine rooms containing Python scripts for feature math (`features.py`), dataset setup (`data_processing.py`), model logic (`model.py`), inference (`predict.py`), and pipeline execution (`pipeline.py`).
- **[`data/`](file:///d:/UDEEPA/Projects/SynthBio/data)**: Holds raw input CSV files (`data/raw/`) and cleaned sequence feature tables (`data/processed/`).
- **[`models/`](file:///d:/UDEEPA/Projects/SynthBio/models)**: Stores saved trained model files (`baseline_model.joblib`) so the web app can load them instantly.
- **[`config/`](file:///d:/UDEEPA/Projects/SynthBio/config)**: Contains `config.yaml` with settings like default learning rates and target host preferences.
- **[`tests/`](file:///d:/UDEEPA/Projects/SynthBio/tests)**: Contains automated test scripts to ensure math and code don't break when edits are made.

---

## 🚀 How to Run the Code (2 Simple Commands)

You don't need to write any code to test it! Run these two terminal commands:

1. **Train the ML model & run cross-validation:**
   ```bash
   python -m src.pipeline
   ```

2. **Launch the web application dashboard:**
   ```bash
   streamlit run app/main.py
   ```
