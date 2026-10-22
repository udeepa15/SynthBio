"""
Inference & Risk Scoring Module:
Takes gene sequence inputs, computes core features, runs model prediction,
computes construct failure risk score (0-100%), and calculates feature importances via SHAP.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from src.features import extract_features
from src.model import get_shap_explainer, explain_prediction_shap

# Configure logger
logger = logging.getLogger("SynthBio.predict")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.predict] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FEATURE_NAMES = ["gc_content", "codon_adaptation_index", "mrna_folding_stability"]
TIER_MAPPING = {
    0: "Very Low (Tier 0)",
    1: "Low (Tier 1)",
    2: "Medium-Low (Tier 2)",
    3: "Medium (Tier 3)",
    4: "High (Tier 4)",
    5: "Very High (Tier 5)"
}


def calculate_failure_risk(features: Dict[str, float]) -> float:
    """
    Calculate risk score (0 to 100%) indicating construct failure risk based on feature values:
    - Optimal GC content: 40-60% (Risk increases outside this range)
    - Optimal CAI: >= 0.70 (Low CAI increases risk)
    - Optimal MFE stability: >= -12.0 kcal/mol (Very negative MFE increases risk of hairpin blockages)
    """
    gc = features["gc_content"]
    cai = features["codon_adaptation_index"]
    mfe = features["mrna_folding_stability"]
    
    logger.info(f"Computing construct failure risk penalties for GC={gc}%, CAI={cai}, MFE={mfe} kcal/mol...")
    
    risk = 0.0
    gc_penalty = 0.0
    cai_penalty = 0.0
    mfe_penalty = 0.0
    
    # GC penalty
    if gc < 40.0:
        gc_penalty = (40.0 - gc) * 1.5
    elif gc > 60.0:
        gc_penalty = (gc - 60.0) * 1.5
    risk += gc_penalty
        
    # CAI penalty
    if cai < 0.70:
        cai_penalty = (0.70 - cai) * 60.0
    risk += cai_penalty
        
    # MFE penalty (tight folding)
    if mfe < -12.0:
        mfe_penalty = abs(mfe + 12.0) * 2.0
    risk += mfe_penalty
        
    final_risk = min(round(risk, 1), 99.9)
    logger.info(
        f"Risk Penalties -> GC Penalty: {gc_penalty:.1f}%, CAI Penalty: {cai_penalty:.1f}%, "
        f"MFE Penalty: {mfe_penalty:.1f}% => Total Risk Score: {final_risk}%"
    )
    return final_risk


def predict_construct(sequence: str, model: Any = None, explainer: Any = None) -> Dict[str, Any]:
    """Run full inference pipeline for a given gene construct sequence."""
    logger.info(f"Starting construct prediction pipeline for input sequence ({len(sequence)} bp)...")
    
    if not sequence or not sequence.strip():
        logger.error("Empty or invalid sequence input provided to predict_construct.")
        raise ValueError("Sequence input cannot be empty.")
        
    features = extract_features(sequence)
    risk_score = calculate_failure_risk(features)
    
    # Prepare feature dataframe matching model expectations
    feat_df = pd.DataFrame([{
        "GC_percent": features["gc_content"],
        "CAI": features["codon_adaptation_index"],
        "folding_energy": features["mrna_folding_stability"],
        "gravy_score": features.get("gravy_score", np.nan),
        "pI": features.get("pI", np.nan)
    }])
    
    tier = None
    pred_yield = None
    pred_score = None
    
    # Run model prediction if trained model artifact is available
    if model is not None and hasattr(model, "predict"):
        try:
            logger.info("Executing ML model prediction on construct features...")
            if hasattr(model, "feature_names_in_"):
                model_cols = list(model.feature_names_in_)
                eval_df = feat_df[model_cols]
            else:
                eval_df = feat_df
            pred_class = int(model.predict(eval_df)[0])
            pred_score = pred_class
            if pred_class in TIER_MAPPING:
                tier = TIER_MAPPING[pred_class]
            else:
                tier = f"Tier {pred_class}"
            pred_yield = round(10.0 + (pred_class * 17.0), 1)
            logger.info(f"ML Model Output -> Predicted Tier Score: {pred_score} ({tier}), Yield: {pred_yield} AU")
        except Exception as err:
            logger.error(f"ML model prediction failed: {str(err)}. Falling back to biophysical heuristic mode.", exc_info=False)

    # Baseline fallback predictions if model prediction was not executed
    if tier is None:
        logger.info("Using biophysical heuristic rules for tier and yield prediction...")
        if features["gc_content"] >= 45 and features["codon_adaptation_index"] >= 0.65:
            tier = "High"
            pred_yield = 85.0
            pred_score = 5
        elif features["gc_content"] >= 35 and features["codon_adaptation_index"] >= 0.45:
            tier = "Medium"
            pred_yield = 48.0
            pred_score = 3
        else:
            tier = "Low"
            pred_yield = 12.0
            pred_score = 1
        logger.info(f"Heuristic Output -> Tier: {tier}, Yield: {pred_yield} AU")
        
    # Compute SHAP feature importances if explainer is provided
    shap_importances = {}
    if explainer is not None:
        logger.info("Computing local SHAP feature attributions...")
        shap_importances = explain_prediction_shap(explainer, feat_df)
        
    if shap_importances:
        feature_importances = {
            "GC Content (%)": round(shap_importances.get("GC_percent", 0.25), 3),
            "Codon Adaptation Index (CAI)": round(shap_importances.get("CAI", 0.35), 3),
            "mRNA Folding Stability (MFE)": round(shap_importances.get("folding_energy", 0.20), 3),
            "GRAVY Score": round(shap_importances.get("gravy_score", 0.10), 3),
            "Isoelectric Point (pI)": round(shap_importances.get("pI", 0.10), 3)
        }
    else:
        logger.info("SHAP explainer not active; computing heuristic feature contribution weights...")
        gravy_val = features.get("gravy_score", 0.0)
        gravy_val = 0.0 if np.isnan(gravy_val) else gravy_val
        pi_val = features.get("pI", 7.0)
        pi_val = 7.0 if np.isnan(pi_val) else pi_val
        feature_importances = {
            "GC Content (%)": round(0.25 * (1 + (features["gc_content"] - 50)/100), 3),
            "Codon Adaptation Index (CAI)": round(0.35 * features["codon_adaptation_index"], 3),
            "mRNA Folding Stability (MFE)": round(0.20 * abs(features["mrna_folding_stability"])/20, 3),
            "GRAVY Score": round(0.10 * (abs(gravy_val) + 0.1), 3),
            "Isoelectric Point (pI)": round(0.10 * (pi_val / 14.0), 3)
        }
    
    result_summary = {
        "sequence_length": len(sequence),
        "features": features,
        "risk_score": risk_score,
        "predicted_score": pred_score,
        "predicted_tier": tier,
        "predicted_yield": pred_yield,
        "feature_importances": feature_importances
    }
    logger.info("Construct prediction pipeline completed successfully.")
    return result_summary
