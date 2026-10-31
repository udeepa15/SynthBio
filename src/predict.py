"""
Inference & Risk Scoring Module:
Takes gene sequence inputs, computes core features, runs model prediction,
computes construct failure risk score (0-100%), and calculates feature importances via SHAP.
"""

import logging
from typing import Any, Dict, Optional, Tuple

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

# Which of the two paths produced the tier on screen.
MODEL_SOURCE = "trained_model"
HEURISTIC_SOURCE = "heuristic_rules"

# Which predictor arrangement ran. See `predict_construct`.
MODE_SINGLE_MODEL = "single_model"
MODE_AGENTS = "agents"

# Feature columns this module can compute from a pasted sequence, and therefore
# the only columns a model may be fitted on if the application is to serve it.
#
# Boel's own Gz48v and Gave48 are deliberately absent: Gz48v folds the
# expression vector's 5' UTR together with the gene head, so it depends on the
# vector rather than the construct. Measured on the full Boel set, leaving it
# out costs almost nothing (ranking 0.333 against 0.349, binary AUC 0.660
# against 0.666), so the vector dependency is not worth taking on.
SERVABLE_FEATURES = [
    "GC_percent",
    "GC3",
    "CAI",
    "rare_codon_fraction",
    "rare_codons_head30",
    "head_folding_48",
    "genewide_folding_avg",
    "gravy_score",
    "pI",
    "length_codons",
    "folding_energy",
]
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


def build_feature_row(features: Dict[str, float]) -> pd.DataFrame:
    """
    One-row frame of model feature columns, from an `extract_features` dict.

    The feature extractor and the model use different names for the same
    quantities, so this is the single place the two vocabularies meet. Anything
    that scores a sequence through the model goes through here, so that the
    redesign engine cannot drift from what the predictor feeds the model.

    `folding_energy` carries the historical GC-derived proxy, kept because
    artifacts fitted on it still load. `genewide_folding_avg` is the real
    ViennaRNA energy.
    """
    return pd.DataFrame([{
        "GC_percent": features["gc_content"],
        "GC3": features.get("gc3_content", np.nan),
        "CAI": features["codon_adaptation_index"],
        "rare_codon_fraction": features.get("rare_codon_fraction", np.nan),
        "rare_codons_head30": features.get("rare_codons_head30", np.nan),
        "head_folding_48": features.get("head_folding_48", np.nan),
        "genewide_folding_avg": features.get("genewide_folding_avg", np.nan),
        "gravy_score": features.get("gravy_score", np.nan),
        "pI": features.get("pI", np.nan),
        "length_codons": features.get("length_codons", np.nan),
        "folding_energy": features["mrna_folding_stability"],
    }])


def check_feature_ranges(
    feat_df: pd.DataFrame,
    ranges: Optional[Dict[str, Dict[str, float]]]
) -> list:
    """
    Features of this construct that fall outside the model's training range.

    A tree model extrapolates by repeating the nearest leaf it learned, so a
    value beyond the training range yields a confident-looking answer with
    nothing behind it. The caller is expected to surface these, not hide them.
    """
    if not ranges:
        return []

    outside = []
    for name, bounds in ranges.items():
        if name not in feat_df.columns:
            continue
        value = feat_df.iloc[0][name]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and value < low:
            outside.append({"feature": name, "value": float(value), "min": low, "max": high, "side": "below"})
        elif high is not None and value > high:
            outside.append({"feature": name, "value": float(value), "min": low, "max": high, "side": "above"})

    if outside:
        logger.warning(
            "Construct features outside the training range: "
            + ", ".join(f"{o['feature']}={o['value']} ({o['side']} {o['min']}..{o['max']})" for o in outside)
        )
    return outside


def predict_construct(
    sequence: str,
    model: Any = None,
    explainer: Any = None,
    feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    mode: str = MODE_SINGLE_MODEL,
    comparison: Optional[Tuple[str, ...]] = None,
    leader_5prime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full inference pipeline for a given gene construct sequence.

    `mode` selects which path produces the figures. `MODE_SINGLE_MODEL` is the
    behaviour this function has always had and stays the default.
    `MODE_AGENTS` additionally runs the four agents and the critic and returns
    the verdict under the `critic_verdict` key, without changing any existing
    key, so the interface can adopt the agent path one panel at a time.

    The switch is deliberately one way and is not yet flipped. The plan is to
    flip it once at least one agent has passed its gate, because two live code
    paths drift, and this project has already been bitten twice by a label
    inversion that only appeared under testing. No agent has passed a gate, so
    the default remains the single model.
    """
    logger.info(f"Starting construct prediction pipeline for input sequence ({len(sequence)} bp)...")
    
    if not sequence or not sequence.strip():
        logger.error("Empty or invalid sequence input provided to predict_construct.")
        raise ValueError("Sequence input cannot be empty.")
        
    features = extract_features(sequence)
    risk_score = calculate_failure_risk(features)
    
    # Prepare feature dataframe matching model expectations
    feat_df = build_feature_row(features)


    tier = None
    pred_yield = None
    pred_score = None
    confidence = None
    class_probabilities = None
    prediction_source = HEURISTIC_SOURCE
    unservable_features = []

    # Run model prediction if trained model artifact is available.
    #
    # An artifact can require features this module does not compute: the Boel
    # folding columns are RNAstructure energies over the vector 5' UTR, which no
    # sequence-only calculation here reproduces. That is a configuration
    # mismatch, not a runtime error, so it is reported by name rather than
    # surfacing as a bare KeyError and quietly reverting to the rules.
    if model is not None and hasattr(model, "predict"):
        model_cols = list(getattr(model, "feature_names_in_", feat_df.columns))
        unservable_features = [c for c in model_cols if c not in feat_df.columns]
        if unservable_features:
            logger.error(
                "Trained model requires features this pipeline cannot compute from a sequence: "
                f"{', '.join(unservable_features)}. Falling back to biophysical heuristic rules."
            )
    if model is not None and hasattr(model, "predict") and not unservable_features:
        try:
            logger.info("Executing ML model prediction on construct features...")
            eval_df = feat_df[model_cols]
            pred_class = int(model.predict(eval_df)[0])
            pred_score = pred_class
            if pred_class in TIER_MAPPING:
                tier = TIER_MAPPING[pred_class]
            else:
                tier = f"Tier {pred_class}"
            pred_yield = round(10.0 + (pred_class * 17.0), 1)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(eval_df)[0]
                classes = [int(c) for c in getattr(model, "classes_", range(len(probabilities)))]
                class_probabilities = {c: float(p) for c, p in zip(classes, probabilities)}
                confidence = float(max(probabilities))
            prediction_source = MODEL_SOURCE
            logger.info(
                f"ML Model Output -> Predicted Tier Score: {pred_score} ({tier}), Yield: {pred_yield} AU"
                + (f", Confidence: {confidence:.2f}" if confidence is not None else "")
            )
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
        
    # Local attribution for this construct, keyed by the model's own feature
    # names.
    #
    # What this replaced is worth recording. The display dict here used to map
    # five hardcoded labels onto SHAP values, with `folding_energy` among them.
    # `folding_energy` is not a model feature, so that row fell back to its
    # default of 0.20 for every sequence ever scored, and the six features that
    # do carry attribution were dropped. A constant presented as an attribution
    # is worse than no attribution, so the labels are gone and the values are
    # whatever the explainer actually returns.
    feature_importances: Dict[str, float] = {}
    if explainer is not None and model is not None:
        logger.info("Computing local SHAP feature attributions...")
        shap_df = feat_df[list(model.feature_names_in_)] if hasattr(model, "feature_names_in_") else feat_df
        feature_importances = {
            str(name): float(value)
            for name, value in (explain_prediction_shap(explainer, shap_df, pred_class) or {}).items()
        }

    result_summary = {
        "sequence_length": len(sequence),
        "features": features,
        "risk_score": risk_score,
        "predicted_score": pred_score,
        "predicted_tier": tier,
        "predicted_yield": pred_yield,
        "feature_importances": feature_importances,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "out_of_range_features": check_feature_ranges(feat_df, feature_ranges),
        "prediction_source": prediction_source,
        "unservable_features": unservable_features,
        "feat_df": feat_df,
        "mode": mode,
        "critic_verdict": None,
    }

    if mode == MODE_AGENTS:
        # Imported here rather than at module scope: the agents import this
        # module for `build_feature_row` and `check_feature_ranges`, so a top
        # level import would be circular.
        from src.agents.graph import assess

        result_summary["critic_verdict"] = assess(
            sequence,
            model=model,
            explainer=explainer,
            feature_ranges=feature_ranges,
            comparison=comparison or (),
            leader_5prime=leader_5prime,
        )

    logger.info("Construct prediction pipeline completed successfully.")
    return result_summary
