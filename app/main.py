"""
Streamlit interface for SynthBio: Construct Failure Predictor.

Layout is a single vertical hierarchy: one verdict, then its provenance, then
the five computed features, then the biological diagnostics, then the model's
feature attribution. There is deliberately no two-column split of verdict
against features, so that the screen carries one answer rather than two.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logger
logger = logging.getLogger("SynthBio.app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.app] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from src.predict import predict_construct
from src.model import load_model, get_shap_explainer, load_model_metadata
from src.features import fetch_ncbi_sequence, FOLDING_WINDOW_NT
from src.xai import render_shap_waterfall
from src.validation import validate_sequence, blocking_issues, advisory_issues, is_scorable
from src import provenance as prov_mod

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "baseline_model.joblib")

SAMPLE_SEQUENCE = (
    "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
)

# Verdict states. Each carries a text label and a distinct glyph, so the state
# is legible without relying on the colour of the band.
VERDICT_STATES = {
    "low": {"label": "LOW RISK", "glyph": "●", "icon": "✅"},
    "moderate": {"label": "MODERATE RISK", "glyph": "◆", "icon": "⚠️"},
    "high": {"label": "HIGH RISK", "glyph": "▲", "icon": "⛔"},
}


@st.cache_resource
def load_trained_artifacts():
    """
    Load the model, its SHAP explainer and the training feature ranges.

    The ranges come from the metadata sidecar the pipeline writes beside the
    artifact; they are what lets the interface say when a construct sits outside
    anything the model was fitted on.
    """
    logger.info(f"Checking for trained model artifact at '{MODEL_PATH}'...")
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            logger.info("Model loaded. Initializing SHAP TreeExplainer...")
            explainer = get_shap_explainer(model)
            metadata = load_model_metadata(MODEL_PATH) or {}
            return model, explainer, metadata.get("feature_ranges")
        except Exception as err:
            logger.error(f"Failed to load model artifact: {err}", exc_info=True)
            logger.warning("Falling back to biophysical heuristic prediction mode.")
            return None, None, None
    logger.warning(f"No model artifact at '{MODEL_PATH}'. Running in heuristic mode.")
    return None, None, None


def render_figure(container, value: str, label: str, note: str = "") -> None:
    """
    One figure with its full label.

    st.metric truncates a long label to a single clipped line, which is how
    "5' folding proxy (48 nt)" became "5' folding proxy …" and hid the very
    qualification it exists to carry. A heading plus a caption wraps instead,
    so the label stays readable at any column width, using native elements
    rather than injected CSS.
    """
    container.markdown(f"### {value}")
    container.caption(label)
    if note:
        container.caption(note)


def risk_state(risk: float) -> str:
    """Map the risk figure to a verdict state."""
    if risk < 30.0:
        return "low"
    if risk < 60.0:
        return "moderate"
    return "high"


def render_verdict(res: dict, prov) -> None:
    """
    The single verdict band: tier, risk state and provenance.

    Note: the risk figure and the tier are computed independently today, the
    risk from a biophysical penalty sum and the tier from the model. Making
    one quantity drive both is tracked as a separate change; this function is
    the one place the band is rendered, so that change lands here.
    """
    risk = res["risk_score"]
    state = risk_state(risk)
    meta = VERDICT_STATES[state]

    st.markdown(f"### {meta['glyph']} {meta['label']} — construct failure risk {risk}%")

    banner = {"low": st.success, "moderate": st.warning, "high": st.error}[state]
    banner(
        f"**{meta['label']}** ({risk}% failure risk)  ·  "
        f"Predicted expression tier: **{res['predicted_tier']}**",
        icon=meta["icon"],
    )

    confidence = res.get("confidence")
    tier_cols = st.columns(4 if confidence is not None else 3)
    render_figure(tier_cols[0], str(res["predicted_tier"]), "Predicted expression tier")
    render_figure(
        tier_cols[1], f"{res['predicted_yield']}", "Tier midpoint (arbitrary units)",
        "Not a measured yield. It is the nominal midpoint assigned to the predicted tier.",
    )
    render_figure(
        tier_cols[2], f"{risk}%", "Construct failure risk",
        "A fixed penalty rule over GC, CAI and the folding proxy. It does not read the model's tier.",
    )
    if confidence is not None:
        render_figure(
            tier_cols[3], f"{confidence * 100:.0f}%", "Confidence in the tier",
            "The model's own probability for the tier above. With six tiers, chance alone is 17%.",
        )

    render_unservable_notice(res)
    render_out_of_range_notice(res)

    st.caption(prov.as_line())
    for note in prov.caveats():
        st.caption(f"Note: {note}")


def render_unservable_notice(res: dict) -> None:
    """
    Say when the loaded artifact could not be used at all.

    A model fitted on the Boel folding columns needs RNAstructure energies over
    the vector 5' UTR, which cannot be derived from the pasted sequence alone. The
    tier then comes from the fixed rules, and the provenance line would otherwise
    still name a trained model.
    """
    missing = res.get("unservable_features") or []
    if not missing:
        return

    st.error(
        "**The trained model was not used.** It was fitted on "
        f"{', '.join(f'`{name}`' for name in missing)}, which cannot be computed from a sequence "
        "alone. The tier above comes from the fixed biophysical rules instead, so it carries no "
        "information from the training data. Retrain on features this application can compute, or "
        "supply the missing ones, before relying on the tier.",
        icon="⛔",
    )


def render_out_of_range_notice(res: dict) -> None:
    """
    Say when a construct sits outside the data the model was fitted on.

    A tree model answers such a construct by reusing the nearest leaf it learned,
    which looks like a prediction but carries no evidence, so the figure above is
    marked as out of scope rather than presented as a result.
    """
    outside = res.get("out_of_range_features") or []
    if not outside:
        return

    parts = [
        f"**{o['feature']}** is {o['value']:.3g}, {o['side']} the training range "
        f"{o['min']:.3g} to {o['max']:.3g}"
        for o in outside
    ]
    st.warning(
        "**Outside the model's training range.** " + "; ".join(parts) + ". "
        "The model has seen nothing like this construct, so treat the tier above as out of scope "
        "rather than as a prediction. The failure risk and the diagnostics below are fixed rules "
        "and are unaffected.",
        icon="⚠️",
    )


def render_features(res: dict) -> None:
    """The five computed features, full width, labelled for what they are."""
    st.markdown("#### Computed features")

    f = res["features"]
    gravy_score = f.get("gravy_score")
    pi_value = f.get("pI")

    def fmt(value, suffix=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value}{suffix}"

    cols = st.columns(5)
    render_figure(cols[0], fmt(f["gc_content"], "%"), "GC content", "Whole transcript.")
    render_figure(
        cols[1], fmt(f["codon_adaptation_index"]), "Codon adaptation index (CAI)",
        "Against a simplified E. coli K-12 codon table.",
    )
    render_figure(
        cols[2], fmt(f["mrna_folding_stability"]), f"5' folding proxy, kcal/mol ({FOLDING_WINDOW_NT} nt)",
        "GC-derived proxy, not a thermodynamic MFE. No folding model is evaluated.",
    )
    render_figure(
        cols[3], fmt(gravy_score), "GRAVY (hydrophobicity)",
        "Grand average of hydropathy over the translated product.",
    )
    render_figure(cols[4], fmt(pi_value), "Isoelectric point (pI)", "Of the translated product.")

    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (gravy_score, pi_value)):
        st.caption(
            "GRAVY and pI read N/A because the sequence could not be translated to a standard "
            "amino-acid string. The diagnostics below that depend on them are therefore omitted."
        )


def render_diagnostics(res: dict) -> None:
    """Hardcoded biological heuristics, stated as heuristics."""
    st.markdown("#### Biological construct diagnostics")
    st.caption("Fixed threshold rules applied to the computed features, independent of the model.")

    f = res["features"]
    gravy_score = f.get("gravy_score")
    pi_value = f.get("pI")
    has_warning = False

    if gravy_score is not None and not np.isnan(gravy_score) and gravy_score > 0.4:
        has_warning = True
        st.warning(
            f"**Inclusion body risk.** GRAVY is {gravy_score}, above the 0.4 threshold. "
            "The product is hydrophobic enough that it is likely to partition into insoluble "
            "inclusion bodies rather than stay in the soluble fraction.",
            icon="⚠️",
        )

    if pi_value is not None and not np.isnan(pi_value) and (7.2 <= pi_value <= 7.8):
        has_warning = True
        st.warning(
            f"**Aggregation risk.** The pI of {pi_value} falls inside the E. coli cytoplasmic pH range "
            "(7.2 to 7.8), so the product carries close to no net charge under expression conditions "
            "and is prone to aggregation. A charged fusion tag would shift the pI away from this window.",
            icon="⚠️",
        )

    cai_val = f["codon_adaptation_index"]
    mfe_val = f["mrna_folding_stability"]
    pred_score = res.get("predicted_score")
    if pred_score is None:
        pred_score = 0 if "Low" in str(res["predicted_tier"]) else (5 if "High" in str(res["predicted_tier"]) else 3)

    if cai_val > 0.85 and pred_score <= 2 and mfe_val < -15.0:
        has_warning = True
        logger.warning("Over-optimization trap detected for user construct.")
        st.warning(
            f"**Over-optimisation trap.** CAI is {cai_val}, which is high, yet the predicted tier is "
            f"{res['predicted_tier']}. Aggressive codon optimisation has raised GC content and so the "
            "5' folding proxy to " f"{mfe_val} kcal/mol, consistent with hairpins that obstruct ribosome "
            "entry. Relaxing CAI to open the 5' region is the usual remedy.",
            icon="⚠️",
        )

    if not has_warning:
        st.success(
            "**No threshold exceeded.** GRAVY, pI and the codon-optimisation check are all inside their "
            "normal ranges. This covers only those rules; it is not a prediction of success.",
            icon="✅",
        )


def render_validation_issues(issues) -> None:
    """Show each validation problem with its cause and its remedy."""
    for issue in blocking_issues(issues):
        st.error(f"**{issue.title}.** {issue.detail} {issue.remedy}", icon="⛔")
    for issue in advisory_issues(issues):
        st.warning(f"**{issue.title}.** {issue.detail} {issue.remedy}", icon="⚠️")


def main():
    st.set_page_config(
        page_title="SynthBio | Construct Failure Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    model, explainer, training_ranges = load_trained_artifacts()
    prov = prov_mod.describe(model, MODEL_PATH if model is not None else None)

    # Sidebar
    st.sidebar.header("Settings")
    st.sidebar.selectbox("Target host", ["Escherichia coli (K-12)", "Escherichia coli (BL21)"], index=0)
    st.sidebar.markdown("**Prediction source**")
    st.sidebar.write(prov.mode_label)
    if prov.is_trained:
        st.sidebar.caption(
            f"{prov.model_class}, built {prov.model_built:%Y-%m-%d}"
            if prov.model_built else f"{prov.model_class}"
        )
    else:
        st.sidebar.caption("Train a model with the pipeline to replace the heuristic rules.")

    st.title("SynthBio: Construct Failure Predictor")
    st.markdown(
        "Predicts *E. coli* expression tier and construct failure risk from five computed features: "
        "GC content, codon adaptation index, a 5' folding proxy, GRAVY and isoelectric point."
    )

    # Input
    st.markdown("#### Input sequence")
    tab_raw, tab_ncbi = st.tabs(["Raw DNA sequence", "NCBI accession"])

    sequence_input = ""
    with tab_raw:
        raw_seq = st.text_area(
            "Coding DNA sequence, 5' to 3'",
            value=SAMPLE_SEQUENCE,
            height=130,
            help="Must start at the start codon. A sample construct is pre-filled; replace it with your own.",
        )
        sequence_input = raw_seq

    with tab_ncbi:
        accession_id = st.text_input(
            "NCBI accession number",
            value="",
            placeholder="e.g. M19447, J04639.1, NC_000913",
        ).strip()
        if accession_id:
            logger.info(f"NCBI accession fetch requested for '{accession_id}'")
            try:
                with st.spinner(f"Fetching '{accession_id}' from NCBI..."):
                    fetched_seq = fetch_ncbi_sequence(accession_id)
                if fetched_seq:
                    st.success(f"Fetched {len(fetched_seq)} bp from NCBI.")
                    sequence_input = fetched_seq
            except Exception as err:
                logger.error(f"NCBI lookup failed for '{accession_id}': {err}", exc_info=True)
                st.error(f"Could not fetch '{accession_id}' from NCBI: {err}")

    normalised, issues = validate_sequence(sequence_input)

    if normalised:
        with st.expander(f"Sequence preview ({len(normalised)} nt)", expanded=False):
            st.text_area("Normalised sequence", value=normalised, height=120, disabled=True)

    run_btn = st.button("Analyse construct", type="primary")

    # Gate the run on an explicit click. The sequence box ships with a sample,
    # so without this the app would predict on page load and the button would
    # have no effect.
    if run_btn:
        st.session_state["analysis_requested"] = True
        st.session_state["analysed_sequence"] = normalised

    if not st.session_state.get("analysis_requested"):
        st.info("Enter a sequence and select Analyse construct to score it.", icon="ℹ️")
        return

    # A sequence edited after the last run should not silently show stale results.
    if st.session_state.get("analysed_sequence") != normalised:
        st.info("The sequence has changed since the last analysis. Select Analyse construct to rescore.", icon="ℹ️")
        return

    st.divider()

    if issues:
        render_validation_issues(issues)

    if not is_scorable(issues):
        st.caption("Scoring is withheld until the problems above are resolved, so that no figure is shown "
                   "that the input cannot support.")
        return

    logger.info(f"Running prediction for {len(normalised)} nt sequence...")
    try:
        res = predict_construct(
            normalised, model=model, explainer=explainer, feature_ranges=training_ranges
        )
        logger.info(
            f"Prediction complete -> Risk: {res['risk_score']}%, Tier: {res['predicted_tier']}, "
            f"Tier midpoint: {res['predicted_yield']}"
        )
    except Exception as pipeline_err:
        logger.error(f"Prediction failed: {pipeline_err}", exc_info=True)
        st.error(f"The prediction could not be completed: {pipeline_err}")
        return

    # Vertical hierarchy: verdict, features, diagnostics, attribution.
    render_verdict(res, prov)
    st.divider()
    render_features(res)
    st.divider()
    render_diagnostics(res)
    st.divider()

    st.markdown("#### Model attribution")
    feat_df = pd.DataFrame([{
        "GC_percent": res["features"]["gc_content"],
        "CAI": res["features"]["codon_adaptation_index"],
        "folding_energy": res["features"]["mrna_folding_stability"],
        "gravy_score": res["features"].get("gravy_score", np.nan),
        "pI": res["features"].get("pI", np.nan),
    }])
    render_shap_waterfall(model, feat_df, predicted_class=res.get("predicted_score"))

    with st.expander("Feature contribution for this construct", expanded=False):
        st.caption(
            "Per-construct contributions, not global importances. With a trained model these are the "
            "SHAP values for this sequence; in heuristic mode they are weights derived from the feature "
            "values themselves and carry no model information."
        )
        fi_df = pd.DataFrame(
            list(res["feature_importances"].items()),
            columns=["Feature", "Contribution"],
        ).sort_values("Contribution", ascending=True)
        st.bar_chart(fi_df.set_index("Feature"))


if __name__ == "__main__":
    main()
