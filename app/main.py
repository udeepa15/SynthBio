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

from src.predict import MODE_AGENTS, predict_construct, build_feature_row
from src.model import load_model, get_shap_explainer, load_model_metadata
from src.features import fetch_ncbi_cds, FOLDING_WINDOW_NT, FOLDING_AVG_WINDOW_NT
from src.xai import render_shap_waterfall
from src.validation import validate_sequence, blocking_issues, advisory_issues, is_scorable
from src import provenance as prov_mod
from src import vectors
from src.agents.base import AGENT_ORDER
from src.redesign import (
    CONSTRAINED,
    CONTRADICTED,
    CORROBORATED,
    DEFAULT_CANDIDATES_PER_ROUND,
    DEFAULT_ROUNDS,
    WELCH_CALIBRATION,
    optimize_sequence,
    parse_conserved_regions,
    split_codons,
)

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


@st.cache_resource
def load_variant_agent():
    """
    Agent C, the within gene codon variant ranker, loaded once per process.

    The agent loads its own artifact and abstains when there is none or when
    the artifact's recorded gate result is not a pass, which is the current
    state: there is no within gene training data in the repository, so Gate C
    has not been run. While it abstains the redesign search keeps ranking on
    the tier model with the GC3 corroboration check as its guard.

    Loading it is cheap and failing to load it must not take the application
    down, so any error here is logged and the agent is dropped.
    """
    try:
        from src.agents.variant import VariantAgent

        agent = VariantAgent()
        logger.info(
            "Agent C loaded. Artifact present: "
            f"{agent.ranker is not None}. Gate: {agent.gate().status}."
        )
        return agent
    except Exception as err:
        logger.error(f"Agent C could not be loaded: {err}", exc_info=False)
        return None


@st.cache_resource
def load_structure_agent():
    """
    Agent D, the initiation structure agent, loaded once per process.

    Like Agent C it reads its own gate result and abstains unless it records a
    pass. Unlike Agent C it also needs an input: without a declared vector
    there is no leader, so there is no initiation window to fold and the agent
    has no opinion.
    """
    try:
        from src.agents.structure import StructureAgent

        agent = StructureAgent()
        logger.info(
            "Agent D loaded. Artifact present: "
            f"{agent.model is not None}. Gate: {agent.gate().status}."
        )
        return agent
    except Exception as err:
        logger.error(f"Agent D could not be loaded: {err}", exc_info=False)
        return None


def render_vector_input() -> str:
    """
    Ask which expression vector the construct is going into.

    This input exists because of a measurement rather than a preference. Boel's
    strongest 5' features fold the vector's untranslated leader together with
    the gene head, and the application could never compute them from a pasted
    coding sequence because it did not know the vector. Gate D tested whether
    asking is worth it, by checking that leader plus insert folding beats
    insert only folding on the same rows under an identity clustered split. It
    does, so the question is asked. If that gate had failed the input would
    have been dropped, and `reports/gates.md` records the margin.
    """
    agent = load_structure_agent()
    gate = agent.gate() if agent is not None else None

    with st.expander("Expression vector, for the 5' initiation window", expanded=False):
        if gate is None or not gate.passed:
            st.caption(
                "The structure agent has not passed its gate, so the vector is recorded but does "
                "not enter any prediction. "
                + (gate.summary if gate is not None else "")
            )
        else:
            st.caption(
                "The transcript a ribosome reads starts in the vector, not at your start codon. "
                "Declaring the vector lets the 5' folding energy be computed over the real "
                "initiation window, leader plus insert, rather than over the insert alone."
            )
        choice = st.selectbox(
            "Expression vector",
            vectors.choices(),
            index=0,
            format_func=lambda key: vectors.get(key).label,
        )
        custom = st.text_input(
            "Or paste the 5' leader yourself, from the transcription start to the base before ATG",
            value="",
            placeholder="Leave empty to use the vector above",
            help=(
                "A wrong leader produces a confident folding number with nothing behind it, so "
                "supply one only if you know it."
            ),
        ).strip()
        leader = vectors.resolve_leader(choice, custom)
        if leader:
            source = vectors.get(choice).source if not custom else "Supplied by you."
            st.caption(f"Leader in use: {len(leader)} nt. {source}")
        else:
            st.caption(
                "No vector declared, so the structure agent abstains and the 5' folding figures "
                "below are computed over the construct's own head only."
            )
    return leader


def render_agent_panel(verdict) -> None:
    """
    One row per agent: its question, whether it applied, and its gate result.

    Showing the gate result in the interface is unusual and is the strongest
    honesty signal this application can give. Three of the four agents abstain
    at the moment, and each says why, which is more informative than a blank.

    The tier and the risk badge above still come from the single model path.
    This panel reports the agents beside it rather than replacing it, because
    the plan only flips that switch once the tier bearing agent passes its own
    gate, and Agent A has not.
    """
    if verdict is None:
        return

    st.markdown("#### The agents")
    st.caption(
        "Each agent answers one question and is allowed to decline. An agent speaks only when it "
        "holds an artifact whose recorded gate is a pass, so an abstention here is a result and "
        "not an error. Every figure is in `reports/gates.md`."
    )

    for name in AGENT_ORDER:
        opinion = verdict.opinions.get(name)
        if opinion is None:
            continue
        gate = opinion.gate
        mark = "✅" if opinion.applies else "➖"
        with st.expander(f"{mark} {name.title()}: {opinion.question}", expanded=False):
            if opinion.applies:
                if opinion.proba and set(opinion.proba) == {0, 1}:
                    st.write(
                        f"Probability of the lowest expression band: **{opinion.proba[1]:.0%}**"
                    )
                elif opinion.score is not None:
                    st.write(f"Score: **{opinion.score:.3f}**")
                top = opinion.top_attribution(3)
                if top:
                    st.caption(
                        "Top attribution: "
                        + ", ".join(f"{feature} {value:+.3f}" for feature, value in top)
                    )
            else:
                st.caption(opinion.abstain_reason or "No reason recorded.")
            st.caption(gate.as_line())
            for note in opinion.domain.notes:
                st.caption(note)

    st.caption(
        f"Critic: {verdict.state_label}. {verdict.explanation} {verdict.agreement.as_line()}."
    )


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

    The risk figure and the tier now come from the same place when the agent
    path produced a verdict: the critic derives risk from the probability the
    agents put on the low expression band, so a tier and a risk can no longer
    contradict each other. The old biophysical penalty sum survives as a
    diagnostic and as agent input features, which is where it belongs, and is
    still used when no agent could answer.
    """
    verdict = res.get("critic_verdict")
    critic_risk = verdict.risk if verdict is not None and verdict.risk is not None else None
    risk = critic_risk if critic_risk is not None else res["risk_score"]
    risk_note = (
        verdict.risk_basis
        if critic_risk is not None
        else "A fixed penalty rule over GC, CAI and the folding proxy. It does not read the "
        "model's tier."
    )
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
    render_figure(tier_cols[2], f"{risk}%", "Construct failure risk", risk_note)
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

    # Second row: the features added after the Welch analysis. They are measured
    # on every construct and shown, but the deployed artifact was fitted before
    # they existed, so they do not yet enter the prediction.
    st.markdown("###### Additional measures")
    extra = st.columns(5)
    render_figure(
        extra[0], fmt(f.get("gc3_content"), "%"), "GC3, wobble position",
        "GC at the third base of each codon. The strongest single predictor of yield "
        "on Welch's synonymous variants.",
    )
    render_figure(
        extra[1], fmt(f.get("rare_codon_fraction")), "Rare codon fraction",
        "Share of slowly-read codons. A moderate share is associated with higher yield, "
        "not lower.",
    )
    render_figure(
        extra[2], fmt(f.get("genewide_folding_avg")), "Folding energy, kcal/mol",
        f"Real ViennaRNA minimum free energy, averaged over {FOLDING_AVG_WINDOW_NT} nt sliding "
        "windows. Unlike the proxy above, an actual thermodynamic fold.",
    )

    if f.get("genewide_folding_avg") is None or (
        isinstance(f.get("genewide_folding_avg"), float) and np.isnan(f["genewide_folding_avg"])
    ):
        st.caption(
            "Folding energy reads N/A because ViennaRNA is not installed. Install it with "
            "`pip install ViennaRNA` to compute real folding energies."
        )

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


def render_calibration_notice() -> None:
    """
    How the redesign objective behaved on the one dataset that can test it.

    Shown before any result, not folded away behind an expander, because a tier
    gain from this model reads as a result and on one of the two Welch series
    the objective runs against measured yield. A user who sees the gain without
    this line would draw a conclusion the evidence does not support.
    """
    lines = ", ".join(
        f"{row['series']} (n={row['n']}) {row['objective']:+.2f} for the model against "
        f"{row['gc3']:+.2f} for GC3"
        for row in WELCH_CALIBRATION
    )
    st.warning(
        "**Read the score with this in mind.** The model this search optimises against currently "
        "performs at the majority class baseline, so a rise in predicted tier is not by itself "
        "evidence of a rise in measured yield. On the Welch 2009 series of synonymous variants, "
        "where measured yield is available, the rank correlation of the objective against yield is "
        f"{lines}. The objective therefore points the right way in one series and the wrong way in "
        "the other, while GC3 points the right way in both. The corroboration check below compares "
        "the two on your construct.",
        icon="⚠️",
    )


def render_redesign_controls(normalised: str) -> dict:
    """The conserved-region lock, the two search constraints and the search budget."""
    n_codons = len(split_codons(normalised))

    conserved_text = st.text_input(
        "Amino acid indices to conserve",
        value="",
        placeholder="e.g. 45-60, 102, 210-215",
        help=(
            "Positions held fixed, counting the initial methionine as 1. Use this for a catalytic "
            "site, a binding motif or any region where the exact codons matter. Everything else is "
            "open to synonymous change. Leave empty to allow changes throughout."
        ),
    )

    spec = parse_conserved_regions(conserved_text, n_codons)
    for problem in spec.problems:
        st.warning(problem, icon="⚠️")
    if spec.regions:
        st.caption(
            f"Holding {len(spec.locked)} of {n_codons} codons fixed at amino acids {spec.described}. "
            "The start codon and any terminal stop codon are always held fixed as well."
        )

    hold_gc3 = st.checkbox(
        "Hold GC3 at or above the original",
        value=True,
        help=(
            "Forbids any swap that turns a G or C at the third codon position into an A or T. On the "
            "Welch synonymous variants GC3 tracks measured yield in both series while the model "
            "ranks one series backwards, so this stops the search buying a model gain by lowering "
            "GC3. Because agreement with GC3 is then built in, the corroboration check can no "
            "longer count it as independent evidence."
        ),
    )
    region_codons = int(st.number_input(
        "Limit changes to the first N codons (0 for the whole gene)",
        min_value=0, max_value=max(n_codons, 1), value=0, step=1,
        help=(
            "Restricts the redesign to the start of the gene, where the evidence for a synonymous "
            "effect is best: the fold over the ribosome binding region and rare codons at "
            "initiation. Everything after codon N stays exactly as you entered it."
        ),
    ))

    with st.expander("Search budget", expanded=False):
        st.caption(
            "Each candidate costs one full feature extraction, dominated by the ViennaRNA fold, so "
            "the run time scales with both of these and with the length of the construct."
        )
        budget_cols = st.columns(2)
        candidates = budget_cols[0].slider(
            "Candidates per round", min_value=10, max_value=100,
            value=DEFAULT_CANDIDATES_PER_ROUND, step=10,
        )
        rounds = budget_cols[1].slider(
            "Rounds", min_value=2, max_value=20, value=DEFAULT_ROUNDS, step=1,
        )

    return {
        "conserved_text": conserved_text, "candidates": candidates, "rounds": rounds,
        "hold_gc3": hold_gc3, "region_codons": region_codons,
    }


def render_tag_recommendations(result) -> None:
    """
    The failure modes a synonymous redesign cannot touch.

    GRAVY and the isoelectric point follow from the amino acid sequence, so a
    codon swap leaves both exactly where they were. Where either drives the
    failure, the remedy is a fusion tag, and saying so is the whole point of
    checking.
    """
    if not result.tag_recommendations:
        return

    drivers = " and ".join(rec.driver for rec in result.tag_recommendations)
    st.error(
        f"**Synonymous optimisation cannot fix this construct on its own.** The failure is driven by "
        f"{drivers}, which a codon redesign leaves unchanged by construction.",
        icon="⛔",
    )
    for rec in result.tag_recommendations:
        st.markdown(f"**{rec.driver}**")
        st.caption(rec.detail)
        st.info(f"**Recommendation.** {rec.recommendation}", icon="🧪")


def render_redesign_verdict(result) -> None:
    """Tier before and after, the corroboration check, then the evidence."""
    original, best = result.original, result.best

    st.markdown("#### Redesign outcome")
    cols = st.columns(4)
    render_figure(cols[0], str(original.tier_class), "Original tier", original.tier_label)
    render_figure(cols[1], str(best.tier_class), "Redesigned tier", best.tier_label)
    objective_note = (
        f"Objective {original.objective_score:.3f} to {best.objective_score:.3f}, on "
        f"{result.objective_label}."
    )
    if result.ranked_on_variant_agent:
        objective_note += (
            " Agent C ranks synonymous variants of one gene against each other, which is the "
            "question this search asks. Its score has no units and orders candidates only."
        )
    else:
        objective_note += (
            " That is the probability weighted mean over the six tiers. It comes from the model "
            "that separates one gene from another, which is not the question this search asks, so "
            "read the corroboration check below before acting on a gain."
        )
    render_figure(
        cols[2], f"{result.objective_gain:+.3f}", "Change in the objective", objective_note,
    )
    render_figure(
        cols[3], f"{result.codons_changed}", "Codons changed",
        f"Of {result.codons_mutable} that were open to change, from "
        f"{result.candidates_scored} candidates scored.",
    )

    if not result.improved:
        st.info(
            "The search did not find a synonymous variant the model scores above your construct. "
            "That is a result in itself: on this model's view the codon usage is already as good as "
            "this protein allows, so a redesign is not where the remedy lies.",
            icon="ℹ️",
        )

    corr = result.corroboration()
    if corr.verdict == CONTRADICTED:
        st.error(
            "**The two signals disagree, so do not synthesise this redesign.** " + corr.summary,
            icon="⛔",
        )
    elif corr.verdict == CORROBORATED:
        st.success("**Corroborated by GC3.** " + corr.summary, icon="✅")
    elif corr.verdict == CONSTRAINED:
        st.warning("**GC3 was held, so this is not independent corroboration.** " + corr.summary, icon="⚠️")
    else:
        st.info("**Corroboration inconclusive.** " + corr.summary, icon="ℹ️")

    st.caption(
        "The protein is unchanged. Every candidate is checked against the original translation "
        "before it is accepted, so the redesigned DNA encodes exactly the same amino acid sequence."
    )


def render_redesign_evidence(result) -> None:
    """The feature movement, the changed codons and the search trace."""
    st.markdown("#### What changed")
    st.caption(
        "Only the features a codon swap can move are listed. GRAVY and the isoelectric point are "
        "identical by construction and are therefore left out."
    )
    st.dataframe(result.feature_comparison(), hide_index=True, use_container_width=True)

    changed = result.changed_codon_table()
    if not changed.empty:
        with st.expander(f"The {len(changed)} codons that changed", expanded=False):
            st.dataframe(changed, hide_index=True, use_container_width=True)

    with st.expander("Search trace", expanded=False):
        st.caption(
            "The best objective value after each round. A flat run of rounds is the search failing "
            "to improve, at which point it halves the number of swaps per candidate and tries a "
            "finer step."
        )
        st.line_chart(pd.DataFrame({"Best objective": result.history}))

    st.markdown("#### Redesigned sequence")
    st.text_area(
        "Optimised coding DNA, 5' to 3'", value=result.best.sequence, height=160, disabled=True,
    )
    st.download_button(
        "Download the redesigned sequence (FASTA)",
        data=f">synthbio_redesign codons_changed={result.codons_changed}\n{result.best.sequence}\n",
        file_name="synthbio_redesign.fasta",
        mime="text/plain",
    )


def render_redesign(normalised: str, model, res: dict) -> None:
    """
    The redesign tab: search for a better synonymous variant of this construct.

    The run is gated on a button because it scores hundreds of candidates and
    takes tens of seconds, which is not something to repeat on every Streamlit
    rerun.
    """
    st.markdown(
        "Searches for a DNA sequence encoding the same protein that the model scores higher, by "
        "swapping codons for synonymous alternatives. The protein never changes."
    )

    if model is None:
        st.error(
            "**A redesign needs the trained model.** The search ranks candidates by the model's "
            "predicted tier, and this session is running on the biophysical fallback rules, which "
            "have no tier to optimise against. Train a model with the pipeline, or restore the "
            "artifact in models/, to enable the redesign.",
            icon="⛔",
        )
        return

    render_calibration_notice()
    controls = render_redesign_controls(normalised)

    run_key = (
        normalised, controls["conserved_text"], controls["candidates"], controls["rounds"],
        controls["hold_gc3"], controls["region_codons"],
    )
    cached = st.session_state.get("redesign_result")
    cached_key = st.session_state.get("redesign_key")

    if st.button("Redesign construct", type="primary"):
        status = st.empty()
        bar = st.progress(0.0)

        def report(round_index: int, rounds: int, score: float) -> None:
            bar.progress(round_index / rounds)
            status.caption(f"Round {round_index} of {rounds}. Best objective so far {score:.3f}.")

        logger.info(f"Redesign requested for {len(normalised)} nt, conserved='{controls['conserved_text']}'")
        try:
            result = optimize_sequence(
                normalised,
                model=model,
                conserved_text=controls["conserved_text"],
                candidates_per_round=controls["candidates"],
                rounds=controls["rounds"],
                hold_gc3=controls["hold_gc3"],
                region_codons=controls["region_codons"],
                progress=report,
                variant_agent=load_variant_agent(),
            )
        except ValueError as err:
            bar.empty()
            status.empty()
            st.error(f"The redesign could not run: {err}", icon="⛔")
            return
        except Exception as err:
            bar.empty()
            status.empty()
            logger.error(f"Redesign failed: {err}", exc_info=True)
            st.error(f"The redesign failed: {err}", icon="⛔")
            return

        bar.empty()
        status.empty()
        st.session_state["redesign_result"] = result
        st.session_state["redesign_key"] = run_key
        cached, cached_key = result, run_key

    if cached is None:
        st.info(
            "Select Redesign construct to run the search. It scores a few hundred candidate "
            "sequences, so expect it to take tens of seconds.",
            icon="ℹ️",
        )
        return

    if cached_key != run_key:
        st.info(
            "The sequence or the settings changed since the last redesign. Select Redesign "
            "construct to run it again.",
            icon="ℹ️",
        )
        return

    render_tag_recommendations(cached)
    st.divider()
    render_redesign_verdict(cached)
    st.divider()
    render_redesign_evidence(cached)


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

    # Input.
    #
    # A radio rather than tabs, because Streamlit renders every tab on every run
    # regardless of which one is open. With tabs, both inputs were always live
    # and the accession silently won, so a pasted sequence was scored as whatever
    # accession happened to be left in the other tab, and clearing the accession
    # silently reverted to the pre-filled sample. The source has to be one choice.
    st.markdown("#### Input sequence")
    source = st.radio(
        "Sequence source",
        ["Paste a sequence", "Fetch by NCBI accession"],
        horizontal=True,
        label_visibility="collapsed",
    )

    sequence_input = ""
    if source == "Paste a sequence":
        sequence_input = st.text_area(
            "Coding DNA sequence, 5' to 3'",
            value=SAMPLE_SEQUENCE,
            height=130,
            help="Must start at the start codon. A sample construct is pre-filled; replace it with your own.",
        )
    else:
        # The examples are single-gene records on purpose. An earlier revision
        # offered NC_000913, the E. coli K-12 chromosome, which fetches 4.6 Mb
        # and can only ever be rejected.
        accession_id = st.text_input(
            "NCBI accession number",
            value="",
            placeholder="e.g. NM_000207 (insulin), NM_000605, NM_000454",
            help=(
                "The annotated coding sequence is used, so an mRNA record whose "
                "untranslated regions would otherwise break the reading frame works. "
                "A genome or non-coding record has no CDS to take."
            ),
        ).strip()
        if accession_id:
            logger.info(f"NCBI accession fetch requested for '{accession_id}'")
            try:
                with st.spinner(f"Fetching '{accession_id}' from NCBI..."):
                    sequence_input, note = fetch_ncbi_cds(accession_id)
                st.success(note)
            except LookupError as err:
                logger.warning(f"No CDS on '{accession_id}': {err}")
                st.error(
                    f"{err} Scoring needs one gene's coding sequence, so a genome, plasmid or "
                    "non-coding record cannot be used. Open the record's annotation on NCBI and "
                    "take the CDS entry for the gene you want.",
                    icon="⛔",
                )
            except Exception as err:
                logger.error(f"NCBI lookup failed for '{accession_id}': {err}", exc_info=True)
                st.error(f"Could not fetch '{accession_id}' from NCBI: {err}")

    leader = render_vector_input()

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
            normalised,
            model=model,
            explainer=explainer,
            feature_ranges=training_ranges,
            mode=MODE_AGENTS,
            leader_5prime=leader or None,
        )
        logger.info(
            f"Prediction complete -> Risk: {res['risk_score']}%, Tier: {res['predicted_tier']}, "
            f"Tier midpoint: {res['predicted_yield']}"
        )
    except Exception as pipeline_err:
        logger.error(f"Prediction failed: {pipeline_err}", exc_info=True)
        st.error(f"The prediction could not be completed: {pipeline_err}")
        return

    # Two output tabs. Only the results are split this way, never the inputs:
    # Streamlit renders every tab on every run, so an input inside a tab stays
    # live whether or not the tab is open, which is the bug that made the
    # sequence source ambiguous before. Reading a rendered result costs nothing.
    diagnosis_tab, redesign_tab = st.tabs(["Diagnosis", "Redesign"])

    with diagnosis_tab:
        render_diagnosis(res, prov, model)
        render_agent_panel(res.get("critic_verdict"))

    with redesign_tab:
        render_redesign(normalised, model, res)


def render_diagnosis(res: dict, prov, model) -> None:
    """Vertical hierarchy: verdict, features, diagnostics, attribution."""
    render_verdict(res, prov)
    st.divider()
    render_features(res)
    st.divider()
    render_diagnostics(res)
    st.divider()

    st.markdown("#### Model attribution")
    feat_df = res.get("feat_df")
    if feat_df is None:
        feat_df = build_feature_row(res["features"])
    render_shap_waterfall(model, feat_df, predicted_class=res.get("predicted_score"))



if __name__ == "__main__":
    main()
