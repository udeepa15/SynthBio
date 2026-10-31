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
from src.agents.critic import AGREED, AGREED_OUT_OF_DOMAIN, CONFLICTED, NOT_ANSWERABLE
from app import agents_view, sequence_map, ui
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
    """One figure with its full label, through the shared card primitive."""
    ui.figure(container, value, label, note)


# How each critic state is rendered: the band colour, the headline, and the
# action the screen is arranged around. The conflicted state is the one this
# architecture exists to be able to draw, so it gets the same weight as an
# answer rather than the styling of an error.
VERDICT_LAYOUT = {
    AGREED: {"state": "good", "action": "Open the Redesign tab to look for a better codon choice."},
    AGREED_OUT_OF_DOMAIN: {
        "state": "caution",
        "action": "Treat the tier as an extrapolation. The evidence below says which feature is "
                  "outside which range.",
    },
    CONFLICTED: {
        "state": "bad",
        "action": "Do not read a tier off this. Measure one construct and come back.",
    },
    NOT_ANSWERABLE: {
        "state": "neutral",
        "action": "Nothing trained could answer. Only the validated GRAVY rule is left.",
    },
}

TIER_SENTENCE = {
    0: "The agents put this construct in the lowest expression band.",
    1: "The agents put this construct in the lowest expression band.",
    2: "The agents put this construct in the middle of the range.",
    3: "The agents put this construct in the middle of the range.",
    4: "The agents put this construct in the highest expression band.",
    5: "The agents put this construct in the highest expression band.",
}


def verdict_headline(res: dict, verdict) -> tuple:
    """
    The one statement at the top of the screen, and the sentence under it.

    Driven by the critic's four states rather than by the risk percentage. A
    percentage is an input to a decision; the state is the decision, which is
    why it is what the rest of the page is arranged around.
    """
    if verdict is None:
        tier = res.get("predicted_tier")
        return "good", f"Predicted {tier}", "", ""

    layout = VERDICT_LAYOUT.get(verdict.state, VERDICT_LAYOUT[NOT_ANSWERABLE])
    if verdict.shows_tier:
        headline = str(res.get("predicted_tier") or f"Tier {verdict.tier}")
        if verdict.state == AGREED_OUT_OF_DOMAIN:
            headline += ", outside the training range"
        lede = TIER_SENTENCE.get(verdict.tier, "")
    elif verdict.state == CONFLICTED:
        headline = "The evidence disagrees"
        lede = "No tier is shown, because the agents point in different directions on this construct."
    else:
        headline = "Not answerable"
        lede = "No agent that has passed its gate could answer for this construct."
    return layout["state"], headline, lede, layout["action"]


def render_verdict(res: dict, prov) -> None:
    """
    Depth one: one statement, one sentence, one action, and the risk beside it.

    The risk figure and the tier now come from the same place when the agent
    path produced a verdict: the critic derives risk from the probability the
    agents put on the low expression band, so a tier and a risk can no longer
    contradict each other. The old biophysical penalty sum survives as the
    fallback and as agent input features, which is where it belongs.
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

    state, headline, lede, action = verdict_headline(res, verdict)
    ui.band(state, headline, lede)
    if action:
        ui.note(action)

    confidence = res.get("confidence")
    figures = [{
        "value": ui.percent(risk, 1),
        "label": "Construct failure risk",
        "note": risk_note,
    }]
    if confidence is not None and (verdict is None or verdict.shows_tier):
        figures.append({
            "value": ui.percent(confidence, 0, already_scaled=False),
            "label": "Confidence in the tier",
            "note": "The model's own probability for the tier above. With six tiers, chance alone "
                    "is 17%.",
        })
    if verdict is not None:
        figures.append({
            "value": f"{verdict.agreement.n_applying} of 4",
            "label": "Agents that applied",
            "note": "An agent answers only when it holds an artifact whose gate is a pass and the "
                    "input is inside its competence.",
        })
    ui.figures(figures)

    if verdict is not None and verdict.conflicts:
        for conflict in verdict.conflicts:
            st.error(conflict, icon="⛔")

    render_unservable_notice(res)
    render_out_of_range_notice(res)


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


def top_drivers(res: dict, verdict, limit: int = 3) -> list:
    """
    The features that actually moved this answer, largest absolute first.

    Chosen by attribution rather than fixed, which is the point: the old
    interface showed the same five features in the same order for every
    construct, under a heading that called three of the strongest ones
    additional.
    """
    attribution = {}
    if verdict is not None:
        opinion = verdict.opinions.get("expression")
        if opinion is not None and opinion.attribution:
            attribution = opinion.attribution
    if not attribution:
        attribution = res.get("feature_importances") or {}
    ranked = sorted(attribution.items(), key=lambda kv: -abs(kv[1]))[:limit]
    return [name for name, _ in ranked]


# Model feature name to the display label and the extractor key behind it.
FEATURE_DISPLAY = {
    "GC_percent": ("GC content", "gc_content", "%", 1),
    "GC3": ("GC3, wobble position", "gc3_content", "%", 1),
    "CAI": ("Codon adaptation index", "codon_adaptation_index", "", 3),
    "rare_codon_fraction": ("Rare codon fraction", "rare_codon_fraction", "", 3),
    "rare_codons_head30": ("Rare codons, first 30", "rare_codons_head30", "", 3),
    "head_folding_48": ("5' folding, 48 nt", "head_folding_48", " kcal/mol", 2),
    "genewide_folding_avg": ("Folding energy, gene wide", "genewide_folding_avg", " kcal/mol", 2),
    "gravy_score": ("GRAVY, hydrophobicity", "gravy_score", "", 3),
    "pI": ("Isoelectric point", "pI", "", 2),
    "length_codons": ("Length", "length_codons", " codons", 0),
}

FEATURE_NOTES = {
    "GC3": "The strongest single predictor of yield on Welch's synonymous variants.",
    "rare_codon_fraction": "A moderate share goes with higher yield, not lower.",
    "head_folding_48": "Real ViennaRNA energy over the construct's own head.",
    "genewide_folding_avg": "Real ViennaRNA energy, averaged over sliding windows.",
    "gravy_score": "Above 0.4 is the validated flag for the lowest expression band.",
}


def feature_figure(res: dict, model_name: str, with_note: bool = True) -> dict:
    label, key, suffix, places = FEATURE_DISPLAY.get(
        model_name, (model_name, model_name, "", 3)
    )
    value = res["features"].get(key)
    return {
        "value": ui.number(value, places, suffix),
        "label": label,
        "note": FEATURE_NOTES.get(model_name, "") if with_note else "",
    }


def render_drivers(res: dict, verdict) -> None:
    """Depth two: only the features that moved this particular answer."""
    drivers = top_drivers(res, verdict)
    if not drivers:
        return
    ui.section(
        "What moved this answer",
        "Chosen by attribution for this construct, not a fixed list. Signed SHAP values for the "
        "predicted tier are in the evidence below.",
    )
    ui.figures([feature_figure(res, name) for name in drivers])


def render_features(res: dict) -> None:
    """
    Depth three: every computed feature, in one row rather than two.

    The old layout split these across Computed features and Additional
    measures, a heading that told the reader the second row mattered less. GC3
    was in the second row and is the strongest single predictor on Welch, so
    the split was not only unhelpful, it was backwards.

    The GC derived folding proxy is gone from the display. reports/gates.md now
    records it as a monotonic function of 5' GC at a correlation of exactly
    minus one, so showing it beside the real ViennaRNA energy invited a reader
    to treat one quantity under two names as two measurements.
    """
    names = [n for n in FEATURE_DISPLAY if res["features"].get(FEATURE_DISPLAY[n][1]) is not None]
    for start_index in range(0, len(names), 5):
        ui.figures([feature_figure(res, n, with_note=False) for n in names[start_index:start_index + 5]])

    f = res["features"]
    if _is_blank(f.get("genewide_folding_avg")):
        ui.note(
            "Folding energy reads N/A because ViennaRNA is not installed. Install it with "
            "`pip install ViennaRNA` to compute real folding energies."
        )
    if _is_blank(f.get("gravy_score")) or _is_blank(f.get("pI")):
        ui.note(
            "GRAVY and pI read N/A because the sequence could not be translated to a standard "
            "amino acid string, so the diagnostics that depend on them are omitted."
        )


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


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
    """
    What changed, drawn before it is tabulated.

    The engine already guarantees the protein is unchanged and checks it on
    every candidate. Two aligned tracks make that guarantee visible: at every
    outlined position the amino acid in both tooltips is the same. The old
    layout printed the two sequences as separate text boxes and left the user
    to diff them by eye.
    """
    conserved = sorted(result.conserved.locked) if result.conserved.locked else None

    ui.section(
        "What changed",
        f"{result.codons_changed} of {result.codons_mutable} changeable codons moved. Original "
        "above, redesigned below, with every change outlined.",
    )
    channel = sequence_map.channel_picker("redesign_channel")
    sequence_map.render_diff(result.original.sequence, result.best.sequence, channel, conserved)
    sequence_map.render_legend(result.original.sequence, conserved)

    runs = sequence_map.changed_runs(result.original.sequence, result.best.sequence)
    if not runs.empty:
        st.dataframe(runs, hide_index=True, use_container_width=True)
        ui.note(
            "The amino acid column is the proof the protein did not move: every run encodes the "
            "same residues before and after."
        )

    ui.section("The evidence")
    with st.expander("Feature movement", expanded=False):
        st.caption(
            "Only the features a codon swap can move. GRAVY and the isoelectric point are "
            "identical by construction and are left out."
        )
        st.dataframe(result.feature_comparison(), hide_index=True, use_container_width=True)

    with st.expander("Search trace", expanded=False):
        st.caption(
            "The best objective value after each round. A flat run of rounds is the search failing "
            "to improve, at which point it halves the number of swaps per candidate and tries a "
            "finer step."
        )
        st.line_chart(pd.DataFrame({"Best objective": result.history}))

    with st.expander("How this objective has behaved on measured data", expanded=False):
        render_calibration_notice()

    with st.expander("The redesigned sequence", expanded=False):
        st.text_area(
            "Optimised coding DNA, 5' to 3'", value=result.best.sequence, height=160, disabled=True,
        )

    st.download_button(
        "Download the redesigned sequence (FASTA)",
        data=(
            f">synthbio_redesign codons_changed={result.codons_changed}\n"
            f"{result.best.sequence}\n"
        ),
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

    ui.inject_css()

    model, explainer, training_ranges = load_trained_artifacts()
    prov = prov_mod.describe(model, MODEL_PATH if model is not None else None)

    # Sidebar
    st.sidebar.header("Settings")
    st.sidebar.caption(
        "Host: Escherichia coli. The codon table, the training data and every gate in this "
        "project are E. coli, so there is nothing here to choose."
    )
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
    ui.note(
        "Four narrow models read a coding sequence, each answers one question, and a critic says "
        "what they agree on. Any of them may decline, and declining is an answer."
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
        if st.session_state.get("sequence_box") is None:
            st.session_state["sequence_box"] = ""
        sequence_input = st.text_area(
            "Coding DNA sequence, 5' to 3'",
            key="sequence_box",
            height=130,
            placeholder="Paste a coding sequence starting at ATG",
            help="Must start at the start codon.",
        )
        if not sequence_input.strip():
            # An empty state that says what the tool does and offers a construct
            # to try, rather than a pre filled sample that quietly becomes the
            # thing being scored.
            ui.note(
                "Nothing to score yet. Paste a coding sequence, fetch one by accession, or load "
                "the example below to see what the tool produces."
            )
            st.button(
                "Load an example construct",
                on_click=lambda: st.session_state.update(sequence_box=SAMPLE_SEQUENCE),
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

    # An untouched box is not a mistake, so an empty input says nothing. The
    # validation messages appear the moment there is something to validate, and
    # they appear above the button so they are read before the person acts.
    if issues and sequence_input.strip():
        render_validation_issues(issues)

    run_btn = st.button(
        "Analyse construct", type="primary", disabled=not is_scorable(issues) or not normalised
    )

    # Gate the run on an explicit click. The sequence box ships with a sample,
    # so without this the app would predict on page load and the button would
    # have no effect.
    if run_btn:
        st.session_state["analysis_requested"] = True
        st.session_state["analysed_sequence"] = normalised

    if not st.session_state.get("analysis_requested"):
        if normalised and is_scorable(issues):
            ui.note("Ready. Select Analyse construct to score it.")
        return

    # A sequence edited after the last run should not silently show stale results.
    if st.session_state.get("analysed_sequence") != normalised:
        st.info("The sequence has changed since the last analysis. Select Analyse construct to rescore.", icon="ℹ️")
        return

    if not is_scorable(issues):
        st.caption("Scoring is withheld until the problems above are resolved, so that no figure is shown "
                   "that the input cannot support.")
        return

    logger.info(f"Running prediction for {len(normalised)} nt sequence...")
    try:
        with st.spinner("Computing features, folding the transcript and asking the agents..."):
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
        render_diagnosis(res, prov, model, normalised)
        render_agent_panel(res.get("critic_verdict"))

    with redesign_tab:
        render_redesign(normalised, model, res)


def render_diagnosis(res: dict, prov, model, sequence: str) -> None:
    """
    Three depths: the verdict, why, and the evidence.

    Depth one is the verdict band and one action. Depth two is the sequence
    map, the features that moved this answer, and the agent strip. Depth three
    is everything auditable, behind expanders, because a reader who wants it is
    in the minority and should not pay for it in scroll.

    Nothing was deleted on the way. Every caveat that used to sit between
    sections now sits under the figure it qualifies, which is where it gets
    read rather than skipped.
    """
    verdict = res.get("critic_verdict")

    render_verdict(res, prov)

    ui.section(
        "Your construct",
        "One cell per codon. Hover for the codon, its amino acid and the value behind the colour.",
    )
    channel = sequence_map.channel_picker("diagnosis_channel")
    sequence_map.render_track(sequence, channel, key="diagnosis_track")
    sequence_map.render_legend(sequence)

    render_drivers(res, verdict)

    if verdict is not None:
        ui.section(
            "The agents",
            "Four narrow models, each with one question and the right to decline.",
        )
        agents_view.render_strip(verdict)
        ui.note(verdict.explanation)
        agents_view.render_how_it_works(verdict)

    ui.section("The evidence", "Everything the verdict above rests on.")

    with st.expander("All computed features", expanded=False):
        render_features(res)

    with st.expander("Model attribution for this construct", expanded=False):
        st.caption(
            "Signed SHAP values for the predicted tier: positive pushed the model towards the "
            "tier it chose, negative pushed away."
        )
        feat_df = res.get("feat_df")
        if feat_df is None:
            feat_df = build_feature_row(res["features"])
        render_shap_waterfall(model, feat_df, predicted_class=res.get("predicted_score"))

    if verdict is not None:
        with st.expander("Each agent in detail, with its gate", expanded=False):
            agents_view.render_detail(verdict)

    with st.expander("Biological heuristics, which are rules rather than models", expanded=False):
        render_diagnostics(res)

    with st.expander("Where these figures came from", expanded=False):
        st.caption(prov.as_line())
        for line in prov.caveats():
            st.caption(line)
        if verdict is not None:
            st.caption(
                f"Agreement: {verdict.agreement.as_line()}. Conflict threshold "
                + ("fitted on the reserved critic pool." if verdict.thresholds_fitted
                   else "declared, not fitted, because no critic has been trained.")
            )


if __name__ == "__main__":
    main()
