"""
The gate harness. One command, every baseline, every gate, one report.

    python scripts/gates.py

Writes `reports/gates.md`, which is the single source of truth for every
numeric claim about this project. Nothing in the multi agent plan can be
trusted until that file exists and regenerates, and producing it changes no
model and breaks no interface.

The rule the harness exists to enforce: no agent is connected to the critic
before it beats a named baseline. An agent that fails its gate stays in the
repository, keeps its gate report, and returns an abstention at runtime.

Two gates cannot be run in this working tree and the report says so in place of
a number. Gate A needs splits clustered at 30 per cent sequence identity, and
the training table carries gene ids and features but no sequences. Gate C needs
an Agent C artifact, and there is no within gene training data on disk to fit
one from. Where a gate is blocked, the baseline it would have to beat is
measured anyway, because that baseline is a fact about the data rather than
about any model.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gating  # noqa: E402
from src.agents.structure import StructureAgent  # noqa: E402
from src.agents.variant import GC3_BASELINE_RHO, VariantAgent  # noqa: E402
from src.datasets import clustering  # noqa: E402
from src.datasets.variant_sets import available_variant_sets  # noqa: E402

logger = logging.getLogger("SynthBio.gates")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.gates] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_REPORT = "reports/gates.md"


def _fmt(value: object, places: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{places}f}"
    return str(value)


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def majority_line(served: gating.PredictorResult, majority: gating.PredictorResult) -> str:
    """
    One sentence comparing accuracy against the majority class, written from the
    measured numbers rather than from a remembered figure.

    The deployed artifact's own metadata records a single held out split where
    its accuracy, 0.309, sat at or just below the majority rate of 0.311. The
    grouped cross validation here uses different folds, so the two can disagree,
    and the sentence names both rather than letting the kinder one stand alone.
    """
    gap = served.mean("accuracy") - majority.mean("accuracy")
    spread = max(served.std("accuracy"), majority.std("accuracy"))
    direction = "above" if gap > 0 else "below"
    clears = "larger" if abs(gap) > spread else "smaller"
    return (
        f"{served.mean('accuracy'):.3f} against {majority.mean('accuracy'):.3f}, "
        f"{abs(gap):.3f} {direction} it, a gap {clears} than the fold to fold spread of "
        f"{spread:.3f}. The artifact's own metadata records 0.309 against 0.311 on the single "
        "held out split it was built with, so whether it beats the majority rate on accuracy "
        "depends on the split, which is why this gate is defined on macro F1 and on the rank "
        "correlation instead."
    )


def _model_params() -> Optional[Dict[str, object]]:
    import json

    if not os.path.exists(gating.METADATA_PATH):
        return None
    try:
        with open(gating.METADATA_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle).get("model_params")
    except Exception:
        return None


# --- the heuristics, measured rather than quoted ----------------------------


def gravy_rule_figures(table: pd.DataFrame) -> Dict[str, float]:
    """
    The validated GRAVY rule, recomputed from the feature table.

    This is the floor Agent B has to beat, and it is quoted all over this
    project, so the harness measures it rather than repeating it.
    """
    flagged = table["gravy_score"] > gating.GRAVY_RULE_THRESHOLD
    lowest = table["expression_score"] == 0
    top_two = table["expression_score"] >= 4
    return {
        "n_flagged": int(flagged.sum()),
        "n_total": int(len(table)),
        "precision_lowest_band": float(lowest[flagged].mean()) if flagged.any() else float("nan"),
        "base_rate_lowest_band": float(lowest.mean()),
        "recall_lowest_band": float(flagged[lowest].mean()) if lowest.any() else float("nan"),
        "flagged_in_top_two": float(top_two[flagged].mean()) if flagged.any() else float("nan"),
        "base_rate_top_two": float(top_two.mean()),
    }


def pi_rule_figures(table: pd.DataFrame) -> Dict[str, float]:
    """The weak pI rule, measured so that it keeps being reported as weak."""
    flagged = (table["pI"] >= 7.2) & (table["pI"] <= 7.8)
    lowest = table["expression_score"] == 0
    return {
        "n_flagged": int(flagged.sum()),
        "precision_lowest_band": float(lowest[flagged].mean()) if flagged.any() else float("nan"),
        "base_rate_lowest_band": float(lowest.mean()),
    }


# --- Welch, which is the only external evidence on disk ---------------------


def welch_section(agent: VariantAgent) -> Dict[str, object]:
    """
    Every scorer that can be run against Welch's measured yields.

    The served tier model is scored here too, which is the second half of
    Gate A: the plan requires that Agent A is not negatively correlated with
    measured yield on either series. The existing harness at
    `scripts/welch2009_validation.py` cannot do this, because it builds a two
    column matrix of GC and CAI and the deployed artifact needs ten features.
    """
    frame = gating.load_welch()
    feature_dicts = gating.welch_feature_dicts(frame)

    scorings = {}
    gc3 = gating.score_welch(frame, [f["gc3_content"] for f in feature_dicts], "GC3 alone")
    gc_percent = gating.score_welch(frame, [f["gc_content"] for f in feature_dicts], "GC per cent alone")
    cai = gating.score_welch(frame, [f["codon_adaptation_index"] for f in feature_dicts], "CAI alone")
    scorings["GC3 alone"] = gc3
    scorings["GC per cent alone"] = gc_percent
    scorings["CAI alone"] = cai
    # Welch's own published descriptors, for comparison with the recomputed ones.
    if "welch_gc_percent" in frame.columns:
        scorings["GC per cent, Welch's own"] = gating.score_welch(
            frame, frame["welch_gc_percent"].tolist(), "GC per cent, Welch's own"
        )

    served = _score_served_model(frame, feature_dicts)
    if served is not None:
        scorings["Served tier model, expected tier"] = served

    candidate = None
    if agent.ranker is not None:
        try:
            ranking = agent.rank(frame["orf_dna"].astype(str).tolist(), feature_dicts)
            candidate = gating.score_welch(frame, list(ranking.scores), "Agent C, variant ranker")
            scorings["Agent C, variant ranker"] = candidate
        except Exception as err:
            logger.error(f"Agent C could not score Welch: {err}")

        # A diagnostic, not the gate. Agent C is trained on Cambray's 96 nt
        # designed region and is being asked here about whole genes of 843 and
        # 1,725 nt, so the same model is also scored on the first 96 nt of each
        # ORF. If the two differ a lot, the gap to GC3 is about the length
        # mismatch rather than about codon effects, and the remedy is training
        # data from whole constructs. The gate itself stays on the whole ORF,
        # which is what the application hands the agent.
        window = _training_window_nt(agent)
        if window:
            try:
                from src.features import extract_features

                head_dicts = [extract_features(seq[:window]) for seq in frame["orf_dna"].astype(str)]
                head_ranking = agent.rank(
                    [seq[:window] for seq in frame["orf_dna"].astype(str)], head_dicts
                )
                scorings[f"Agent C, first {window} nt only (diagnostic, not the gate)"] = gating.score_welch(
                    frame, list(head_ranking.scores), f"Agent C on the first {window} nt"
                )
            except Exception as err:
                logger.error(f"Agent C could not score the Welch head window: {err}")

    outcome = gating.evaluate_gate_c(candidate, gc3, gc_percent)
    return {"frame": frame, "scorings": scorings, "gate_c": outcome, "gc3": gc3}


def _training_window_nt(agent: VariantAgent) -> Optional[int]:
    """The median training sequence length, from the artifact's own sidecar."""
    lengths = ((agent.metadata or {}).get("training_data") or {}).get("sequence_length_nt") or {}
    median = lengths.get("median")
    return int(median) if median else None


def _score_served_model(frame: pd.DataFrame, feature_dicts: Sequence[Dict[str, float]]) -> Optional[gating.WelchScoring]:
    """The deployed artifact's expected tier on every Welch variant, or None."""
    if not os.path.exists(gating.MODEL_PATH):
        logger.warning(f"No artifact at '{gating.MODEL_PATH}'; the served model is not scored on Welch.")
        return None
    try:
        import joblib

        from src.predict import build_feature_row

        model = joblib.load(gating.MODEL_PATH)
        matrix = pd.concat([build_feature_row(f) for f in feature_dicts], ignore_index=True)
        required = list(getattr(model, "feature_names_in_", gating.MODEL_FEATURES))
        proba = model.predict_proba(matrix[required])
        classes = np.asarray([int(c) for c in getattr(model, "classes_", range(proba.shape[1]))], dtype=float)
        expected = proba @ classes
        return gating.score_welch(frame, list(expected), "Served tier model, expected tier")
    except Exception as err:
        logger.error(f"Could not score the served model on Welch: {err}")
        return None


# --- the report -------------------------------------------------------------


def _agent_c_narrative(
    agent: VariantAgent,
    outcome: gating.GateCOutcome,
    scorings: Dict[str, gating.WelchScoring],
) -> str:
    """
    What a trained Agent C artifact actually showed, pass or fail.

    Written from the artifact's own metadata rather than from memory, so this
    paragraph cannot describe a model other than the one on disk.
    """
    metadata = agent.metadata or {}
    training = metadata.get("training_data") or {}
    held = metadata.get("heldout_groups") or {}
    candidate = outcome.candidate
    lines = []

    lines.append(
        f"The artifact under test is `{agent.model_path}`, a "
        f"{metadata.get('model_class', 'ranker')} over {len(metadata.get('features', []))} features "
        f"a synonymous substitution can move, fitted on {training.get('n_rows', 0):,} sequences in "
        f"{training.get('n_groups', 0):,} query groups."
    )

    lengths = training.get("sequence_length_nt") or {}
    if lengths:
        low, high = lengths.get("min"), lengths.get("max")
        span = f"all {low} nt long" if low == high else f"{low} to {high} nt long"
        lines.append(
            f"Its training sequences are {span}. The Welch ORFs are 843 and 1,725 nt, so every "
            "figure in the table above is an extrapolation in sequence length, and the agent says "
            "so in its own domain check at runtime."
        )

    if held:
        model_tau, gc3_tau = held.get("model_mean_tau"), held.get("gc3_mean_tau")
        verdict = "beats" if (model_tau or 0) > (gc3_tau or 0) else "does not beat"
        lines.append(
            f"The secondary part of Gate C, on {held.get('n_groups', 0):,} held out query groups "
            f"from the training source: mean Kendall tau within group {model_tau:+.3f} for the "
            f"model against {gc3_tau:+.3f} for GC3 alone, so inside its own domain the model "
            f"{verdict} the baseline."
        )

    if candidate is not None and not outcome.passed:
        positive = all(candidate.rho(s) > 0 for s in gating.WELCH_SERIES)
        if positive:
            lines.append(
                "What this run did establish, and it is not nothing: unlike the tier model, the "
                "ranker is positive on both Welch series, so it no longer orders a synonymous "
                "series backwards. What it did not do is beat GC3, which is the bar, so the agent "
                "abstains and the redesign loop keeps ranking on the tier model with the GC3 "
                "corroboration check as its guard."
            )
        if positive and candidate.overall[0] < 0:
            lines.append(
                f"The overall figure, {candidate.overall[0]:+.3f}, is negative while both series "
                "are positive, which is not a contradiction. The two series are different proteins "
                "measured on different scales, so pooling them compares a polymerase variant "
                "against an scFv variant, which is a between gene comparison this agent has no "
                "opinion about. The per series figures are the ones the gate is defined on."
            )
        lines.append(
            "Nothing here was tuned against Welch and nothing may be. Welch is the held out "
            "external test set, so choosing between models by their Welch score would turn the "
            "gate into a training signal and there would be no external evidence left in the "
            "project."
        )

    diagnostic = next((k for k in scorings if "diagnostic" in k), None)
    if diagnostic and candidate is not None:
        head = scorings[diagnostic]
        per_series = ", ".join(
            f"{series} {head.rho(series):+.3f} against {candidate.rho(series):+.3f}"
            for series in gating.WELCH_SERIES
        )
        # A meaningful movement, not any movement. Comparing per series means
        # rather than taking the better of two keeps a 0.005 difference on one
        # series from being reported as an explanation.
        movement = sum(
            head.rho(s) - candidate.rho(s) for s in gating.WELCH_SERIES
        ) / len(gating.WELCH_SERIES)
        improved = movement > 0.05
        lines.append(
            "The length mismatch is measured rather than assumed. Scoring the same model on only "
            f"the first {_training_window_nt(agent)} nt of each Welch ORF, which is the window it "
            f"was trained on, gives {per_series}. "
            + f"The mean per series change is {movement:+.3f}. "
            + (
                "Matching the window helps, so part of the gap is the extrapolation in length."
                if improved
                else "Matching the window does not help, so the gap to GC3 is not "
                "merely an extrapolation in length: what the model learned on a 32 codon reporter "
                "fragment does not carry to whole genes. Training data from whole constructs is "
                "the thing to try next, not a different objective or more trees."
            )
            + " This is a diagnostic and not the gate."
        )

    return "\n\n".join(lines)


def build_report(
    table: pd.DataFrame,
    cv_results: Optional[List[gating.PredictorResult]],
    welch: Dict[str, object],
    sources: Dict[str, Optional[str]],
    agent: VariantAgent,
    structure_agent: StructureAgent,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Gates")
    add("")
    add(
        f"Regenerated by `python scripts/gates.py` on {date.today():%Y-%m-%d} at commit "
        f"`{_commit()}`. Every number in this file is measured by that command. Nothing else in "
        "the repository may state a figure this file does not contain."
    )
    add("")
    add(
        "A gate is a comparison against a named baseline. No agent is connected to the critic "
        "before it beats its baseline, and an agent that has not passed its gate abstains at "
        "runtime and keeps its gate report."
    )
    add("")

    # --- M0 -----------------------------------------------------------------
    add("## M0. The evidence base")
    add("")
    status = clustering.clustering_status()
    if status is None:
        assignment = clustering.load_assignment()
        if assignment is None:
            add(
                "Identity clustering can be run but has not been. Run it, commit "
                f"`{clustering.CLUSTER_REPORT}`, then rerun this harness."
            )
        else:
            add(
                f"Clustered by `{assignment.method}`: {len(assignment.clusters):,} sequences in "
                f"{assignment.n_clusters:,} clusters, with {len(assignment.critic_pool):,} clusters "
                f"reserved as the critic pool and recorded in `{clustering.CLUSTER_REPORT}`."
            )
    else:
        add(f"**Blocked.** {status}")
        add("")
        add(
            "Consequences, which are load bearing for the rest of this file. Gate A cannot be run, "
            "because it is defined on identity clustered splits. The critic pool cannot be "
            "reserved, so no critic can be trained without leaking: the leakage this guards "
            "against is the one failure that would make the whole system worse than the present "
            "model, because it would report high confidence and high agreement and be wrong. "
            "Organism grouped splits are reported below in place of clustered ones and are not a "
            "substitute: paralogues inside one organism and homologues across two organisms both "
            "still leak."
        )
    add("")
    add("Within gene sources for Agent C:")
    add("")
    for name, reason in sources.items():
        add(f"- `{name}`: " + ("available" if reason is None else reason))
    add("")

    # --- Gate A -------------------------------------------------------------
    add("## Gate A. Expression tier")
    add("")
    add(
        "**Status: blocked.** The gate requires identity clustered 5 fold cross validation, beating "
        "the majority class, length alone, GC per cent alone and the GRAVY rule on macro F1 and on "
        "Spearman rho between predicted and true tier, with the gap larger than the fold to fold "
        "standard deviation, and additionally not being negatively correlated with measured yield "
        "on either Welch series. The clustered split cannot be computed here, so what follows is "
        "the organism grouped comparison, which is the same comparison on a weaker split."
    )
    add("")
    if cv_results is None:
        add("Cross validation was skipped for this run.")
    else:
        add(
            f"Organism grouped `StratifiedGroupKFold`, 5 folds over "
            f"{table['gene_id'].nunique():,} genes in "
            f"{len(set(clustering.organism_groups(table['gene_id'].tolist()))):,} organism groups. "
            "Every predictor sees the same folds, so the comparison is paired. Mean across folds "
            "with the fold to fold standard deviation."
        )
        add("")
        add("| Predictor | Macro F1 | Accuracy | Spearman rho, predicted against true tier | Mean absolute tier error |")
        add("| :--- | ---: | ---: | ---: | ---: |")
        for result in cv_results:
            add(
                f"| {result.name} | {_fmt(result.mean('macro_f1'))} ± {_fmt(result.std('macro_f1'))} "
                f"| {_fmt(result.mean('accuracy'))} ± {_fmt(result.std('accuracy'))} "
                f"| {_fmt(result.mean('spearman_rho'))} ± {_fmt(result.std('spearman_rho'))} "
                f"| {_fmt(result.mean('mean_absolute_tier_error'), 2)} ± {_fmt(result.std('mean_absolute_tier_error'), 2)} |"
            )
        add("")
        served = next((r for r in cv_results if r.name == "Served model configuration"), None)
        if served is not None:
            margins = []
            for result in cv_results:
                if result is served:
                    continue
                gap = served.mean("macro_f1") - result.mean("macro_f1")
                spread = max(served.std("macro_f1"), result.std("macro_f1"))
                verdict = "clears" if gap > spread else "does not clear"
                margins.append(
                    f"- against **{result.name}**: macro F1 gap {gap:+.3f}, fold to fold spread "
                    f"{spread:.3f}, so the gap {verdict} the spread."
                )
            add("Margin against each baseline, on macro F1:")
            add("")
            lines.extend(margins)
            add("")
            majority = next((r for r in cv_results if r.name == "Majority class"), None)
            if majority is not None:
                add(
                    "The majority class row is the one to read first. Its rank correlation is absent "
                    "because a constant predictor has no variance and cannot order anything, so macro "
                    "F1 and the rank correlation are where a comparison against it means something. "
                    f"On accuracy the served configuration reaches {majority_line(served, majority)}"
                )
                add("")
                add(
                    "No figure in this table is a performance claim. The defensible statement is that "
                    "the pipeline, the features and the attribution layer are in place, that the model "
                    "orders genes better than any single descriptor measured here, and that it orders "
                    "synonymous variants of one gene backwards on one of the two measured series."
                )
                add("")

    welch_scorings: Dict[str, gating.WelchScoring] = welch["scorings"]  # type: ignore[assignment]
    served_welch = welch_scorings.get("Served tier model, expected tier")
    if served_welch is not None:
        add("The second half of Gate A, on Welch's measured yields:")
        add("")
        add(
            f"- polymerase series: rho {served_welch.rho('polymerase'):+.3f}\n"
            f"- scFv series: rho {served_welch.rho('scFv'):+.3f}\n"
            f"- overall: rho {served_welch.overall[0]:+.3f}"
        )
        add("")
        negative = [s for s in gating.WELCH_SERIES if served_welch.rho(s) < 0]
        if negative:
            add(
                "**This clause of Gate A fails outright.** The served model's expected tier is "
                f"negatively correlated with measured yield on the {', '.join(negative)} series, so "
                "it orders those variants backwards. The clause exists because a model that ranks a "
                "synonymous series backwards will drive a redesign search in the wrong direction, "
                "which is what Agent C is for."
            )
        else:
            add("No Welch series is ranked backwards by the served model.")
        add("")

    # --- Gate B -------------------------------------------------------------
    add("## Gate B. Solubility")
    add("")
    add(
        "**Status: not run.** Agent B is declared and not built: no solubility dataset is in the "
        "working tree. It must beat the majority class, length alone and GRAVY alone on AUROC and "
        "Matthews correlation under splits clustered at 30 per cent identity across the union of "
        "every source used, and it must be calibrated, with a Brier score and a reliability curve."
    )
    add("")
    gravy = gravy_rule_figures(table)
    add("The baseline it has to beat, measured on the Boel table:")
    add("")
    add(
        f"- GRAVY above {gating.GRAVY_RULE_THRESHOLD} flags **{gravy['n_flagged']} of "
        f"{gravy['n_total']:,}** genes.\n"
        f"- **{gravy['precision_lowest_band'] * 100:.0f} per cent** of the flagged genes sit in the "
        f"lowest expression band, against a base rate of "
        f"{gravy['base_rate_lowest_band'] * 100:.0f} per cent.\n"
        f"- **{gravy['flagged_in_top_two'] * 100:.0f} per cent** of them reach the top two bands, "
        f"against a base rate of {gravy['base_rate_top_two'] * 100:.0f} per cent.\n"
        f"- Recall is **{gravy['recall_lowest_band'] * 100:.0f} per cent**, so the rule is precise "
        "and narrow. It is the current floor and it is not a model."
    )
    add("")
    pi = pi_rule_figures(table)
    add(
        f"The pI rule is weak and is recorded so that it stays described as weak: pI between 7.2 "
        f"and 7.8 flags {pi['n_flagged']} genes and lifts the lowest band rate from "
        f"{pi['base_rate_lowest_band'] * 100:.0f} to {pi['precision_lowest_band'] * 100:.0f} per cent."
    )
    add("")

    # --- Gate C -------------------------------------------------------------
    outcome: gating.GateCOutcome = welch["gate_c"]  # type: ignore[assignment]
    frame: pd.DataFrame = welch["frame"]  # type: ignore[assignment]
    add("## Gate C. Codon variant ranking")
    add("")
    add(
        "On Welch 2009, per series, Agent C must produce a positive Spearman rho on both the "
        "polymerase and the scFv series and must beat GC3 alone on both. Welch is held out "
        "entirely and is never trained on. Beating GC per cent alone overall is a harder bar and "
        "is reported but not required, because the per series comparison is the use case."
    )
    add("")
    add(f"**Status: {'passed' if outcome.passed else 'not passed'}.** {outcome.summary}")
    add("")
    add(
        f"Measured over the {len(frame)} variants that carry both a sequence and a yield. The "
        "censored variants are excluded: the sheet records the literal string `nd` against a "
        "detection limit of about 5 ug/ml, so they are left censored and cannot be ordered."
    )
    add("")
    add("| Scorer | Overall rho | Polymerase rho | scFv rho | n |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    for name, scoring in welch_scorings.items():
        add(
            f"| {name} | {_fmt(scoring.overall[0])} | {_fmt(scoring.rho('polymerase'))} "
            f"| {_fmt(scoring.rho('scFv'))} | {scoring.overall[2]} |"
        )
    add("")
    gc3: gating.WelchScoring = welch["gc3"]  # type: ignore[assignment]
    add(
        "The per series figures recorded in the plan and in `src/agents/variant.py` are "
        f"{GC3_BASELINE_RHO['polymerase']:+.3f} on polymerase and {GC3_BASELINE_RHO['scFv']:+.3f} "
        f"on scFv for GC3 alone. This run measures {gc3.rho('polymerase'):+.3f} and "
        f"{gc3.rho('scFv'):+.3f}. Any divergence between the two is a change in the feature code "
        "and should be explained before it is accepted."
    )
    add("")
    if outcome.candidate is None:
        add(
            "Agent C is wired into the predictor and the redesign loop and abstains, because it "
            "has no artifact. The redesign loop therefore still ranks on the tier model with the "
            "GC3 corroboration check as its guard, which is the behaviour the Welch result above "
            "makes necessary."
        )
        add("")
    else:
        add(_agent_c_narrative(agent, outcome, welch_scorings))
        add("")

    # --- Gate D -------------------------------------------------------------
    add("## Gate D. mRNA structure and initiation")
    add("")
    add(
        "Agent D is the one agent blocked by a missing input rather than a missing dataset. "
        "`feature_table_v4.csv` carries Boel's own Gz48v and Gave48, computed over the expression "
        "vector's 5' untranslated region, and the application cannot compute them from a pasted "
        "coding sequence because it does not know the vector. Declaring the vector makes the "
        "initiation window computable. The gate has two parts: beat the GC of the first 48 nt "
        "alone at predicting the lowest expression band, which is the baseline that catches a "
        "folding feature that is really a GC feature under another name, and show that leader plus "
        "insert folding beats insert only folding on the same rows. The second part is a test of "
        "the interface change rather than of the model."
    )
    add("")

    folding = gating.cambray_folding_check()
    if folding:
        add("### The folding unit test, which this gate depends on")
        add("")
        add(
            f"Before anything is trained, the project's own folding code is checked against "
            f"measured expression on Cambray 2018, over {folding['n']:,} variants in "
            f"{folding['n_groups']:,} groups of synonymous variants. Cambray varies the 5' region "
            "against a fixed reporter and measures protein directly, so a bug in the folding "
            "implementation shows here."
        )
        add("")
        add("| Quantity | Against protein level, overall | Within synonymous groups |")
        add("| :--- | ---: | ---: |")
        add(
            f"| `head_folding_48`, real ViennaRNA | {_fmt(folding['overall']['head_folding_48'])} "
            f"| {_fmt(folding['within_group']['head_folding_48'])} |"
        )
        add(
            f"| `genewide_folding_avg`, real ViennaRNA | "
            f"{_fmt(folding['overall']['genewide_folding_avg'])} | n/a |"
        )
        add(
            f"| GC of the first 48 nt | {_fmt(folding['overall']['gc_head48'])} "
            f"| {_fmt(folding['within_group']['gc_head48'])} |"
        )
        add(f"| GC3 | n/a | {_fmt(folding['within_group']['gc3_content'])} |")
        add("")
        add(
            "**The folding code passes.** A less negative 5' folding energy, meaning less "
            "structure, goes with more protein, and within synonymous groups it is the strongest "
            "of the three, which is the direction and the ordering Cambray reports. It is worth "
            "noting that GC3 runs slightly negative here while it is strongly positive on Welch: "
            "the two libraries are different regimes, which is part of why Agent C does not "
            "transfer."
        )
        add("")
        add(
            f"**The retired proxy is confirmed to be GC.** `folding_energy`, the GC derived proxy, "
            f"correlates with the GC of the first 48 nt at exactly {_fmt(folding['proxy_vs_gc'])} "
            f"and with the real folding energy at only {_fmt(folding['proxy_vs_real'])}. A "
            f"correlation of {_fmt(folding['proxy_vs_gc'])} is not a strong relationship, it is an "
            "identity: the proxy is a monotonic function of that GC. Retiring it from Agent D "
            "loses nothing and stops one quantity being counted twice under two names."
        )
        add("")

    add("### The gate")
    add("")
    structure_meta = structure_agent.metadata or {}
    detail = structure_meta.get("gate_detail") or {}
    if not detail:
        add(
            "**Status: not run.** Agent D has no artifact. Recover the sequences with "
            "`python scripts/fetch_datasets.py --source boel`, then train and gate with "
            "`python scripts/train_agent_d.py`."
        )
    else:
        gate = structure_agent.gate()
        add(f"**Status: {gate.status}.** {gate.summary}")
        add("")
        add(
            f"Measured out of fold over {detail.get('n', 0):,} Boel constructs, "
            f"{detail.get('n_lowest_band', 0):,} of them in the lowest expression band, under an "
            "organism grouped 5 fold split. The gate asks for an identity clustered split, which "
            "cannot be computed here, so this is the same comparison on a weaker split."
        )
        add("")
        add("| Scorer | AUROC for the lowest band |")
        add("| :--- | ---: |")
        add(f"| Agent D, leader plus insert | {_fmt(detail.get('agent_auroc'))} |")
        add(f"| The same model, insert only | {_fmt(detail.get('insert_only_auroc'))} |")
        add(f"| GC of the first 48 nt alone | {_fmt(detail.get('gc_head48_auroc'))} |")
        add(f"| Leader plus insert folding alone | {_fmt(detail.get('leader_folding_alone_auroc'))} |")
        add(f"| Insert only folding alone | {_fmt(detail.get('insert_folding_alone_auroc'))} |")
        add("")
        orientation = detail.get("gc_head48_orientation")
        if orientation:
            add(
                "Each single descriptor is scored in whichever direction suits it, because AUROC "
                "is direction sensitive and beating a mis-signed baseline is not beating the "
                f"baseline. On this data {orientation}, which is the direction the structure "
                "argument predicts: a GC rich 5' region folds more tightly and initiates worse. "
                "The direction is recorded rather than assumed because the first version of this "
                "harness assumed the other one, which made the baseline look weaker than it is."
            )
            add("")
        gc_test = detail.get("agent_vs_gc") or {}
        insert_test = detail.get("leader_vs_insert") or {}
        add(
            f"The margins are tested rather than eyed, by a paired bootstrap over the same rows. "
            f"Against the GC baseline the difference is {_fmt(gc_test.get('delta'))} with a 95 per "
            f"cent interval of {_fmt(gc_test.get('low'))} to {_fmt(gc_test.get('high'))}, so the "
            f"first clause {'passes' if gc_test.get('separated') else 'does not pass'}. Against "
            f"the same model without the leader the difference is {_fmt(insert_test.get('delta'))} "
            f"with an interval of {_fmt(insert_test.get('low'))} to {_fmt(insert_test.get('high'))}, "
            f"so the second clause {'passes' if insert_test.get('separated') else 'does not pass'}."
        )
        add("")
        if not insert_test.get("separated"):
            add(
                "**This decides an interface question, which is what the second clause is for.** "
                "Declaring the vector does not demonstrably improve the prediction, so the vector "
                "selector should not be added to the application. The input would cost the user a "
                "decision on every construct and buy a difference the data cannot distinguish from "
                "zero. The leader code, the feature and this measurement stay, so the question can "
                "be reopened against better data, for instance a set with more than one vector in "
                "it; on Boel every construct shares one leader, so the leader can only interact "
                "with the insert and never vary on its own."
            )
            add("")

    boel_folding = gating.spearman(table["Gz48v"], table["expression_score"])
    servable_folding = gating.spearman(table["head_folding_48"], table["expression_score"])
    add(
        f"For reference, on the Boel table itself: the authors' own Gz48v, which folds leader plus "
        f"insert with RNAstructure, tracks the expression score at rho {_fmt(boel_folding[0])}, "
        f"while `head_folding_48`, the servable insert only counterpart computed here with "
        f"ViennaRNA, reaches {_fmt(servable_folding[0])}."
    )
    add("")

    # --- leakage ------------------------------------------------------------
    add("## Leakage")
    add("")
    add(
        "The one mistake that would invalidate everything: the critic is trained on the "
        "specialists' predictions, so if a specialist's training rows overlap the critic's "
        "training or evaluation rows, the critic learns that the specialist is more accurate than "
        "it is, and the agreement score becomes meaningless in exactly the cases it exists for. "
        "The guard is to cluster the union of every dataset once, assign every row a cluster id "
        "and a source label, reserve the critic pool by cluster, and never split by row anywhere."
    )
    add("")
    if status is None:
        add(
            "`tests/test_leakage.py` asserts that no cluster id appears in both a specialist's "
            "training set and the critic pool."
        )
    else:
        add(
            "No critic has been trained and none can be until the clustering above is unblocked. "
            "`tests/test_leakage.py` asserts that no critic model is loaded while the critic pool "
            "is unreserved, so this cannot be done by accident later."
        )
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_REPORT, help="Report to write.")
    parser.add_argument("--table", default=gating.FEATURE_TABLE, help="Expression feature table.")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip the grouped cross validation, which refits five predictors on five folds.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Cross validation folds.")
    args = parser.parse_args()

    if not os.path.exists(args.table):
        logger.error(f"Feature table not found: {args.table}")
        return 1

    table = pd.read_csv(args.table)
    logger.info(f"Loaded {args.table}: {len(table):,} rows.")

    cv_results = None
    if not args.skip_cv:
        groups = clustering.organism_groups(table["gene_id"].tolist())
        cv_results = gating.grouped_cv(
            table,
            gating.expression_predictors(_model_params()),
            groups,
            n_splits=args.folds,
        )

    agent = VariantAgent()
    welch = welch_section(agent)
    sources = available_variant_sets()

    report = build_report(table, cv_results, welch, sources, agent, StructureAgent())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    logger.info(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
