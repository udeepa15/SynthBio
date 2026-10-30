"""
Orchestration. Four function calls, a dict of opinions and a critic.

The design was described as a graph, and a graph framework would give real
things: declared conditional paths, per node retry and caching around the two
nodes that can be slow or can fail, which are the NCBI fetch and ViennaRNA
folding, a trace of what ran, and state that persists across a redesign run.

It would give no accuracy. The numbers come from the specialists and their data.
So the agents sit behind the single `AgentOpinion` interface from the start and
are orchestrated here with plain Python functions. The whole runtime behaviour
of the design fits in this file with no new dependency, no version pinning
against Streamlit and no new way for the deployed application to fail to start,
which matters because it runs on free hosting alongside xgboost, shap,
biopython and ViennaRNA already.

Adopting a framework later is a local change: the interface is already right,
and only this file moves.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.agents.base import (
    AGENT_ORDER,
    EXPRESSION,
    VARIANT,
    AgentOpinion,
    AgentRequest,
)
from src.agents.critic import CriticVerdict, combine
from src.agents.expression import ExpressionAgent
from src.agents.solubility import SolubilityAgent
from src.agents.structure import StructureAgent
from src.agents.variant import GC3Baseline, VariantAgent

logger = logging.getLogger("SynthBio.agents.graph")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.graph] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_agents(
    model: Any = None,
    explainer: Any = None,
    feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    variant_agent: Optional[VariantAgent] = None,
) -> Dict[str, Any]:
    """
    The four agents, keyed by name.

    The expression model and its explainer are passed in so that the
    application's single cached load is reused. The variant ranker loads its own
    artifact, which is a few kilobytes and is absent in the usual case.
    """
    return {
        EXPRESSION: ExpressionAgent(model=model, explainer=explainer, feature_ranges=feature_ranges),
        "solubility": SolubilityAgent(),
        VARIANT: variant_agent if variant_agent is not None else VariantAgent(),
        "structure": StructureAgent(),
    }


def run_agents(request: AgentRequest, agents: Dict[str, Any]) -> List[AgentOpinion]:
    """
    Every agent's opinion on one request, in a fixed order.

    Features are computed once, on the request, and the agents share them.
    Agents A, B and D run on every submitted sequence; Agent C runs too and
    abstains unless the request carries a comparison set, which is the
    conditional edge of the graph expressed as a precondition inside the agent
    rather than as a branch out here.
    """
    request.resolved_features()
    opinions = []
    for name in AGENT_ORDER:
        agent = agents.get(name)
        if agent is None:
            continue
        opinions.append(agent.opine(request))
    return opinions


def rank_agreement(
    ranker_order: Sequence[int],
    feature_dicts: Sequence[Dict[str, float]],
) -> Optional[float]:
    """
    Whether the variant ranker and GC3 order a group the same way.

    Returns Spearman rho between the two orderings, so positive means they
    agree and negative means the ranker is driving in the opposite direction to
    the one descriptor that is positive on both measured Welch series. This is
    the check the redesign tab already performs on a single pair, generalised to
    a group and promoted to a term of the agreement score.

    None when there are fewer than three sequences, where a rank correlation is
    not defined, or when GC3 does not vary.
    """
    if len(ranker_order) < 3:
        return None
    gc3 = GC3Baseline.score(feature_dicts)
    if len({g for g in gc3 if g == g}) < 2:
        return None

    from scipy.stats import spearmanr

    ranker_rank = [0] * len(ranker_order)
    for position, index in enumerate(ranker_order):
        ranker_rank[index] = position
    # Negate GC3 so that both vectors are "position, best first".
    result = spearmanr(ranker_rank, [-g for g in gc3])
    rho = float(result.statistic)
    return rho if rho == rho else None


def assess(
    sequence: str,
    model: Any = None,
    explainer: Any = None,
    feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    comparison: Sequence[str] = (),
    leader_5prime: Optional[str] = None,
    agents: Optional[Dict[str, Any]] = None,
) -> CriticVerdict:
    """
    The whole graph for one construct: features, four agents, one verdict.

    This is the entry point the application calls when the agent path is
    enabled. It never raises for an input the agents cannot handle; an input
    nothing can answer comes back as the `not_answerable` verdict.
    """
    request = AgentRequest(sequence=sequence, comparison=tuple(comparison), leader_5prime=leader_5prime)
    agents = agents or build_agents(model=model, explainer=explainer, feature_ranges=feature_ranges)
    opinions = run_agents(request, agents)

    agreement = None
    variant = next((o for o in opinions if o.agent == VARIANT and o.applies), None)
    if variant is not None and variant.ranking:
        order = [i for i, _ in sorted(
            ((entry_index, entry) for entry_index, entry in enumerate(variant.ranking)),
            key=lambda pair: pair[1]["rank"],
        )]
        from src.features import extract_features

        feature_dicts = [extract_features(entry["sequence"]) for entry in variant.ranking]
        agreement = rank_agreement(order, feature_dicts)

    verdict = combine(opinions, rank_agreement=agreement)
    logger.info(
        f"Graph finished: state '{verdict.state}', {verdict.agreement.n_applying} agent(s) applied, "
        f"{verdict.agreement.n_abstained} abstained."
    )
    return verdict
