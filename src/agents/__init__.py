"""
The multi agent construct predictor.

Four narrow agents and a critic that is allowed to say they disagree. Each
agent answers one question, declares the domain it is competent over, carries
the result of its own named gate, and returns one `AgentOpinion`. An agent that
has not passed its gate abstains; an abstention is an answer here, not a
failure.

Current standing, which the interface is expected to show rather than summarise:

- Agent A, expression tier. Wraps the served artifact unchanged. Now measured
  on the identity clustered split the gate asks for, with the reserved critic
  pool held out of training, tuning and reporting alike.
- Agent B, solubility. Declared, not built. No solubility dataset on disk.
- Agent C, codon variant ranking. Built, trained on Cambray 2018 and gated
  against Welch 2009. It is positive on both Welch series, so unlike the tier
  model it does not order a synonymous series backwards, but it does not beat
  GC3 alone on either, which is the bar. So it abstains, and the redesign loop
  keeps ranking on the tier model with the GC3 corroboration check as its guard.
  `reports/gates.md` carries the numbers.
- Agent D, structure. Built, trained on Boel with the vector's 5' leader
  declared, and it passes Gate D on the identity clustered split: AUROC 0.612
  against 0.584 for the GC of the first 48 nt alone, and leader plus insert
  folding separates from insert only, 0.612 against 0.603. The second margin is
  slim and its interval only just excludes zero, so the claim is that declaring
  the vector helps, not that it helps much. It applies only when a vector is
  declared, which is why the application now asks.

The critic exists and is trained. On the reserved pool it did not beat the
best single agent, macro F1 0.375 against 0.426, so the single agent produces
the tier and the critic is kept for the agreement score, the four verdict
states and the abstention policy. Its conflict threshold is fitted on that pool
rather than declared. Declining on disagreement lifts accuracy on the answered
subset by about three points while declining about half the inputs, which is a
working mechanism on a weak signal.

Nothing here should be presented as an accuracy improvement. If the specialists
pass their gates the combined system will be better than one overloaded model,
but the gain will come from the specialists and their data, not from the act of
combining them.
"""

from src.agents.base import (  # noqa: F401
    AGENT_ORDER,
    AGENT_QUESTIONS,
    EXPRESSION,
    GATE_BLOCKED,
    GATE_FAILED,
    GATE_PASSED,
    GATE_UNGATED,
    SOLUBILITY,
    STRUCTURE,
    VARIANT,
    Agent,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    DomainCheck,
    GateResult,
    abstain,
)
from src.agents.critic import (  # noqa: F401
    AGREED,
    AGREED_OUT_OF_DOMAIN,
    CONFLICTED,
    NOT_ANSWERABLE,
    AgreementScore,
    CriticVerdict,
    combine,
)
from src.agents.graph import assess, build_agents, rank_agreement, run_agents  # noqa: F401
from src.agents.variant import GC3Baseline, VariantAgent  # noqa: F401

__all__ = [
    "AGENT_ORDER",
    "AGENT_QUESTIONS",
    "AGREED",
    "AGREED_OUT_OF_DOMAIN",
    "CONFLICTED",
    "NOT_ANSWERABLE",
    "Agent",
    "AgentOpinion",
    "AgentProvenance",
    "AgentRequest",
    "AgreementScore",
    "CriticVerdict",
    "DomainCheck",
    "GC3Baseline",
    "GateResult",
    "VariantAgent",
    "abstain",
    "assess",
    "build_agents",
    "combine",
    "rank_agreement",
    "run_agents",
]
