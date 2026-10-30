"""
The multi agent construct predictor.

Four narrow agents and a critic that is allowed to say they disagree. Each
agent answers one question, declares the domain it is competent over, carries
the result of its own named gate, and returns one `AgentOpinion`. An agent that
has not passed its gate abstains; an abstention is an answer here, not a
failure.

Current standing, which the interface is expected to show rather than summarise:

- Agent A, expression tier. Wraps the served artifact unchanged. Gate blocked:
  identity clustered splits need MMseqs2, which is not installed. The training
  sequences themselves have been recovered, so this is now an install and not a
  missing dataset.
- Agent B, solubility. Declared, not built. No solubility dataset on disk.
- Agent C, codon variant ranking. Built, trained on Cambray 2018 and gated
  against Welch 2009. It is positive on both Welch series, so unlike the tier
  model it does not order a synonymous series backwards, but it does not beat
  GC3 alone on either, which is the bar. So it abstains, and the redesign loop
  keeps ranking on the tier model with the GC3 corroboration check as its guard.
  `reports/gates.md` carries the numbers.
- Agent D, structure. Declared, not built. Blocked by a missing input, the
  expression vector's 5' leader, rather than by a missing dataset.

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
