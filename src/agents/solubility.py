"""
Agent B, solubility. Declared and abstaining.

Question: given that it expresses, will it be soluble rather than inclusion
bodies.

This agent exists as a declared empty slot, which is deliberate. The plan's
scope recommendation is two agents built properly rather than four built
thinly, with the other two kept in the design and in the interface as declared,
abstaining agents with their gates written down. An architecture with two
working agents and two honest gaps is a stronger piece of work than four agents
that each sit at their baseline.

What this agent needs before it can be built, from the plan's section 3:

- RP3Net SGC Stockholm, the only public set reported to ship construct DNA
  including tags for every entry. The DNA claim has to be verified on the real
  download before anything is planned around it.
- SoluProt's TargetTrack E. coli set, 10,912 sequences clustered at 25 per cent
  identity, protein only, so the nucleotide features need per row recovery.
- UESolDS filtered, only after re deriving the split by intersecting against the
  eSOL ORF list, because the release bundles cell free rows that violate the in
  vivo host filter.

Its baseline to beat is already measured and is not trivial. GRAVY above 0.4
flags 86 of 6,348 Boel genes, 63 per cent of which land in the lowest
expression band against a base rate of 28 per cent, and only 6 per cent in the
top two bands against about 50 per cent. High precision, low recall. Until
Agent B beats that on AUROC and Matthews correlation under clustered splits,
the rule is the better predictor and the interface should say so.

This is milestone M5.
"""

from __future__ import annotations

from src.agents.base import (
    GATE_BLOCKED,
    SOLUBILITY,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    GateResult,
    abstain,
)

GATE_B = GateResult(
    status=GATE_BLOCKED,
    summary=(
        "Gate B has not been run: this agent has no training data and no artifact. It must beat "
        "the majority class, length alone and GRAVY alone on AUROC and Matthews correlation under "
        "splits clustered at 30 per cent identity, and must be calibrated, with a Brier score and "
        "a reliability curve reported."
    ),
    report_path="reports/gates.md",
)

ABSTAIN_REASON = (
    "The solubility agent is declared but not built. No solubility dataset is in the working tree. "
    "The validated GRAVY rule, GRAVY above 0.4, is the current best predictor of a construct "
    "landing in the lowest expression band and is shown in the diagnostics instead."
)


class SolubilityAgent:
    """Agent B. Always abstains, and says what it would need not to."""

    name = SOLUBILITY

    def provenance(self) -> AgentProvenance:
        return AgentProvenance(
            dataset=None,
            gate=GATE_B,
            notes=(
                "Not built. The modelled object is still an open decision: the tagged construct as "
                "it is really expressed, or the native protein. The plan recommends the tagged "
                "construct, since RP3Net ships tagged DNA and a tag changes GRAVY, pI and folding.",
            ),
        )

    def opine(self, request: AgentRequest) -> AgentOpinion:
        return abstain(self.name, ABSTAIN_REASON, provenance=self.provenance())
