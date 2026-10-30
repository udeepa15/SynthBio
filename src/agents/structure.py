"""
Agent D, mRNA structure and initiation. Declared and abstaining.

Question: does the 5' region of the transcript fold so as to block initiation.

Unlike the other three, this agent is not blocked by a missing dataset. It is
blocked by a missing input, and the distinction matters because the fix is a
small change to the interface rather than a download.

`feature_table_v4.csv` carries Boel's own RNAstructure energies, Gz48v and
Gave48, computed over the expression vector's 5' untranslated region. They are
the strongest features in the Boel table and the pipeline deliberately excludes
them from the served model, because the application cannot compute them from a
pasted coding sequence: it does not know which vector the sequence will go
into. A model fitted on them would load and then never run. Welch has the same
shape of problem from the other side, where the 5' leader is a single 71 nt
sequence shared by all 81 constructs, so no leader derived feature can rank
them.

The information is not missing from biology. It is missing from the input. If
the interface asks which vector the construct is going into, with pET as the
default, the leader becomes known, the transcript's true first 48 nt can be
assembled from leader plus insert, and the strongest feature in the Boel table
becomes servable for the first time.

So this agent abstains with a precondition rather than with a gap: hand it a
`leader_5prime` on the request and the work to build it becomes possible. Its
gate then has two parts, and the second one is a test of the interface change
rather than of the model: leader plus insert folding must beat insert only
folding on the same rows, or the vector input is not earning its place and
should be dropped.

This is milestone M6.
"""

from __future__ import annotations

from src.agents.base import (
    GATE_BLOCKED,
    STRUCTURE,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    GateResult,
    abstain,
)

GATE_D = GateResult(
    status=GATE_BLOCKED,
    summary=(
        "Gate D has not been run: this agent has no artifact. It must beat the GC of the first "
        "48 nt alone at predicting the lowest expression band under the clustered split, which is "
        "the baseline that catches a folding feature that is really a GC feature wearing another "
        "label, and leader plus insert folding must beat insert only folding on the same rows."
    ),
    report_path="reports/gates.md",
)

NO_LEADER_REASON = (
    "The structure agent needs the expression vector's 5' leader, because initiation folding is a "
    "property of leader plus insert and not of the coding sequence alone. No vector was declared, "
    "so this agent has no opinion. The 5' folding figures shown in the diagnostics are computed "
    "over the construct's own head only."
)

NOT_BUILT_REASON = (
    "A 5' leader was supplied, but the structure agent has no artifact yet. Before one is trained, "
    "the plan requires the folding implementation itself to be unit tested against the Cambray 5' "
    "library, which varies the 5' region against a fixed reporter."
)


class StructureAgent:
    """Agent D. Abstains, and distinguishes a missing input from a missing model."""

    name = STRUCTURE

    def provenance(self) -> AgentProvenance:
        return AgentProvenance(
            dataset=None,
            gate=GATE_D,
            notes=(
                "Boel's Gz48v and Gave48 become servable only once a vector is declared. "
                "folding_energy, the GC derived proxy, is retired from this agent: it is a strictly "
                "monotonic function of 5' GC, so it would duplicate GC and split the attribution.",
            ),
        )

    def opine(self, request: AgentRequest) -> AgentOpinion:
        reason = NOT_BUILT_REASON if request.leader_5prime else NO_LEADER_REASON
        return abstain(self.name, reason, provenance=self.provenance())
