"""
Agent D, mRNA structure and initiation.

Question: does the 5' region of the transcript fold so as to block initiation.

This is the one agent in the plan blocked by a missing input rather than a
missing dataset, and the distinction matters because the fix is a small change
to the interface rather than a download.

`feature_table_v4.csv` carries Boel's own RNAstructure energies, Gz48v and
Gave48, computed over the expression vector's 5' untranslated region. They are
the strongest features in that table and the pipeline deliberately excludes
them from the served model, because the application cannot compute them from a
pasted coding sequence: it does not know which vector the sequence is going
into. A model fitted on them would load and then never run. Welch has the same
problem from the other side, where the 5' leader is a single 71 nt sequence
shared by all 81 constructs, so no leader derived feature can rank them.

The information is not missing from biology. It is missing from the input. So
this agent asks for it: hand it a `leader_5prime` on the request, or name a
vector, and the transcript's initiation window becomes computable from leader
plus insert.

The agent abstains with a precondition rather than a gap. No leader means no
opinion, which is correct behaviour and not a failure, and it is why the vector
selector in the interface is a feature of this agent rather than a decoration.

Gate D has two parts and the second is a test of the interface change rather
than of the model: the agent must beat the GC of the first 48 nucleotides alone
at predicting the lowest expression band, which is the baseline that catches a
folding feature that is really a GC feature under another name, and leader plus
insert folding must beat insert only folding on the same rows. If the second
fails, the vector input is not earning its place and should be dropped.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.agents.base import (
    GATE_BLOCKED,
    GATE_FAILED,
    GATE_PASSED,
    GATE_UNGATED,
    STRUCTURE,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    DomainCheck,
    GateResult,
    abstain,
)

logger = logging.getLogger("SynthBio.agents.structure")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.structure] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "models/agent_d_structure.joblib"
DEFAULT_METADATA_PATH = "models/agent_d_structure_metadata.json"

# Features of the initiation region, and nothing else. This agent does not
# answer "which tier", so it is not given length, composition or protein
# features: a narrow agent is one that can be gated against a named baseline.
#
# `folding_energy`, the GC derived proxy, is deliberately absent. It is a
# strictly monotonic function of 5' GC, measured at Spearman -1.000 against the
# GC of the first 48 nt on Cambray's 74,900 variants, so inside this agent it
# would duplicate GC and split the attribution between two names for one thing.
STRUCTURE_FEATURES: Tuple[str, ...] = (
    "initiation_folding_leader",
    "initiation_folding_insert",
    "head_folding_48",
    "gc_head48",
    "rare_codons_head30",
)

GATE_D_NOT_RUN = GateResult(
    status=GATE_BLOCKED,
    summary=(
        "Gate D has not been run because this agent has no artifact. Recover the sequences with "
        "`python scripts/fetch_datasets.py --source boel`, then train and gate in one step with "
        "`python scripts/train_agent_d.py`."
    ),
    report_path="reports/gates.md",
)

NO_LEADER_REASON = (
    "The structure agent needs the expression vector's 5' leader, because initiation folding is a "
    "property of leader plus insert and not of the coding sequence alone. No vector was declared, "
    "so this agent has no opinion. The 5' folding figures shown in the diagnostics are computed "
    "over the construct's own head only."
)


def structure_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """The feature matrix this agent sees, in a fixed column order."""
    missing = [c for c in STRUCTURE_FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"Initiation features missing: {', '.join(missing)}.")
    return features[list(STRUCTURE_FEATURES)]


class StructureAgent:
    """
    Agent D. Scores the initiation region once a vector is declared.

    Abstains for three distinct reasons and says which: no leader on the
    request, no artifact on disk, or an artifact whose recorded gate is not a
    pass. The first is a precondition and the other two are the usual
    discipline.
    """

    name = STRUCTURE

    def __init__(
        self,
        model: Any = None,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        require_gate: bool = True,
        load: bool = True,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = model
        self.metadata: Dict[str, Any] = {}
        self.require_gate = require_gate
        if self.model is None and load:
            self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.info(f"No structure artifact at '{self.model_path}'; Agent D will abstain.")
            return
        try:
            import joblib

            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded structure model {type(self.model).__name__} from '{self.model_path}'.")
        except Exception as err:
            logger.error(f"Could not load the structure model from '{self.model_path}': {err}")
            self.model = None
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as err:
                logger.warning(f"Could not read '{self.metadata_path}': {err}")

    # -- provenance ---------------------------------------------------------

    def gate(self) -> GateResult:
        """The gate recorded beside the artifact, or the blocked gate."""
        if not self.metadata:
            return GATE_D_NOT_RUN
        recorded = self.metadata.get("gate") or {}
        status = recorded.get("status", GATE_UNGATED)
        if status not in (GATE_PASSED, GATE_FAILED, GATE_BLOCKED, GATE_UNGATED):
            status = GATE_UNGATED
        measured = recorded.get("measured_on")
        try:
            measured_on = date.fromisoformat(measured) if measured else None
        except ValueError:
            measured_on = None
        return GateResult(
            status=status,
            summary=recorded.get("summary", "No gate summary was recorded beside this artifact."),
            measured_on=measured_on,
            report_path=recorded.get("report_path", "reports/gates.md"),
            baselines_beaten=tuple(recorded.get("baselines_beaten", ())),
        )

    def provenance(self) -> AgentProvenance:
        built = None
        if self.model is not None and os.path.exists(self.model_path):
            try:
                built = date.fromtimestamp(os.stat(self.model_path).st_mtime)
            except OSError:
                built = None
        vector = (self.metadata or {}).get("vector") or {}
        notes = [
            "Boel's Gz48v and Gave48 are servable only once a vector is declared. folding_energy, "
            "the GC derived proxy, is retired from this agent: it is a strictly monotonic function "
            "of 5' GC, so it would duplicate GC and split the attribution.",
        ]
        if vector.get("key"):
            notes.append(
                f"Fitted with the '{vector['key']}' leader, {len(vector.get('leader', ''))} nt. "
                + str(vector.get("source", ""))
            )
        return AgentProvenance(
            dataset=((self.metadata or {}).get("training_data") or {}).get("dataset"),
            artifact_path=self.model_path if self.model is not None else None,
            artifact_class=type(self.model).__name__ if self.model is not None else None,
            built_on=built,
            gate=self.gate(),
            notes=tuple(notes),
        )

    # -- the one function ---------------------------------------------------

    def opine(self, request: AgentRequest) -> AgentOpinion:
        provenance = self.provenance()

        if not request.leader_5prime:
            return abstain(self.name, NO_LEADER_REASON, provenance=provenance)

        if self.model is None:
            return abstain(
                self.name,
                "A 5' leader was supplied, but no structure artifact is on disk. "
                + provenance.gate.summary,
                provenance=provenance,
            )

        if self.require_gate and not provenance.gate.passed:
            return abstain(
                self.name,
                "This agent has not passed Gate D, so it is not connected. " + provenance.gate.summary,
                provenance=provenance,
            )

        features = self.initiation_features(request.sequence, request.leader_5prime)
        row = structure_feature_frame(pd.DataFrame([features])).astype(float)
        probability = float(self.model.predict_proba(row)[0][1])

        # The agent's own scalar is the initiation penalty: how much of the
        # lowest band risk it attributes to the 5' region. Its probability
        # vector is binary, and the critic maps it onto the common band scale
        # itself rather than this agent pretending to assign a tier.
        logger.info(
            f"Agent D: probability of the lowest band {probability:.3f} from the initiation window."
        )
        return AgentOpinion(
            agent=self.name,
            applies=True,
            score=probability,
            proba={0: 1.0 - probability, 1: probability},
            features={k: float(v) for k, v in features.items()},
            attribution=self._attribution(row),
            domain=self._domain(row, request),
            provenance=provenance,
        )

    @staticmethod
    def initiation_features(sequence: str, leader: str) -> Dict[str, float]:
        """
        The initiation window features, computed exactly as the trainer does.

        Shared with `scripts/train_agent_d.py` through this one function, so a
        gate result cannot come from a feature path the application does not
        use.
        """
        from src.features import (
            FOLDING_WINDOW_NT,
            calculate_folding_energy,
            calculate_initiation_folding,
            calculate_rare_codon_fraction,
        )

        head = sequence[:FOLDING_WINDOW_NT]
        values = {
            "initiation_folding_leader": calculate_initiation_folding(leader, sequence),
            "initiation_folding_insert": calculate_initiation_folding("", sequence),
            "head_folding_48": calculate_folding_energy(head),
            "gc_head48": 100.0 * (head.count("G") + head.count("C")) / max(len(head), 1),
            "rare_codons_head30": calculate_rare_codon_fraction(sequence, head_codons=30),
        }
        return {k: (float("nan") if v is None else float(v)) for k, v in values.items()}

    def _domain(self, row: pd.DataFrame, request: AgentRequest) -> DomainCheck:
        """
        Whether this construct's initiation region resembles the training set.

        The leader is part of the domain here, not just the insert: a model
        fitted with one vector's leader is being extrapolated when handed
        another, and the note says which leader it was fitted with.
        """
        from src.agents.domain import describe_domain

        check = describe_domain(
            row,
            (self.metadata or {}).get("feature_ranges"),
            sequence=None,
            check_identity=False,
        )
        notes = list(check.notes)
        vector = (self.metadata or {}).get("vector") or {}
        trained_leader = vector.get("leader") or ""
        if trained_leader and request.leader_5prime != trained_leader:
            notes.append(
                f"Fitted with the '{vector.get('key')}' leader of {len(trained_leader)} nt and asked "
                f"about a {len(request.leader_5prime)} nt leader, so the initiation window differs "
                "from the one it learned on."
            )
        notes.append(
            "No nearest neighbour identity check is run for this agent: it reads the initiation "
            "window, not the protein, so identity to a training protein says nothing about it."
        )
        return DomainCheck(
            out_of_range=check.out_of_range,
            nearest_identity=None,
            identity_checked=False,
            notes=tuple(notes),
        )

    def _attribution(self, row: pd.DataFrame) -> Dict[str, float]:
        """SHAP values for this construct, empty when SHAP is unavailable."""
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            values = np.asarray(explainer.shap_values(row)).reshape(-1)
            return {name: float(v) for name, v in zip(row.columns, values)}
        except Exception as err:
            logger.warning(f"SHAP attribution unavailable for Agent D: {err}")
            return {}
