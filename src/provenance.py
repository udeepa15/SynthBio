"""
Provenance reporting for SynthBio.

Describes where the numbers on screen came from: the trained model artifact or
the biophysical heuristic fallback, which features the artifact actually
accepts, which dataset it was built from, and when it was produced. A clone of
the repository without the gitignored weights runs in fallback mode, and the
interface has to say so rather than leaving it to a sidebar notice.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional
import logging
import os

logger = logging.getLogger("SynthBio.provenance")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.provenance] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

HEURISTIC = "heuristic"
TRAINED = "trained"

TRAINING_DATASET = "Boël et al. 2016 (JMB), NESG E. coli expression set"

# Features the application can compute from a pasted sequence. Kept here rather
# than imported from predict so that provenance has no dependency on inference.
SERVABLE_FEATURE_NAMES = [
    "GC_percent", "GC3", "CAI", "rare_codon_fraction", "rare_codons_head30",
    "head_folding_48", "genewide_folding_avg", "gravy_score", "pI", "length_codons",
]


@dataclass
class Provenance:
    """What produced the figures currently displayed."""

    mode: str
    model_class: Optional[str]
    model_path: Optional[str]
    model_built: Optional[datetime]
    model_features: List[str]
    dataset: Optional[str]
    analysed_at: datetime

    @property
    def is_trained(self) -> bool:
        return self.mode == TRAINED

    @property
    def mode_label(self) -> str:
        return "Trained model" if self.is_trained else "Heuristic fallback"

    def as_line(self) -> str:
        """One-line provenance string for display under the verdict."""
        parts = [f"Source: {self.mode_label}"]
        if self.is_trained:
            if self.model_class:
                parts.append(f"Model: {self.model_class}")
            if self.model_built:
                parts.append(f"Built: {self.model_built:%Y-%m-%d}")
            if self.model_features:
                parts.append(
                    f"Inputs: {len(self.model_features)} features ({', '.join(self.model_features)})"
                )
            if self.dataset:
                parts.append(f"Training data: {self.dataset}")
        else:
            parts.append("Model: none loaded, biophysical rules only")
            parts.append("Training data: not applicable")
        parts.append(f"Analysed: {self.analysed_at:%Y-%m-%d %H:%M}")
        return "  ·  ".join(parts)

    def caveats(self) -> List[str]:
        """Statements a viewer needs in order to read the figures correctly."""
        notes: List[str] = []
        if not self.is_trained:
            notes.append(
                "No trained model artifact was found, so the tier and the risk figure come from "
                "hardcoded biophysical rules, not from a model fitted to experimental data. "
                "SHAP attribution is unavailable in this mode."
            )
            return notes
        missing = [f for f in SERVABLE_FEATURE_NAMES if f not in (self.model_features or [])]
        if self.model_features and missing:
            notes.append(
                f"The loaded artifact was fitted on {len(self.model_features)} of the "
                f"{len(SERVABLE_FEATURE_NAMES)} computable features. "
                f"{', '.join(missing)} are computed and shown, and are used by the diagnostics "
                "below, but do not enter the model's prediction."
            )
        notes.append(
            "The model distinguishes one gene from another. It should not be used to choose "
            "between synonymous codon variants of the same gene: on Welch's measured variant "
            "series it ranks those backwards."
        )
        return notes


def describe(model: Any, model_path: Optional[str] = None) -> Provenance:
    """Build a Provenance record for the model currently in use, if any."""
    now = datetime.now()

    if model is None:
        logger.info("Provenance: heuristic fallback mode (no model artifact).")
        return Provenance(
            mode=HEURISTIC,
            model_class=None,
            model_path=None,
            model_built=None,
            model_features=[],
            dataset=None,
            analysed_at=now,
        )

    built = None
    if model_path and os.path.exists(model_path):
        try:
            built = datetime.fromtimestamp(os.stat(model_path).st_mtime)
        except OSError as err:
            logger.warning(f"Could not read model artifact timestamp: {err}")

    features = []
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        features = [str(n) for n in names]

    prov = Provenance(
        mode=TRAINED,
        model_class=type(model).__name__,
        model_path=model_path,
        model_built=built,
        model_features=features,
        dataset=TRAINING_DATASET,
        analysed_at=now,
    )
    logger.info(f"Provenance: {prov.as_line()}")
    return prov
