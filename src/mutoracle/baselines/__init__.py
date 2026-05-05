"""Phase 7 response-level baseline harnesses."""

from mutoracle.baselines.calibration import (
    LabeledScore,
    ThresholdCalibration,
    tune_threshold_validation_only,
)
from mutoracle.baselines.metarag_baseline import (
    ClaimVariant,
    LexicalNLIBackend,
    MetaRAGBaseline,
    NLIClaimVerifier,
    NoopVariantGenerator,
    SentenceClaimExtractor,
    SimpleMetamorphicVariantGenerator,
    SpacyClaimExtractor,
)
from mutoracle.baselines.ragas_baseline import (
    OfficialRagasFaithfulnessScorer,
    RagasBaseline,
)
from mutoracle.baselines.runner import run_baselines, write_baseline_outputs
from mutoracle.baselines.schema import (
    BaselineExample,
    BaselineManifest,
    BaselineResult,
    run_id_for,
)

__all__ = [
    "BaselineExample",
    "BaselineManifest",
    "BaselineResult",
    "ClaimVariant",
    "LabeledScore",
    "LexicalNLIBackend",
    "MetaRAGBaseline",
    "NLIClaimVerifier",
    "NoopVariantGenerator",
    "OfficialRagasFaithfulnessScorer",
    "RagasBaseline",
    "SentenceClaimExtractor",
    "SimpleMetamorphicVariantGenerator",
    "SpacyClaimExtractor",
    "ThresholdCalibration",
    "run_baselines",
    "run_id_for",
    "tune_threshold_validation_only",
    "write_baseline_outputs",
]
