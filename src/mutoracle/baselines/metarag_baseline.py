"""Transparent MetaRAG-style baseline approximation."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from mutoracle.baselines.schema import (
    BaselineResult,
    classify_faithfulness,
    merge_model_ids,
    run_id_for,
    run_metadata_model_ids,
    run_metadata_value,
)
from mutoracle.contracts import RAGRun
from mutoracle.oracles.nli import NLIBackend, TransformersNLIBackend

CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
LOCAL_SYNONYMS = {
    "wrote": "authored",
    "uses": "employs",
    "created": "made",
}
LOCAL_ANTONYMS = {
    "before": "after",
    "increase": "decrease",
    "included": "excluded",
    "supports": "contradicts",
    "true": "false",
}


class ClaimExtractor(Protocol):
    """Extract atomic-ish claims from a generated answer."""

    def extract(self, answer: str) -> list[str]:
        """Return claims that can be checked against retrieved context."""


class ClaimVerifier(Protocol):
    """Verify one claim against a context."""

    model_id: str

    def score_claim(self, *, context: str, claim: str) -> float:
        """Return an entailment-like score in [0, 1]."""


@dataclass(frozen=True)
class ClaimVariant:
    """Metamorphic variant derived from one extracted claim."""

    text: str
    kind: Literal["synonym", "antonym", "factoid"]
    expected_supported: bool


class VariantGenerator(Protocol):
    """Generate metamorphic claim variants."""

    def generate(self, claim: str) -> list[ClaimVariant]:
        """Return variants with expected support behavior."""


@dataclass(frozen=True)
class SentenceClaimExtractor:
    """Deterministic sentence-based claim extraction approximation."""

    min_words: int = 3

    def extract(self, answer: str) -> list[str]:
        """Return sentence-level claims from an answer."""

        claims: list[str] = []
        for raw_part in CLAIM_SPLIT_PATTERN.split(answer.strip()):
            claim = raw_part.strip()
            if not claim:
                continue
            if claim[-1] in ".!?":
                claim = claim[:-1].strip()
            if len(claim.split()) >= self.min_words:
                claims.append(claim)
        return claims


@dataclass(frozen=True)
class SpacyClaimExtractor:
    """spaCy claim extractor with deterministic sentence fallback."""

    model_name: str = "en_core_web_sm"
    fallback: SentenceClaimExtractor = SentenceClaimExtractor()

    def extract(self, answer: str) -> list[str]:
        """Return sentence and noun-chunk-like claims when spaCy is available."""

        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError:
            return self.fallback.extract(answer)

        try:
            nlp = spacy.load(self.model_name)
        except OSError:
            return self.fallback.extract(answer)

        doc = nlp(answer)
        claims = [
            sentence.text.strip().rstrip(".!?")
            for sentence in doc.sents
            if len(sentence.text.split()) >= self.fallback.min_words
        ]
        return claims or self.fallback.extract(answer)


class NLIClaimVerifier:
    """Claim verifier backed by the same local NLI model family as MutOracle."""

    def __init__(
        self,
        *,
        backend: NLIBackend | None = None,
        model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ) -> None:
        self.model_id = model_id
        self._backend = backend or TransformersNLIBackend(model_id)

    def score_claim(self, *, context: str, claim: str) -> float:
        """Return the entailment probability for one claim."""

        probabilities = self._backend.probabilities(
            premise=context,
            hypothesis=claim,
        )
        return _entailment_probability(probabilities)


@dataclass(frozen=True)
class LexicalNLIBackend:
    """Credential-free NLI-shaped backend for smoke tests."""

    entailment_cutoff: float = 0.55

    def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
        """Return deterministic pseudo-NLI probabilities from token overlap."""

        premise_tokens = _content_tokens(premise)
        hypothesis_tokens = _content_tokens(hypothesis)
        if not hypothesis_tokens:
            return {"entailment": 1.0, "neutral": 0.0}
        overlap = len(premise_tokens & hypothesis_tokens) / len(hypothesis_tokens)
        if overlap >= self.entailment_cutoff:
            return {"entailment": overlap, "neutral": 1.0 - overlap}
        return {"entailment": overlap, "contradiction": 1.0 - overlap}


@dataclass(frozen=True)
class SimpleMetamorphicVariantGenerator:
    """WordNet-aware variant generator with local deterministic fallback rules."""

    max_variants_per_claim: int = 3

    def generate(self, claim: str) -> list[ClaimVariant]:
        """Return synonym, antonym, and factoid variants for a claim."""

        variants: list[ClaimVariant] = []
        variants.extend(_wordnet_variants(claim))
        variants.extend(_local_lexical_variants(claim))
        variants.extend(_factoid_variants(claim))

        deduped: list[ClaimVariant] = []
        seen = {claim}
        for variant in variants:
            if variant.text in seen:
                continue
            seen.add(variant.text)
            deduped.append(variant)
            if len(deduped) >= self.max_variants_per_claim:
                break
        return deduped


@dataclass(frozen=True)
class NoopVariantGenerator:
    """Variant generator for exact claim-verification-only tests."""

    def generate(self, claim: str) -> list[ClaimVariant]:
        """Return no variants."""

        del claim
        return []


class MetaRAGBaseline:
    """Local MetaRAG approximation using claim extraction plus NLI verification."""

    name = "metarag"

    def __init__(
        self,
        *,
        extractor: ClaimExtractor | None = None,
        verifier: ClaimVerifier | None = None,
        variant_generator: VariantGenerator | None = None,
        entailment_threshold: float = 0.5,
    ) -> None:
        self._extractor = extractor or SpacyClaimExtractor()
        self._verifier = verifier or NLIClaimVerifier()
        self._variant_generator = (
            variant_generator or SimpleMetamorphicVariantGenerator()
        )
        if entailment_threshold < 0.0 or entailment_threshold > 1.0:
            msg = "entailment_threshold must be in [0, 1]."
            raise ValueError(msg)
        self._entailment_threshold = entailment_threshold

    def run(
        self,
        run: RAGRun,
        *,
        threshold: float = 0.5,
        reference: str | None = None,
    ) -> BaselineResult:
        """Score one RAG output with the MetaRAG approximation."""

        del reference
        started = time.perf_counter()
        claims = self._extractor.extract(run.answer)
        context = "\n\n".join(passage.strip() for passage in run.passages)
        claim_scores = [
            self._verifier.score_claim(context=context, claim=claim) for claim in claims
        ]
        variant_records: list[dict[str, object]] = []
        variant_violations = 0
        if not claims:
            faithfulness = 1.0
            supported = 0
            empty_claim_set = True
        else:
            supported = sum(
                1 for score in claim_scores if score >= self._entailment_threshold
            )
            for claim in claims:
                for variant in self._variant_generator.generate(claim):
                    variant_score = self._verifier.score_claim(
                        context=context,
                        claim=variant.text,
                    )
                    violation = _is_variant_violation(
                        variant,
                        score=variant_score,
                        threshold=self._entailment_threshold,
                    )
                    if violation:
                        variant_violations += 1
                    variant_records.append(
                        {
                            "claim": claim,
                            "variant": variant.text,
                            "kind": variant.kind,
                            "expected_supported": variant.expected_supported,
                            "score": variant_score,
                            "violation": violation,
                        }
                    )
            original_support_rate = supported / len(claims)
            if variant_records:
                variant_violation_rate = variant_violations / len(variant_records)
                faithfulness = original_support_rate * (1.0 - variant_violation_rate)
            else:
                faithfulness = original_support_rate
            empty_claim_set = False

        faithfulness = min(1.0, max(0.0, faithfulness))
        scoring_latency = time.perf_counter() - started
        generation_latency = run_metadata_value(
            run,
            ("generation", "latency_seconds"),
            run_metadata_value(run, ("latency", "generation_seconds"), 0.0),
        )
        generation_cost = run_metadata_value(
            run,
            ("generation", "estimated_cost_usd"),
            0.0,
        )
        total_latency = generation_latency + scoring_latency
        model_ids = merge_model_ids(
            run_metadata_model_ids(run),
            [self._verifier.model_id],
        )
        return BaselineResult(
            run_id=run_id_for(run),
            baseline_name=self.name,
            query=run.query,
            score=faithfulness,
            threshold=threshold,
            predicted_label=classify_faithfulness(
                score=faithfulness,
                threshold=threshold,
            ),
            latency_seconds=total_latency,
            cost_usd=generation_cost,
            model_ids=model_ids,
            scores={"faithfulness": faithfulness},
            metadata={
                "claim_count": len(claims),
                "claim_scores": claim_scores,
                "empty_claim_set": empty_claim_set,
                "supported_claims": supported,
                "entailment_threshold": self._entailment_threshold,
                "variant_count": len(variant_records),
                "variant_violations": variant_violations,
                "variants": variant_records,
                "implementation": "spacy_or_sentence_claims_plus_metamorphic_nli",
                "cost_scope": "generation_only",
                "latency_breakdown_seconds": {
                    "generation": generation_latency,
                    "baseline": scoring_latency,
                },
                "cost_breakdown_usd": {
                    "generation": generation_cost,
                },
            },
        )


def _entailment_probability(probabilities: dict[str, float]) -> float:
    for label, probability in probabilities.items():
        normalized = label.lower().replace("_", " ")
        if "entail" in normalized:
            return min(1.0, max(0.0, float(probability)))
    return 0.0


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2
    }


def _wordnet_variants(claim: str) -> list[ClaimVariant]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-not-found]
    except ImportError:
        return []

    words = claim.split()
    variants: list[ClaimVariant] = []
    for index, word in enumerate(words):
        replacement = _wordnet_replacement(word, wn=wn, antonym=False)
        if replacement:
            variants.append(
                ClaimVariant(
                    text=_replace_word(words, index, replacement),
                    kind="synonym",
                    expected_supported=True,
                )
            )
        replacement = _wordnet_replacement(word, wn=wn, antonym=True)
        if replacement:
            variants.append(
                ClaimVariant(
                    text=_replace_word(words, index, replacement),
                    kind="antonym",
                    expected_supported=False,
                )
            )
    return variants


def _wordnet_replacement(word: str, *, wn: Any, antonym: bool) -> str | None:
    try:
        synsets = wn.synsets(word)
    except LookupError:
        return None
    normalized_word = word.lower().strip(".,;:!?")
    for synset in synsets:
        for lemma in synset.lemmas():
            candidates = lemma.antonyms() if antonym else [lemma]
            for candidate in candidates:
                replacement = candidate.name().replace("_", " ")
                if replacement.lower() != normalized_word:
                    return cast("str", replacement)
    return None


def _local_lexical_variants(claim: str) -> list[ClaimVariant]:
    variants: list[ClaimVariant] = []
    words = claim.split()
    for index, word in enumerate(words):
        normalized = word.lower().strip(".,;:!?")
        if normalized in LOCAL_SYNONYMS:
            variants.append(
                ClaimVariant(
                    text=_replace_word(words, index, LOCAL_SYNONYMS[normalized]),
                    kind="synonym",
                    expected_supported=True,
                )
            )
        if normalized in LOCAL_ANTONYMS:
            variants.append(
                ClaimVariant(
                    text=_replace_word(words, index, LOCAL_ANTONYMS[normalized]),
                    kind="antonym",
                    expected_supported=False,
                )
            )
    return variants


def _factoid_variants(claim: str) -> list[ClaimVariant]:
    variants: list[ClaimVariant] = []
    for match in re.finditer(r"\b\d+\b", claim):
        number = int(match.group(0))
        replacement = str(number + 1)
        variants.append(
            ClaimVariant(
                text=f"{claim[: match.start()]}{replacement}{claim[match.end() :]}",
                kind="factoid",
                expected_supported=False,
            )
        )
    return variants


def _replace_word(words: Sequence[str], index: int, replacement: str) -> str:
    updated = list(words)
    updated[index] = replacement
    return " ".join(updated)


def _is_variant_violation(
    variant: ClaimVariant,
    *,
    score: float,
    threshold: float,
) -> bool:
    if variant.expected_supported:
        return score < threshold
    return score >= threshold


def score_claims(
    *,
    claims: Sequence[str],
    context: str,
    verifier: ClaimVerifier,
) -> list[float]:
    """Return claim scores for tests and experiment scripts."""

    return [verifier.score_claim(context=context, claim=claim) for claim in claims]
