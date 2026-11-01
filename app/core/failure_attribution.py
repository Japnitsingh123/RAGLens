"""
Failure Attribution Engine for RAG Pipelines.
Diagnoses root causes when RAG pipelines underperform:
1. Context Limits: Truncation, boundary cutoff, or insufficient context window size.
2. Embedding Weakness: Vocabulary mismatch, acronym blindness, or semantic asymmetry.
3. Hallucination: Model generating claims unsupported by retrieved context.
"""

import re
from typing import List, Dict, Any, Optional
from app.models.schemas import FailureAttribution, RootCauseType, RagasMetrics

class FailureAttributionEngine:
    """
    Analyzes retrieval contexts, query characteristics, and Ragas metrics
    to attribute the primary root cause of pipeline failures.
    """

    @classmethod
    def diagnose(
        cls,
        strategy: str,
        query: str,
        answer: str,
        contexts: List[str],
        ragas_metrics: RagasMetrics,
        ground_truth: Optional[str] = None
    ) -> FailureAttribution:
        total_context_chars = sum(len(c) for c in contexts)
        avg_chunk_len = (total_context_chars / len(contexts)) if contexts else 0

        recall = getattr(ragas_metrics, "context_recall", ragas_metrics.answer_completeness or 0.8)
        precision = getattr(ragas_metrics, "context_precision", ragas_metrics.answer_correctness or 0.8)

        # 1. Analyze Context Limits Severity
        context_limit_score = 0.0
        # If chunks are short (< 150 chars) and recall or completeness is low
        if avg_chunk_len < 180 and strategy == "naive":
            context_limit_score += 0.55
        if recall < 0.70:
            context_limit_score += 0.35
        context_limit_score = min(1.0, round(context_limit_score, 2))

        # 2. Analyze Embedding Weakness Severity
        # Check for technical acronyms, codes, numbers in query
        tech_entities = re.findall(r"\b[A-Z0-9-]{3,}\b|\b\d+(?:\.\d+)?%?\b", query)
        embedding_weakness_score = 0.0
        if tech_entities and strategy == "naive":
            # Pure dense search often dilutes exact acronyms
            embedding_weakness_score += 0.60
        if precision < 0.65 and context_limit_score < 0.6:
            embedding_weakness_score += 0.30
        embedding_weakness_score = min(1.0, round(embedding_weakness_score, 2))

        # 3. Analyze Hallucination Severity
        hallucination_score = round(max(0.0, 1.0 - ragas_metrics.faithfulness), 2)

        # Determine Primary Root Cause
        scores = [
            (RootCauseType.HALLUCINATION, hallucination_score),
            (RootCauseType.CONTEXT_LIMITS, context_limit_score),
            (RootCauseType.EMBEDDING_WEAKNESS, embedding_weakness_score)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_cause, top_score = scores[0]

        if top_score < 0.35 and ragas_metrics.faithfulness >= 0.85 and ragas_metrics.answer_correctness >= 0.80:
            primary_cause = RootCauseType.NONE
            details = "Pipeline operating optimally. Context is well-grounded and embedding recall is high."
            recommendation = "Strategy is production-ready for this query profile."
        elif top_cause == RootCauseType.CONTEXT_LIMITS:
            primary_cause = RootCauseType.CONTEXT_LIMITS
            details = (
                f"Context limits diagnosed: Chunk size ({int(avg_chunk_len)} chars) truncated crucial neighboring facts. "
                f"The generator missed peripheral context required for complete answer synthesis."
            )
            recommendation = "Upgrade to Parent-Child chunk expansion or increase chunk size to 500+ tokens to preserve document context boundaries."
        elif top_cause == RootCauseType.EMBEDDING_WEAKNESS:
            primary_cause = RootCauseType.EMBEDDING_WEAKNESS
            matched_terms = ", ".join(tech_entities[:3]) if tech_entities else "domain terms"
            details = (
                f"Embedding weakness diagnosed: Dense vector search suffered semantic dilution on specific technical tokens ({matched_terms}). "
                f"Pure cosine distance failed to prioritize exact lexical matches."
            )
            recommendation = "Adopt Hybrid BM25 with Reciprocal Rank Fusion (RRF) or Cohere Cross-Encoder reranking to overcome lexical vocabulary mismatch."
        else:
            primary_cause = RootCauseType.HALLUCINATION
            details = (
                f"Hallucination detected: Faithfulness score is {int(ragas_metrics.faithfulness * 100)}%. "
                f"Model generated claims that could not be corroborated from the retrieved context passages."
            )
            recommendation = "Lower LLM generation temperature, constrain prompt system guardrails, or implement strict claim extraction verification."

        return FailureAttribution(
            primary_root_cause=primary_cause,
            context_limits_score=context_limit_score,
            embedding_weakness_score=embedding_weakness_score,
            hallucination_score=hallucination_score,
            diagnosis_details=details,
            recommendation=recommendation
        )
