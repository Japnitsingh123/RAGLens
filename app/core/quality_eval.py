import re
import numpy as np
from typing import List, Dict, Any, Tuple
from app.models.schemas import QualityItemResult, QualitySummary
from app.core.rag_engine import DenseVectorizer

class QualityEvaluator:
    """
    RAG Quality Evaluator implementing the RAGAS framework metrics:
    1. Faithfulness: Is the generated answer grounded strictly in retrieved context?
    2. Answer Relevancy: Does the answer directly address the user's question?
    3. Context Precision: Are relevant context chunks ranked higher in the retrieved window?
    4. Context Recall: Did the retrieval capture all ground-truth facts?
    """
    def __init__(self):
        self.vectorizer = DenseVectorizer(dim=128)

    def _split_into_claims(self, text: str) -> List[str]:
        """Extract atomic factual claims from generated text."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []
        for s in raw_sentences:
            cleaned = s.strip()
            if len(cleaned) > 12:
                # Remove introductory fluff if any
                cleaned = re.sub(r'^(furthermore|additionally|moreover|in addition|according to the docs),?\s*', '', cleaned, flags=re.IGNORECASE)
                claims.append(cleaned)
        return claims if claims else [text.strip()]

    def _extract_keywords(self, text: str) -> set:
        stopwords = {
            "the", "and", "is", "in", "it", "to", "of", "for", "with", "on", "at", 
            "by", "this", "that", "are", "be", "as", "from", "or", "an", "what", 
            "which", "how", "does", "have", "has", "when", "where", "our", "all"
        }
        words = re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())
        return {w for w in words if w not in stopwords}

    def _verify_claim_against_context(self, claim: str, contexts: List[str]) -> Tuple[bool, float]:
        """Check if claim words and semantics are entailed by context."""
        claim_kws = self._extract_keywords(claim)
        if not claim_kws:
            return True, 1.0

        claim_emb = self.vectorizer.encode(claim)
        best_sim = 0.0
        best_kw_overlap = 0.0

        for ctx in contexts:
            ctx_kws = self._extract_keywords(ctx)
            overlap = len(claim_kws.intersection(ctx_kws)) / float(len(claim_kws))
            ctx_emb = self.vectorizer.encode(ctx)
            sim = float(np.dot(claim_emb, ctx_emb))
            if overlap > best_kw_overlap:
                best_kw_overlap = overlap
            if sim > best_sim:
                best_sim = sim

        # Entailed if high keyword overlap and semantic alignment
        is_supported = (best_kw_overlap >= 0.50) or (best_kw_overlap >= 0.35 and best_sim >= 0.65)
        entailment_score = max(best_kw_overlap, best_sim)
        return is_supported, entailment_score

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Faithfulness = (Supported Claims) / (Total Claims in Answer)
        Flags unsupported claims (hallucinations).
        """
        if not answer.strip():
            return 0.0, [], []

        claims = self._split_into_claims(answer)
        if not claims:
            return 1.0, [], []

        supported = []
        unsupported = []

        for claim in claims:
            is_supported, _ = self._verify_claim_against_context(claim, contexts)
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

        score = len(supported) / float(len(claims))
        return round(score, 3), supported, unsupported

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Measures how well the answer addresses the question.
        Calculates semantic alignment and penalizes brevity/evasion.
        """
        if not answer.strip() or "No relevant documentation found" in answer:
            return 0.25

        q_emb = self.vectorizer.encode(question)
        a_emb = self.vectorizer.encode(answer)
        cos_sim = float(np.dot(q_emb, a_emb))

        q_kws = self._extract_keywords(question)
        a_kws = self._extract_keywords(answer)
        overlap = len(q_kws.intersection(a_kws)) / max(len(q_kws), 1)

        # Relevancy combines semantic similarity and keyword coverage
        raw_relevancy = (0.6 * cos_sim) + (0.4 * overlap)
        relevancy_score = min(max(raw_relevancy, 0.0), 1.0)
        return round(relevancy_score, 3)

    def evaluate_context_precision(self, question: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Context Precision@K: Mean Average Precision of relevant chunks.
        Higher if relevant chunks are ranked higher.
        """
        if not contexts:
            return 0.0

        gt_kws = self._extract_keywords(ground_truth)
        q_kws = self._extract_keywords(question)
        target_kws = gt_kws.union(q_kws)

        relevance_flags: List[int] = []
        for ctx in contexts:
            ctx_kws = self._extract_keywords(ctx)
            overlap = len(target_kws.intersection(ctx_kws))
            # Chunk is relevant if it shares at least 2 key terms or 25% of target terms
            is_rel = 1 if (overlap >= 2 or (len(target_kws) > 0 and overlap / len(target_kws) >= 0.25)) else 0
            relevance_flags.append(is_rel)

        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            return 0.0

        # Calculate Mean Average Precision
        running_rel = 0
        precision_sum = 0.0
        for rank, is_rel in enumerate(relevance_flags, start=1):
            if is_rel == 1:
                running_rel += 1
                precision_at_k = running_rel / rank
                precision_sum += precision_at_k

        map_score = precision_sum / total_relevant
        return round(min(map_score, 1.0), 3)

    def evaluate_context_recall(self, ground_truth: str, contexts: List[str], key_facts: List[str] = None) -> float:
        """
        Context Recall: Proportion of ground-truth statements/facts found in retrieved contexts.
        """
        if not contexts:
            return 0.0

        combined_contexts = " ".join(contexts)

        # If key facts are explicitly provided, evaluate coverage of each fact
        if key_facts and len(key_facts) > 0:
            recalled_count = 0
            for fact in key_facts:
                fact_kws = self._extract_keywords(fact)
                if not fact_kws:
                    continue
                ctx_kws = self._extract_keywords(combined_contexts)
                overlap = len(fact_kws.intersection(ctx_kws)) / len(fact_kws)
                if overlap >= 0.50:
                    recalled_count += 1
            recall_score = recalled_count / float(len(key_facts))
            return round(recall_score, 3)

        # Otherwise evaluate ground truth sentence coverage
        gt_sentences = self._split_into_claims(ground_truth)
        recalled = 0
        for s in gt_sentences:
            s_kws = self._extract_keywords(s)
            if not s_kws:
                continue
            ctx_kws = self._extract_keywords(combined_contexts)
            overlap = len(s_kws.intersection(ctx_kws)) / len(s_kws)
            if overlap >= 0.50:
                recalled += 1

        recall_score = (recalled / float(len(gt_sentences))) if gt_sentences else 1.0
        return round(recall_score, 3)

    def evaluate_item(
        self,
        test_id: str,
        question: str,
        ground_truth: str,
        generated_answer: str,
        contexts: List[str],
        key_facts: List[str] = None
    ) -> QualityItemResult:
        faithfulness, supported, unsupported = self.evaluate_faithfulness(generated_answer, contexts)
        relevancy = self.evaluate_answer_relevancy(question, generated_answer)
        precision = self.evaluate_context_precision(question, ground_truth, contexts)
        recall = self.evaluate_context_recall(ground_truth, contexts, key_facts)

        hallucination_detected = len(unsupported) > 0

        return QualityItemResult(
            test_id=test_id,
            question=question,
            faithfulness_score=faithfulness,
            answer_relevancy_score=relevancy,
            context_precision_score=precision,
            context_recall_score=recall,
            hallucination_detected=hallucination_detected,
            unsupported_claims=unsupported,
            supported_claims=supported,
            retrieved_contexts=contexts,
            generated_answer=generated_answer
        )

    def summarize(self, items: List[QualityItemResult]) -> QualitySummary:
        if not items:
            return QualitySummary(
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                overall_quality_score=0.0,
                items=[]
            )

        avg_faith = sum(i.faithfulness_score for i in items) / len(items)
        avg_rel = sum(i.answer_relevancy_score for i in items) / len(items)
        avg_prec = sum(i.context_precision_score for i in items) / len(items)
        avg_rec = sum(i.context_recall_score for i in items) / len(items)

        overall = (avg_faith * 0.35) + (avg_rel * 0.25) + (avg_prec * 0.20) + (avg_rec * 0.20)

        return QualitySummary(
            avg_faithfulness=round(avg_faith, 3),
            avg_answer_relevancy=round(avg_rel, 3),
            avg_context_precision=round(avg_prec, 3),
            avg_context_recall=round(avg_rec, 3),
            overall_quality_score=round(overall, 3),
            items=items
        )
