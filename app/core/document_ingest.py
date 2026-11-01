"""
Document Ingestion & Synthetic Benchmark Question Generation.
Parses uploaded documents (PDF, TXT, Markdown), splits into parent-child chunks,
inserts into ChromaDB vector store, and synthesizes diverse benchmark evaluation questions using LLM.
"""

import io
import re
import json
import uuid
from typing import List, Dict, Any, Tuple
import httpx
from pypdf import PdfReader
from app.core.chroma_vector_store import ChromaVectorStore, global_chroma_store
from app.core.rag_strategies import call_llm_generation

def extract_text_from_bytes(filename: str, content_bytes: bytes) -> str:
    """Extracts text from PDF, TXT, or Markdown bytes."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content_bytes))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text).strip()
    else:
        # Default text / markdown decoding
        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return content_bytes.decode("latin-1", errors="ignore")

def chunk_document(text: str, parent_chunk_size: int = 400, child_chunk_size: int = 100) -> List[Dict[str, Any]]:
    """
    Splits text into parent documents (400-500 words) and fine-grained child chunks (80-100 words).
    Maintains parent-child foreign key linkage.
    """
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        sentences = [text]

    parents = []
    current_parent_sentences = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        current_parent_sentences.append(sent)
        current_words += words
        if current_words >= parent_chunk_size:
            parents.append(" ".join(current_parent_sentences))
            current_parent_sentences = []
            current_words = 0

    if current_parent_sentences:
        parents.append(" ".join(current_parent_sentences))

    # For each parent, generate child chunks
    structured_data = []
    for p_idx, p_text in enumerate(parents):
        p_id = f"user_doc_p{p_idx + 1}_{uuid.uuid4().hex[:6]}"
        p_sentences = re.split(r"(?<=[.!?])\s+", p_text)
        
        children = []
        curr_child_sents = []
        curr_child_words = 0

        for sent in p_sentences:
            curr_child_sents.append(sent)
            curr_child_words += len(sent.split())
            if curr_child_words >= child_chunk_size:
                c_text = " ".join(curr_child_sents).strip()
                if c_text:
                    children.append(c_text)
                curr_child_sents = []
                curr_child_words = 0

        if curr_child_sents:
            c_text = " ".join(curr_child_sents).strip()
            if c_text:
                children.append(c_text)

        # Fallback if no child was produced
        if not children and p_text.strip():
            children.append(p_text.strip())

        structured_data.append({
            "parent_id": p_id,
            "full_content": p_text,
            "children": children
        })

    return structured_data

async def generate_benchmark_questions_from_document(doc_title: str, full_text: str) -> List[Dict[str, str]]:
    """
    Uses LLM (Gemini with deterministic fallback) to generate questions covering different aspects
    and testing the 5 RAG strategies (Keyword match, Context expansion, Asymmetric HyDE, Nuanced Rerank).
    """
    truncated_text = full_text[:4000] # Fit in context window
    prompt = (
        f"You are a Senior RAG Evaluation Architect. Analyze the following document and generate 4 diverse benchmark evaluation questions.\n"
        f"Each question must target a specific RAG challenge:\n"
        f"1. An exact entity, number, or acronym question (tests Hybrid BM25)\n"
        f"2. A nuanced multi-sentence factual question (tests Cohere Cross-Encoder Rerank)\n"
        f"3. A broad contextual architectural question requiring multiple surrounding paragraphs (tests Parent-Child expansion)\n"
        f"4. A high-level conceptual/problem-solving question (tests HyDE)\n\n"
        f"Document Title: {doc_title}\n"
        f"Document Content:\n{truncated_text}\n\n"
        f"Return ONLY a valid JSON array of objects with keys: 'question', 'ground_truth', 'target_strategy_focus', 'aspect'.\n"
        f"Example format:\n"
        f'[{{"question": "...", "ground_truth": "...", "target_strategy_focus": "Hybrid BM25", "aspect": "Exact Acronym Match"}}]'
    )

    raw_response = await call_llm_generation(prompt, system_prompt="You generate strict JSON benchmark testsets for RAG evaluation.")
    
    # Parse JSON from response
    parsed = extract_json_array(raw_response)
    if parsed and len(parsed) >= 2:
        return parsed[:4]

    # Deterministic fallback synthesis
    return synthesize_fallback_questions(doc_title, full_text)

def extract_json_array(text: str) -> List[Dict[str, str]]:
    """Extracts JSON array from text even if enclosed in markdown backticks."""
    try:
        # Strip markdown fences
        clean = re.sub(r"```json\s*", "", text)
        clean = re.sub(r"```\s*", "", clean).strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                valid = []
                for item in data:
                    if isinstance(item, dict) and "question" in item and "ground_truth" in item:
                        valid.append({
                            "question": item.get("question", "").strip(),
                            "ground_truth": item.get("ground_truth", "").strip(),
                            "target_strategy_focus": item.get("target_strategy_focus", "General RAG"),
                            "aspect": item.get("aspect", "Factual Grounding")
                        })
                return valid
    except Exception:
        pass
    return []

def synthesize_fallback_questions(doc_title: str, full_text: str) -> List[Dict[str, str]]:
    """Synthesizes high quality benchmark questions from document sentences."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.split()) >= 6]
    if not sentences:
        sentences = [full_text[:150]]

    # Look for technical terms/numbers
    entities = re.findall(r"\b[A-Z0-9-]{3,}\b|\b\d+(?:\.\d+)?%?\b", full_text)

    q1_focus = entities[0] if entities else "key requirements"
    q1 = f"What are the specifications regarding {q1_focus} mentioned in {doc_title}?"
    gt1 = sentences[0] if sentences else full_text[:150]

    q2 = f"How is the system configured or implemented according to the {doc_title} documentation?"
    gt2 = sentences[1] if len(sentences) > 1 else sentences[0]

    q3 = f"Explain the comprehensive architecture and surrounding guarantees detailed in {doc_title}."
    gt3 = " ".join(sentences[:2]) if len(sentences) >= 2 else sentences[0]

    q4 = f"What main operational problem or use case does {doc_title} address?"
    gt4 = sentences[-1] if len(sentences) > 2 else sentences[0]

    return [
        {
            "question": q1,
            "ground_truth": gt1,
            "target_strategy_focus": "Hybrid BM25",
            "aspect": "Exact Entity / Acronym Match"
        },
        {
            "question": q2,
            "ground_truth": gt2,
            "target_strategy_focus": "Cohere Rerank",
            "aspect": "Nuanced Passage Alignment"
        },
        {
            "question": q3,
            "ground_truth": gt3,
            "target_strategy_focus": "Parent-Child",
            "aspect": "Broad Contextual Expansion"
        },
        {
            "question": q4,
            "ground_truth": gt4,
            "target_strategy_focus": "HyDE",
            "aspect": "High-Level Conceptual Retrieval"
        }
    ]

def ingest_document_into_store(
    title: str,
    text: str,
    store: ChromaVectorStore = global_chroma_store
) -> Tuple[int, int]:
    """
    Chunks document and inserts parent documents and child chunks into the ChromaDB vector store.
    Returns (parents_count, chunks_count).
    """
    structured = chunk_document(text)
    total_parents = 0
    total_chunks = 0

    for item in structured:
        p_id = item["parent_id"]
        store.add_parent_document(
            doc_id=p_id,
            title=title,
            full_content=item["full_content"],
            metadata={"source": title}
        )
        total_parents += 1

        for c_idx, c_text in enumerate(item["children"]):
            c_id = f"{p_id}_c{c_idx + 1}"
            store.add_chunk(
                chunk_id=c_id,
                parent_id=p_id,
                content=c_text,
                title=title,
                metadata={"source": title}
            )
            total_chunks += 1

    return total_parents, total_chunks
