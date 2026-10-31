import re
import json
import os
from typing import List, Dict, Any, Optional
import httpx

class TestsetGenerator:
    """
    Intelligent test question & ground truth synthesizer for RAG evaluation.
    Supports:
    1. Live LLM Generation (Gemini, OpenAI, Groq) via API key
    2. Deep Document-Specific Semantic Analyzer (zero-API fallback that dynamically
       analyzes the exact contents, metrics, entities, and domain of any uploaded document).
    """

    @classmethod
    async def generate_with_llm(
        cls,
        text: str,
        api_key: Optional[str] = None,
        provider: str = "gemini",
        model: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Uses an LLM (Gemini, OpenAI, or Groq) to analyze the uploaded document
        and generate 5 multi-scenario evaluation test cases with ground truths.
        """
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            return None

        prompt = f"""
You are an expert evaluator of Retrieval-Augmented Generation (RAG) systems.
Analyze the following document and generate 5 comprehensive evaluation test cases.
Each test case must test a different aspect or scenario of the RAG pipeline:
1. 'direct_fact': A direct factual question testing basic retrieval of an exact statement or detail.
2. 'quantifiable_impact': A question testing retrieval of specific numbers, metrics, dates, or measurable outcomes.
3. 'multi_hop_synthesis': A reasoning question that requires combining facts from two different parts of the document.
4. 'technical_deep_dive': A detailed question about specific tools, architectures, or implementations mentioned.
5. 'hallucination_trap': A plausible question relevant to this domain about something NOT mentioned in the document. The ground truth must explicitly state that this information is NOT found in the document, testing whether the RAG model hallucinates.

Document:
\"\"\"
{text[:4000]}
\"\"\"

Respond ONLY with a valid JSON array of 5 objects with keys:
- "id": string (e.g. "q1_fact", "q2_impact", "q3_multihop", "q4_technical", "q5_trap")
- "question": string
- "ground_truth": string
- "category": string (one of the 5 scenario types)
- "key_facts": array of short key strings
"""

        try:
            # 1. Gemini with Multi-Model Fallback
            if provider.lower() == "gemini" or key.startswith("AIza") or key.startswith("AQ."):
                models_to_try = [model] if model else ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for m_name in models_to_try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key}"
                        resp = await client.post(url, json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
                        })
                        if resp.status_code == 200:
                            data = resp.json()
                            raw_json = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            # Clean backticks if any
                            if raw_json.startswith("```"):
                                raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json)
                                raw_json = re.sub(r'\s*```$', '', raw_json)
                            parsed = json.loads(raw_json)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                return parsed
                        elif resp.status_code == 429:
                            print(f"[TestsetGenerator] Model {m_name} hit 429 quota, trying next model...")
                            continue

            # 2. OpenAI / Groq compatible
            base_url = "https://api.groq.com/openai/v1/chat/completions" if (provider == "groq" or key.startswith("gsk_")) else "https://api.openai.com/v1/chat/completions"
            model_name = model or ("llama-3.1-8b-instant" if "groq" in base_url else "gpt-4o-mini")

            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    base_url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
        except Exception as e:
            print(f"[TestsetGenerator] LLM call error: {e}")

        return None

    @classmethod
    def generate_from_text(
        cls,
        text: str,
        max_questions: int = 5,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates document-specific evaluation test questions covering all aspects & scenarios:
        1. Factual extraction
        2. Quantifiable metrics & results
        3. Multi-hop / Technical specifics
        4. Roles, dates, or credentials
        5. Negative / Hallucination trap tailored to the document's domain
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if len(s.strip()) > 25]

        # Extract title / entity subject from first few lines
        subject_name = "the candidate"
        for l in lines[:3]:
            if 3 < len(l.split()) < 6 and not any(kw in l.lower() for kw in ["resume", "curriculum", "http", "@", "profile", "summary"]):
                subject_name = l.strip()
                break

        test_cases: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Scenario 1: Quantifiable Metrics & Measurable Impact
        # -------------------------------------------------------------
        metric_regex = r'(\d+%\s*|\$\d+[\d,]*|\b\d+(?:\.\d+)?(?:k|m|b)\+?\b|\b\d+x\b|\b\d{2,}\+?\s*(?:users|queries|requests|days|months|years|engineers|services))'
        sentences_with_metrics = [s for s in sentences if re.search(metric_regex, s, re.IGNORECASE)]

        if sentences_with_metrics:
            chosen = sentences_with_metrics[0]
            # Extract the actual number/metric
            found_metric = re.findall(metric_regex, chosen, re.IGNORECASE)[0]
            test_cases.append({
                "id": "q1_metric_impact",
                "question": f"What measurable results, throughput numbers, or quantifiable metrics are reported regarding '{chosen[:60]}...'?",
                "ground_truth": chosen,
                "category": "quantifiable_impact",
                "key_facts": [found_metric] + [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', chosen)[:3]]
            })

        # -------------------------------------------------------------
        # Scenario 2: Technical Tools, Systems, Frameworks & Implementation
        # -------------------------------------------------------------
        # Find lines with bullet points or technology stacks
        tech_pattern = r'\b(python|javascript|typescript|c\+\+|java|golang|rust|react|fastapi|django|flask|docker|kubernetes|aws|azure|gcp|postgresql|redis|mongodb|qdrant|chroma|pinecone|kafka|spark|ray|rag|llm|langchain|pytorch|tensorflow|transformers)\b'
        tech_matches = set(re.findall(tech_pattern, text, re.IGNORECASE))

        tech_sentences = [s for s in sentences if len(set(re.findall(tech_pattern, s, re.IGNORECASE))) >= 2]
        if tech_sentences:
            chosen_tech = tech_sentences[0]
            matched_terms = list(set(re.findall(tech_pattern, chosen_tech, re.IGNORECASE)))
            tech_str = ", ".join([t.upper() if len(t) <= 3 else t.title() for t in matched_terms[:4]])
            test_cases.append({
                "id": "q2_technical_deepdive",
                "question": f"How are technologies such as {tech_str} utilized in the implementations described?",
                "ground_truth": chosen_tech,
                "category": "technical_deep_dive",
                "key_facts": matched_terms[:4]
            })
        elif tech_matches:
            top_techs = list(tech_matches)[:5]
            tech_str = ", ".join([t.title() for t in top_techs])
            test_cases.append({
                "id": "q2_technical_stack",
                "question": "What primary technologies, programming languages, and infrastructure tools are specified in the document?",
                "ground_truth": f"The document specifies experience with {tech_str}.",
                "category": "technical_skills",
                "key_facts": top_techs
            })

        # -------------------------------------------------------------
        # Scenario 3: Direct Factual Extraction (Roles, Timeline, Company)
        # -------------------------------------------------------------
        company_role_sentences = [s for s in sentences if re.search(r'\b(lead|engineer|architect|manager|director|analyst|developer|consultant)\b', s, re.IGNORECASE)]
        if company_role_sentences:
            # Pick a sentence that wasn't already used
            usable = [s for s in company_role_sentences if not any(s in tc["ground_truth"] for tc in test_cases)]
            if usable:
                chosen_role = usable[0]
                test_cases.append({
                    "id": "q3_direct_fact",
                    "question": f"What specific architectural responsibilities or role achievements are described in: '{chosen_role[:60]}...'?",
                    "ground_truth": chosen_role,
                    "category": "direct_fact",
                    "key_facts": [w for w in re.findall(r'\b[a-zA-Z]{5,}\b', chosen_role)[:4]]
                })

        # -------------------------------------------------------------
        # Scenario 4: Education, Certifications, or Credentials
        # -------------------------------------------------------------
        credential_sentences = [s for s in sentences if re.search(r'\b(bachelor|master|phd|degree|university|college|certified|certification|gpa|magna|cum laude)\b', s, re.IGNORECASE)]
        if credential_sentences:
            chosen_cred = credential_sentences[0]
            test_cases.append({
                "id": "q4_education_credentials",
                "question": "What academic degrees, universities, certifications, or formal credentials are documented?",
                "ground_truth": chosen_cred,
                "category": "education_credentials",
                "key_facts": [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', chosen_cred)[:4]]
            })

        # -------------------------------------------------------------
        # Scenario 5: Document-Aware Negative Hallucination Trap
        # -------------------------------------------------------------
        # Select plausible technologies or topics in this domain that are NOT present in this specific document
        potential_hallucination_topics = [
            ("Salesforce Apex CRM Integration", "salesforce"),
            ("Snowflake Data Warehousing", "snowflake"),
            ("Apache Cassandra NoSQL Cluster", "cassandra"),
            ("Flutter & Dart Mobile Apps", "flutter"),
            ("Kubernetes Certified Administrator (CKA)", "cka"),
            ("GCP Professional Cloud Architect Certification", "gcp professional"),
            ("Blockchain Ethereum Smart Contracts", "ethereum")
        ]

        absent_topic = "Salesforce Apex CRM Integration"
        for topic, keyword in potential_hallucination_topics:
            if keyword not in text.lower():
                absent_topic = topic
                break

        test_cases.append({
            "id": "q5_hallucination_trap",
            "question": f"Does the document mention any hands-on experience or production deployments using {absent_topic}?",
            "ground_truth": f"The document contains no mention or evidence of experience with {absent_topic}. The RAG model should clearly state this is not found rather than hallucinating.",
            "category": "hallucination_trap",
            "key_facts": ["not mentioned", "no mention", "not found"]
        })

        # Fill remaining slots up to max_questions from unique substantial sentences
        idx = len(test_cases) + 1
        for s in sentences:
            if len(test_cases) >= max_questions:
                break
            if len(s) > 45 and not any(s in tc["ground_truth"] for tc in test_cases):
                test_cases.append({
                    "id": f"q{idx}_synthesis",
                    "question": f"According to the text, what specific context is provided regarding '{s[:50]}...'?",
                    "ground_truth": s,
                    "category": "multi_hop_synthesis",
                    "key_facts": [w for w in re.findall(r'\b[a-zA-Z]{5,}\b', s)[:3]]
                })
                idx += 1

        return test_cases[:max_questions]
