"""
High-quality sample resume for demonstration and testing in RAGLens.
"""

SAMPLE_RESUME_TEXT = """
ALEXANDER CHEN
Senior AI & Distributed Systems Engineer
San Francisco, CA | alex.chen@example.com | linkedin.com/in/alexchen-ai | github.com/alexchen-dev

PROFESSIONAL SUMMARY
Results-driven AI Infrastructure and Full-Stack Machine Learning Engineer with 6+ years of experience designing high-throughput RAG systems, scalable vector search pipelines, and microservices on AWS and Kubernetes. Reduced LLM inference latency by 45% and served 20M+ monthly vector queries.

TECHNICAL SKILLS
- Languages: Python, Go, TypeScript, C++, SQL
- Frameworks & Libraries: FastAPI, PyTorch, LangChain, LlamaIndex, Transformers, HuggingFace
- Data & Vector Stores: PostgreSQL, Redis, Pinecone, ChromaDB, Qdrant, Milvus
- Cloud & DevOps: AWS (EKS, Lambda, S3, RDS), Docker, Kubernetes, Terraform, Prometheus
- Concepts: Retrieval-Augmented Generation (RAG), Contextual Query Rewriting, BM25 Hybrid Search, Cross-Encoder Reranking

WORK EXPERIENCE

Lead AI Engineer | SynthAI Systems | San Francisco, CA | 2022 - Present
- Architected and deployed an enterprise Multi-Tenant RAG platform handling 15,000+ internal technical documentation queries daily.
- Built a hybrid retrieval pipeline combining Dense Vector embeddings (OpenAI text-embedding-3-small) and BM25 lexical search with Reciprocal Rank Fusion, improving Context Recall from 71% to 92%.
- Engineered a contextual query reformulation and dynamic reranking layer with cross-encoders, cutting irrelevant retrieval by 65% and saving token expenditures.
- Implemented automated prompt injection defenses and document sanitization guardrails, neutralizing 99.8% of adversarial jailbreak attempts.
- Led a team of 5 engineers across sprint planning, technical design reviews, and automated CI/CD deployment pipelines.

Machine Learning Engineer | NovaScale Analytics | Seattle, WA | 2019 - 2022
- Developed distributed data processing pipelines using Apache Spark and Ray to preprocess 500GB+ daily unstructured log files.
- Fine-tuned BERT and RoBERTa classification models for automated customer support ticket triage, achieving 94.2% F1-score and reducing triage response SLA from 4 hours to 15 minutes.
- Containerized model inference microservices with FastAPI and Docker, orchestrating autoscaling clusters via Kubernetes on AWS EKS.

EDUCATION
Bachelor of Science in Computer Science | University of Washington, Seattle | 2015 - 2019
- Magna Cum Laude, GPA: 3.88/4.0
- Honors Thesis: Fast Approximate Nearest Neighbors for High-Dimensional Sparse Embeddings

CERTIFICATIONS & AWARDS
- AWS Certified Solutions Architect - Professional (2023)
- First Place Winner, Bay Area LLM Hackathon 2023 (Autonomous Multi-Agent Evaluator)
"""
