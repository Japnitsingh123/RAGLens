"""
Enterprise Knowledge Base for RAGLens Benchmarking (MySQL Vector Storage).
Contains parent documents linked to fine-grained child chunks to demonstrate:
- Naive vs Parent-Child context expansion
- Dense vs MySQL Fulltext BM25 keyword matching
- Cohere Cross-Encoder reranking
- HyDE hypothetical embedding retrieval
"""

from typing import List, Dict, Any

SAMPLE_ENTERPRISE_DOCS = [
    {
        "doc_id": "sec_01",
        "title": "Enterprise Data Security & Encryption Architecture",
        "full_content": (
            "Enterprise Data Security Architecture: All sensitive customer data at rest is encrypted using "
            "industry-standard AES-256-GCM authenticated encryption with keys managed in AWS KMS. Keys are automatically "
            "rotated every 90 days. All data in transit across public networks and between internal microservices "
            "is enforced via TLS 1.3 with forward secrecy cipher suites. Strict mutual TLS (mTLS) is mandated for all "
            "service-to-service RPC communications. Hardware Security Modules (HSM) compliant with FIPS 140-3 Level 3 "
            "are deployed for root key generation and cryptographic envelope verification."
        ),
        "chunks": [
            {
                "chunk_id": "sec_01_c1",
                "content": "All sensitive customer data at rest is encrypted using AES-256-GCM authenticated encryption with keys managed in AWS KMS with 90-day automatic rotation."
            },
            {
                "chunk_id": "sec_01_c2",
                "content": "All data in transit across public networks and internal microservices is enforced via TLS 1.3 with forward secrecy cipher suites and mutual TLS (mTLS)."
            },
            {
                "chunk_id": "sec_01_c3",
                "content": "Hardware Security Modules (HSM) compliant with FIPS 140-3 Level 3 are deployed for root key generation and cryptographic envelope verification."
            }
        ]
    },
    {
        "doc_id": "sla_02",
        "title": "Infrastructure High Availability & SLA Guarantees",
        "full_content": (
            "Infrastructure High Availability: The platform guarantees a 99.99% monthly uptime Service Level Agreement (SLA) "
            "for core transactional endpoints. The primary relational database runs on MySQL 8.4 with Group Replication deployed "
            "in an active-passive Multi-AZ configuration with synchronous replication. In the event of primary zone failure, "
            "automated failover is executed by MySQL Router and Orchestrator within 25 seconds, achieving a Recovery Point Objective (RPO) of zero "
            "and a Recovery Time Objective (RTO) under 30 seconds. Point-in-time recovery binlog archives are maintained in S3."
        ),
        "chunks": [
            {
                "chunk_id": "sla_02_c1",
                "content": "The platform guarantees a 99.99% monthly uptime Service Level Agreement (SLA) for core transactional endpoints."
            },
            {
                "chunk_id": "sla_02_c2",
                "content": "The primary relational database runs on MySQL 8.4 with Group Replication deployed in Multi-AZ synchronous replication."
            },
            {
                "chunk_id": "sla_02_c3",
                "content": "Automated failover by MySQL Router and Orchestrator executes within 25 seconds during primary failure, achieving RPO of zero and RTO under 30 seconds."
            }
        ]
    },
    {
        "doc_id": "rate_03",
        "title": "Distributed API Rate Limiting & Token Bucket Policy",
        "full_content": (
            "API Rate Limiting Architecture: Traffic ingress is protected by a distributed sliding-window token bucket algorithm "
            "implemented in Redis cluster. Standard enterprise tiers are allocated 1,000 requests per minute with a burst allowance "
            "of 1,500 requests for up to 10 seconds. When limits are exceeded, API gateways immediately return HTTP 429 Too Many Requests "
            "accompanied by standard RFC 6585 headers: Retry-After, X-RateLimit-Limit, and X-RateLimit-Remaining. Premium dedicated tenants "
            "can scale to 10,000 requests per minute with dedicated VPC peering."
        ),
        "chunks": [
            {
                "chunk_id": "rate_03_c1",
                "content": "Traffic ingress is protected by a distributed sliding-window token bucket algorithm implemented in Redis cluster."
            },
            {
                "chunk_id": "rate_03_c2",
                "content": "Standard enterprise tiers are allocated 1,000 requests per minute with a burst allowance of 1,500 requests for 10 seconds."
            },
            {
                "chunk_id": "rate_03_c3",
                "content": "When limits are exceeded, gateways return HTTP 429 with RFC 6585 headers: Retry-After, X-RateLimit-Limit, and X-RateLimit-Remaining."
            }
        ]
    },
    {
        "doc_id": "vec_04",
        "title": "MySQL 8.4 Vector Indexing & Embedding Storage Architecture",
        "full_content": (
            "MySQL Vector Database Architecture: Vector embeddings are indexed in MySQL 8.4 using native VECTOR(768) columns. "
            "Distance metric calculation utilizes VECTOR_DISTANCE(embedding, query_vec, 'COSINE') combined with FULLTEXT indexes "
            "on content columns for hybrid keyword scoring. Parent documents are normalized into rag_parent_documents with 500 tokens "
            "of rich context, while child chunks in rag_documents store 100 tokens with foreign key references for granular retrieval precision."
        ),
        "chunks": [
            {
                "chunk_id": "vec_04_c1",
                "content": "Vector embeddings are indexed in MySQL 8.4 using native VECTOR(768) columns with cosine distance calculation."
            },
            {
                "chunk_id": "vec_04_c2",
                "content": "Distance calculation uses VECTOR_DISTANCE(..., 'COSINE') combined with MySQL FULLTEXT index for hybrid BM25 scoring."
            },
            {
                "chunk_id": "vec_04_c3",
                "content": "Parent documents in rag_parent_documents store 500 tokens, while child chunks store 100 tokens with foreign key parent_id."
            }
        ]
    }
]

BENCHMARK_PRESET_QUERIES = [
    {
        "title": "Exact Acronym & Cipher Match (Tests Hybrid BM25)",
        "query": "What encryption standard is used for data at rest and how are keys rotated?",
        "ground_truth": "Data at rest is encrypted using AES-256-GCM with keys managed in AWS KMS and automatically rotated every 90 days."
    },
    {
        "title": "High Availability & Failover SLAs (Tests Cohere Rerank)",
        "query": "What is the database failover mechanism and the guaranteed RTO and RPO?",
        "ground_truth": "Automated failover is handled by MySQL Router and Orchestrator within 25 seconds, providing an RPO of zero and RTO under 30 seconds."
    },
    {
        "title": "Broader Context Extraction (Tests Parent-Child Expansion)",
        "query": "Explain how vector indexing is configured in MySQL including VECTOR(768), fulltext indexing, and parent-child token sizes.",
        "ground_truth": "MySQL 8.4 uses VECTOR(768) columns with VECTOR_DISTANCE cosine metric and FULLTEXT indexing, storing 500 tokens in parent chunks and 100 tokens in child chunks."
    },
    {
        "title": "Conceptual Asymmetric Query (Tests HyDE)",
        "query": "How does the system prevent denial of service and enforce request throttling?",
        "ground_truth": "The platform enforces a distributed sliding-window token bucket algorithm in Redis with 1,000 req/min and HTTP 429 responses."
    }
]
