"""
Default sample documentation for the built-in enterprise RAG pipeline.
Covers Cloud Infrastructure, Security, Authentication, API Rate Limits, and SLA policies.
"""

SAMPLE_KNOWLEDGE_BASE = [
    {
        "id": "doc_cloud_arch",
        "title": "Cloud Architecture & Multi-Region Setup",
        "category": "infrastructure",
        "content": """
Our enterprise platform is deployed across three primary AWS regions: us-east-1 (N. Virginia), eu-central-1 (Frankfurt), and ap-southeast-1 (Singapore). 
Each region operates active-active Kubernetes clusters managed by Amazon EKS. Inter-region data replication is achieved using Amazon Aurora Global Database with sub-second replication latency (typically under 800ms).
Traffic distribution is managed via Cloudflare Enterprise with Anycast routing and automated geo-failover. In the event of a regional outage, health-check probes trigger DNS failover within 30 seconds.
All microservices communicate using gRPC with mutual TLS (mTLS) enforced by Istio service mesh. Zero-trust network access (ZTNA) is required for all administrative access.
"""
    },
    {
        "id": "doc_auth_security",
        "title": "Authentication, Authorization & Secret Management",
        "category": "security",
        "content": """
Authentication is powered by OAuth 2.0 and OpenID Connect (OIDC). User access tokens are signed JSON Web Tokens (JWT) using the RS256 algorithm with a lifetime of 15 minutes. 
Refresh tokens are stored in Redis with cryptographic rotation enabled and have a maximum lifetime of 30 days with sliding window expiration.
Multi-Factor Authentication (MFA) via TOTP or FIDO2/WebAuthn hardware keys is mandatory for all internal employees and organization administrators.
Secrets and API credentials must never be committed to source control; they are strictly managed via HashiCorp Vault with dynamic secret leasing and automated 90-day rotation.
Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) are evaluated on every incoming request at the API gateway layer.
"""
    },
    {
        "id": "doc_rate_limits",
        "title": "API Rate Limits and Quotas",
        "category": "api",
        "content": """
Public API endpoints enforce strict rate limiting using a distributed Redis token-bucket algorithm.
Free tier accounts are allocated 60 requests per minute with a burst allowance of 10 requests. 
Professional tier accounts are granted 600 requests per minute with a burst allowance of 50 requests.
Enterprise tier accounts enjoy custom dedicated quotas starting at 5,000 requests per minute with burst allowance of 500 requests and SLA-backed throughput guarantees.
When a client exceeds the allocated rate limit, the API returns HTTP status code 429 (Too Many Requests) along with headers 'Retry-After' and 'X-RateLimit-Reset' indicating the cooldown period.
"""
    },
    {
        "id": "doc_data_retention",
        "title": "Data Retention, Encryption & GDPR Compliance",
        "category": "compliance",
        "content": """
All persistent data is encrypted at rest using AES-256 via AWS KMS customer-managed keys (CMK) with automated annual key rotation. 
All data in transit is encrypted using TLS 1.3 with forward secrecy ciphers.
Customer audit logs and access histories are retained in immutable S3 Glacier Vaults for a minimum of 365 days to comply with SOC2 Type II and ISO 27001 requirements.
In accordance with GDPR Article 17 ('Right to be Forgotten'), customer deletion requests are processed asynchronously and permanently purged from primary datastores within 14 days and from backup snapshots within 30 days.
"""
    },
    {
        "id": "doc_incident_sla",
        "title": "Incident Management, SLA & Disaster Recovery",
        "category": "operations",
        "content": """
The platform provides a 99.99% monthly uptime service level agreement (SLA) for Enterprise tier customers. 
Severity 1 (Critical Outage) incidents require an initial response from the on-call Site Reliability Engineering (SRE) team within 15 minutes, with incident triage updates provided every 30 minutes via the public status page.
Severity 2 (Major Degradation) incidents require an initial response within 60 minutes.
Our Disaster Recovery (DR) plan guarantees a Recovery Point Objective (RPO) of under 5 minutes and a Recovery Time Objective (RTO) of under 30 minutes for catastrophic regional failovers.
Post-mortem root cause analyses (RCA) are published within 72 business hours following any Sev-1 incident.
"""
    }
]
