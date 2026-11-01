# 🔍 RAGLens: RAG Evaluation, Testing & Benchmarking Platform

> **Comprehensive, production-ready evaluation and benchmarking platform for Retrieval-Augmented Generation (RAG) pipelines.**

---

## 🌟 Overview

**RAGLens** is an end-to-end testing platform designed for developers building RAG systems. It connects to your RAG pipeline (or runs a built-in enterprise documentation pipeline) and systematically tests:
1. **RAG Quality (RAGAS Methodology)**: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
2. **Performance & Latency**: End-to-end response time, retrieval vs. generation latency, token breakdown, and query cost estimation.
3. **Security Red-Teaming**: Prompt injection resistance, system instruction override defenses, and poisoned document detection.
4. **Benchmarking Lab**: Empirical A/B testing comparing Vector Search vs. Hybrid Search (BM25 + Dense RRF), Chunk Sizes (256 vs 512 vs 1024), and Top-K values (2 vs 4 vs 8).

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────┐
                                  │      RAGLens UI / API     │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
      │  RAG Quality Suite  │        │  Performance Suite  │        │   Security Suite    │
      ├─────────────────────┤        ├─────────────────────┤        ├─────────────────────┤
      │ • Faithfulness      │        │ • Response Latency  │        │ • Prompt Injection  │
      │ • Answer Relevancy  │        │ • Retrieval Latency │        │ • System Overrides  │
      │ • Context Precision │        │ • Token Consumption │        │ • Poisoned Chunks   │
      │ • Context Recall    │        │ • System Latency    │        │ • Credential Leaks  │
      └─────────────────────┘        │ • Cost / 1k Queries │        └─────────────────────┘
                                     └─────────────────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │  Benchmarking Lab   │
                                     ├─────────────────────┤
                                     │ • Vector vs. Hybrid │
                                     │ • Chunk Size Tuning │
                                     │ • Top-K Exploration │
                                     │ • Multi-Strategy    │
                                     └─────────────────────┘
```

---

## 🚀 Quickstart: Running Locally (No Docker Required)

1. Open PowerShell or Terminal and navigate to the project directory:
   ```powershell
   cd C:\Users\kaurk\.gemini\antigravity\scratch\raglens
   ```

2. Run the application directly with Python:
   ```powershell
   python main.py
   ```
   *Or simply double-click `run.bat` or run `.\run.ps1` in PowerShell.*

3. Open your browser:
   - **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
   - **Interactive OpenAPI / Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ☁️ Deployment (Vercel & Cloud)

### Deploy on Vercel
RAGLens is pre-configured with `vercel.json` and `api/index.py` for instant Vercel deployment:
1. Push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Initial release of RAGLens"
   git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   git push -u origin main
   ```
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your GitHub repository.
4. (Optional) Add your `GEMINI_API_KEY` under **Environment Variables** in Vercel settings for live LLM evaluations.
5. Click **Deploy**. Vercel will build and serve your app globally on serverless Python!

### Deploy on Render / Railway / VPS (Without Docker)
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `python main.py` (or `uvicorn app.main:app --host 0.0.0.0 --port $PORT`)
3. **Environment**: Python 3.10+


---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:
```bash
python -m pytest tests/test_raglens.py -v
```
All 9 test suites verify unit algorithms, RAGAS calculations, adversarial red-teaming, and API routes.

---

## 📊 Features & Evaluation Metrics

### 1. RAG Quality Testing (RAGAS Methodology)
- **Faithfulness**: Breaks answers into atomic factual claims and verifies whether each is entailed by retrieved chunks. Identifies hallucinations.
- **Answer Relevancy**: Computes semantic vector alignment between user question and generated response.
- **Context Precision**: Mean Average Precision (MAP) measuring whether relevant chunks appear at the top of the context window.
- **Context Recall**: Verifies that all ground truth reference facts are present in retrieved contexts.

### 2. Performance Testing
- **Response Latency**: End-to-end execution timer (ms).
- **Retrieval Latency**: Vector / BM25 index query time.
- **Token Tracking**: Prompt, completion, and total tokens.
- **Cost Calculation**: Configurable for GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and Llama 3 ($/1k queries).

### 3. Security Testing (Red-Teaming)
- **Direct Prompt Injections**: Adversarial probes attempting to bypass system directives.
- **System Instruction Overrides**: Adversarial jailbreak personas, delimiter escaping, and developer mode spoofing.
- **Indirect Document Poisoning**: Evaluates whether malicious instructions planted inside knowledge documents are safely mitigated.

### 4. Benchmarking Lab
- **Vector Search vs. Hybrid Search**: Evaluates BM25 + Dense Reciprocal Rank Fusion ($RRF$) against pure dense vectors.
- **Chunk Size Tuning**: Evaluates 256 vs 512 vs 1024 character chunks.
- **Top-K Tuning**: Balances context recall vs. token costs across $k \in \{2, 4, 8\}$.
