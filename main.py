import uvicorn
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure current directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f"  RAGLens: RAG Evaluation, Testing & Benchmarking")
    print(f"  Local Dashboard: http://localhost:{port}")
    print(f"  Interactive Swagger Docs: http://localhost:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="info", reload=True)
