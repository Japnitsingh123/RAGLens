import os
import sys

# Add root directory to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.main import app as fastapi_app

async def app(scope, receive, send):
    """
    ASGI middleware for Vercel Serverless Python.
    Fixes path rewrites so FastAPI routes (e.g. /, /api/upload-document, /api/benchmark)
    resolve identically both on localhost and on Vercel.
    """
    if scope.get("type") in ("http", "websocket"):
        headers = dict(scope.get("headers", []))
        
        # 1. Check for original client request URI from Vercel headers
        orig_uri = None
        for key in (b"x-matched-path", b"x-forwarded-uri", b"x-invoke-path"):
            if key in headers:
                orig_uri = headers[key].decode("latin-1")
                break

        if orig_uri and orig_uri.startswith("/"):
            # Strip query string from orig_uri if present
            scope["path"] = orig_uri.split("?")[0]
        else:
            path = scope.get("path", "")
            # If path was rewritten with /api/index.py prefix
            if path.startswith("/api/index.py"):
                remainder = path[len("/api/index.py"):]
                scope["path"] = remainder if remainder.startswith("/") else ("/" + remainder)
            elif path.startswith("/api/index"):
                remainder = path[len("/api/index"):]
                scope["path"] = remainder if remainder.startswith("/") else ("/" + remainder)

    await fastapi_app(scope, receive, send)
