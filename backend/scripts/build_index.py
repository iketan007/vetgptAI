"""
Standalone script to (re)build the FAISS vector index from the
markdown files in data/knowledge_base/.

Usage:
    cd backend
    python scripts/build_index.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import build_index

if __name__ == "__main__":
    build_index()
