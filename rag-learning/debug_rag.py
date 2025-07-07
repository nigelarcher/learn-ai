#!/usr/bin/env python3
"""Debug RAG to see why queries aren't matching"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_embedder import SimpleEmbedder
from rag_pipeline import SimpleRAGPipeline

def debug_rag():
    # Create RAG pipeline
    rag = SimpleRAGPipeline(
        chunk_size=300,
        chunk_overlap=50,
        embedding_dim=150,
        top_k=3
    )
    
    # Simple test document
    documents = [{
        "content": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with text generation.",
        "metadata": {"source": "test.txt"}
    }]
    
    # Ingest
    rag.ingest_documents(documents)
    print(f"Ingested {len(rag.vector_store.documents)} chunks")
    
    # Test queries
    test_queries = [
        "What is RAG?",
        "rag",
        "retrieval augmented generation",
        "completely unrelated query about pizza"
    ]
    
    embedder = SimpleEmbedder(embedding_dim=150)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get query embedding
        query_emb = embedder.embed_text(query)[0]
        
        # Check similarities with all documents
        for i, doc in enumerate(rag.vector_store.documents):
            sim = embedder.compute_similarity(query_emb, doc.embedding)
            print(f"  Doc {i}: similarity = {sim:.4f}")
            print(f"    Content: {doc.content[:50]}...")
        
        # Try retrieval
        results = rag.retrieve(query)
        print(f"  Retrieved: {len(results)} chunks")
        
        if results:
            for chunk, sim, _ in results:
                print(f"    [{sim:.3f}] {chunk[:50]}...")

if __name__ == "__main__":
    debug_rag()