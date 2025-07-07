#!/usr/bin/env python3
"""Test the chatbot with some queries"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

from simple_chatbot import SimpleRAGChatbot, create_sample_knowledge_base
from rag_pipeline import SimpleRAGPipeline

# Initialize RAG system
print("📚 Setting up knowledge base...")
rag = SimpleRAGPipeline(
    chunk_size=300,
    chunk_overlap=50,
    embedding_dim=150,
    top_k=3
)

# Load knowledge base
documents = create_sample_knowledge_base()
rag.ingest_documents(documents)
print(f"✅ Loaded {len(documents)} documents about RAG")

# Create chatbot
chatbot = SimpleRAGChatbot(rag)

# Test queries
test_queries = [
    "What is RAG?",
    "How does RAG work?",
    "What are the benefits of RAG?",
    "Tell me about vector embeddings",
    "What's Bob's last name?"  # Should not find anything
]

print("\n🧪 Testing chatbot with various queries:\n")

for query in test_queries:
    print(f"You: {query}")
    response = chatbot.chat(query)
    print(f"Bot: {response}\n")
    print("-" * 60 + "\n")