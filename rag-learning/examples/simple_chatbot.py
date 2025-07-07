#!/usr/bin/env python3
"""
Simple RAG Chatbot Interface
A basic chatbot that uses RAG to answer questions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline import SimpleRAGPipeline
import json
from datetime import datetime
from typing import Dict


class SimpleRAGChatbot:
    """A simple chatbot powered by RAG."""
    
    def __init__(self, rag_pipeline: SimpleRAGPipeline):
        """Initialize with a RAG pipeline."""
        self.rag = rag_pipeline
        self.conversation_history = []
        self.session_start = datetime.now()
        
    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get RAG response
        response = self.rag.query(user_input)
        
        # Format response
        if response.confidence < 0.15:
            answer = "I'm not very confident about this, but here's what I found: " + response.answer
        else:
            answer = response.answer
            
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "confidence": response.confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        return answer
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of the conversation."""
        return {
            "session_start": self.session_start.isoformat(),
            "total_messages": len(self.conversation_history),
            "user_questions": len([m for m in self.conversation_history if m["role"] == "user"]),
            "average_confidence": sum(m.get("confidence", 0) for m in self.conversation_history if m["role"] == "assistant") / max(1, len([m for m in self.conversation_history if m["role"] == "assistant"]))
        }
    
    def save_conversation(self, filepath: str):
        """Save conversation history."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                "session": self.get_conversation_summary(),
                "messages": self.conversation_history
            }, f, indent=2)


def create_sample_knowledge_base():
    """Create a sample knowledge base for the chatbot."""
    documents = [
        {
            "content": """RAG stands for Retrieval-Augmented Generation. It's a technique that combines 
            information retrieval with text generation. When you ask a question, RAG first searches 
            through a database of documents to find relevant information, then uses that information 
            to generate an accurate, contextual answer. This approach helps AI systems provide more 
            accurate and up-to-date responses.""",
            "metadata": {"source": "rag_basics.txt", "topic": "RAG"}
        },
        {
            "content": """The key components of a RAG system are: 1) Document Store - where all the 
            knowledge is stored as searchable chunks, 2) Embedder - converts text into numerical 
            vectors for similarity search, 3) Retriever - finds the most relevant documents for a query, 
            and 4) Generator - creates the final answer using the retrieved context. These components 
            work together to provide informed responses.""",
            "metadata": {"source": "rag_architecture.txt", "topic": "RAG Components"}
        },
        {
            "content": """Vector embeddings are numerical representations of text that capture semantic 
            meaning. Similar texts have similar embeddings, which allows us to find relevant documents 
            by computing similarity between embedding vectors. Common embedding models include Word2Vec, 
            GloVe, and modern transformer-based embeddings from models like BERT or sentence-transformers.""",
            "metadata": {"source": "embeddings_guide.txt", "topic": "Embeddings"}
        },
        {
            "content": """Chunking is the process of breaking large documents into smaller, manageable 
            pieces for RAG systems. Good chunking strategies include: splitting by sentences while 
            maintaining context, using sliding windows with overlap, or splitting by paragraphs. The 
            chunk size affects retrieval precision - smaller chunks are more precise but may lack context, 
            while larger chunks provide more context but may be less focused.""",
            "metadata": {"source": "chunking_strategies.txt", "topic": "Chunking"}
        },
        {
            "content": """RAG offers several advantages over traditional LLMs: 1) Always up-to-date 
            information from your documents, 2) Reduced hallucination as answers are grounded in real 
            documents, 3) Transparency with source attribution, 4) Works with private/proprietary data, 
            and 5) More cost-effective than fine-tuning large models. These benefits make RAG ideal 
            for knowledge bases, customer support, and research applications.""",
            "metadata": {"source": "rag_benefits.txt", "topic": "RAG Benefits"}
        }
    ]
    
    return documents


def run_chatbot_demo():
    """Run the chatbot demonstration."""
    print("🤖 RAG Chatbot Demo")
    print("=" * 60)
    print("This chatbot uses RAG to answer questions about RAG itself!")
    print("Commands: 'quit' to exit, 'history' to see conversation")
    print("=" * 60)
    
    # Initialize RAG system
    print("\n📚 Setting up knowledge base...")
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
    
    # Show example questions
    print("\n💡 Example questions you can ask:")
    example_questions = [
        "What is RAG?",
        "How does RAG work?",
        "What are vector embeddings?",
        "What are the benefits of using RAG?",
        "How should I chunk my documents?"
    ]
    for q in example_questions[:3]:
        print(f"  - {q}")
    
    print("\n" + "-" * 60)
    print("Start chatting! I'll answer questions about RAG systems.\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                print("\nBot: Goodbye! Thanks for chatting about RAG!")
                break
                
            elif user_input.lower() == 'history':
                summary = chatbot.get_conversation_summary()
                print(f"\n📊 Conversation Summary:")
                print(f"  Started: {summary['session_start']}")
                print(f"  Questions asked: {summary['user_questions']}")
                print(f"  Average confidence: {summary['average_confidence']:.1%}")
                continue
                
            # Get response
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: Interrupted! Goodbye!")
            break
    
    # Save conversation
    print("\n💾 Saving conversation...")
    chatbot.save_conversation("data/chatbot_conversation.json")
    print("✅ Conversation saved to data/chatbot_conversation.json")
    
    # Show summary
    summary = chatbot.get_conversation_summary()
    print(f"\n📊 Session Summary:")
    print(f"  Total questions: {summary['user_questions']}")
    print(f"  Average confidence: {summary['average_confidence']:.1%}")


if __name__ == "__main__":
    run_chatbot_demo()