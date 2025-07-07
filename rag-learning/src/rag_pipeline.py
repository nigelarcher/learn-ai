#!/usr/bin/env python3
"""
Complete RAG Pipeline: Bringing it all together
Shows how retrieval and generation work together
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from simple_embedder import SimpleEmbedder
from document_processor import DocumentProcessor, DocumentChunk
from simple_vector_store import SimpleVectorStore


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[Dict]
    confidence: float
    retrieved_chunks: List[Tuple[str, float]]  # (content, similarity)


class SimpleRAGPipeline:
    """
    A complete RAG pipeline that:
    1. Ingests documents
    2. Processes and stores them
    3. Retrieves relevant chunks for queries
    4. Generates answers using retrieved context
    """
    
    def __init__(self,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 embedding_dim: int = 384,
                 top_k: int = 3):
        """Initialize RAG pipeline components."""
        self.embedder = SimpleEmbedder(embedding_dim=embedding_dim)
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = SimpleVectorStore(embedding_dim=embedding_dim)
        self.top_k = top_k
        
        # Track ingested documents
        self.document_count = 0
        self.chunk_count = 0
        
    def ingest_documents(self, documents: List[Dict]):
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
        """
        print(f"\n📥 Ingesting {len(documents)} documents...")
        
        all_chunks = []
        
        for doc in documents:
            # Process document into chunks
            source = doc.get('metadata', {}).get('source', f'doc_{self.document_count}')
            chunks = self.processor.process_document(
                text=doc['content'],
                source=source,
                method='sentence'
            )
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(doc.get('metadata', {}))
                
            all_chunks.extend(chunks)
            self.document_count += 1
            
        # Generate embeddings for all chunks
        print(f"🔢 Generating embeddings for {len(all_chunks)} chunks...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed_text(chunk_texts)
        
        # Store in vector database
        self.vector_store.add_documents(
            contents=chunk_texts,
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in all_chunks],
            chunk_ids=[chunk.chunk_id for chunk in all_chunks]
        )
        
        self.chunk_count += len(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"   Total chunks in store: {self.chunk_count}")
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve relevant chunks for a query.
        
        Returns:
            List of (chunk_content, similarity_score, metadata) tuples
        """
        if top_k is None:
            top_k = self.top_k
            
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)[0]
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=0.1  # Lower threshold for simple embedder
        )
        
        # Format results
        retrieved = []
        for doc, similarity in results:
            retrieved.append((doc.content, similarity, doc.metadata))
            
        return retrieved
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_chunks: List[Tuple[str, float, Dict]]) -> str:
        """
        Generate an answer using retrieved context.
        
        In a real system, this would call an LLM API (OpenAI, Anthropic, etc.)
        For demo purposes, we'll create a simple template-based response.
        """
        if not retrieved_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, (chunk, similarity, metadata) in enumerate(retrieved_chunks):
            context_parts.append(f"[{i+1}] {chunk}")
            sources.append(metadata.get('source', 'Unknown'))
            
        context = "\n\n".join(context_parts)
        
        # In production, you would send this to an LLM:
        # prompt = f"""Based on the following context, answer the question.
        # 
        # Context:
        # {context}
        # 
        # Question: {query}
        # 
        # Answer:"""
        # 
        # response = llm.generate(prompt)
        
        # For demo, we'll create a structured response
        answer = self._simple_answer_generation(query, retrieved_chunks)
        
        return answer
    
    def _simple_answer_generation(self, query: str, chunks: List[Tuple[str, float, Dict]]) -> str:
        """Simple rule-based answer generation for demo."""
        # Get the most relevant chunk (highest similarity)
        top_chunk = chunks[0][0]
        top_similarity = chunks[0][1]
        
        # For very high similarity, use the chunk directly
        if top_similarity > 0.5:
            # High confidence - the chunk is very relevant
            answer = top_chunk
            if len(answer) > 400:
                answer = answer[:400] + "..."
        else:
            # Lower confidence - extract most relevant part
            query_lower = query.lower()
            
            # Try to find the most relevant sentence
            sentences = [s.strip() for s in top_chunk.split('.') if s.strip()]
            
            # Find sentence with most query terms
            best_sentence = ""
            best_score = 0
            
            query_words = set(query_lower.split())
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in query_words if word in sentence_lower)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            if best_sentence:
                answer = best_sentence + "."
            else:
                # Fallback to first part of chunk
                answer = top_chunk[:300] + "..." if len(top_chunk) > 300 else top_chunk
            
            # Add context from other chunks if relevant
            if len(chunks) > 1 and chunks[1][1] > 0.2:
                additional = chunks[1][0][:200]
                answer += f" Additionally: {additional}..."
        
        # Add source attribution
        sources = list(set(chunk[2].get('source', 'Unknown') for chunk in chunks[:2]))
        answer += f"\n\nSources: {', '.join(sources)}"
        
        return answer
    
    def query(self, question: str) -> RAGResponse:
        """
        Complete RAG query: retrieve + generate.
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer and metadata
        """
        print(f"\n❓ Query: {question}")
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(question)
        
        if not retrieved:
            return RAGResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                sources=[],
                confidence=0.0,
                retrieved_chunks=[]
            )
        
        # Generate answer
        answer = self.generate_answer(question, retrieved)
        
        # Calculate confidence (average of top similarities)
        avg_similarity = sum(chunk[1] for chunk in retrieved[:3]) / min(3, len(retrieved))
        
        # Prepare response
        response = RAGResponse(
            answer=answer,
            sources=[{
                'source': chunk[2].get('source', 'Unknown'),
                'title': chunk[2].get('title', 'Untitled')
            } for chunk in retrieved],
            confidence=avg_similarity,
            retrieved_chunks=[(chunk[0][:100] + "...", chunk[1]) for chunk in retrieved]
        )
        
        return response
    
    def save_index(self, filepath: str):
        """Save the vector store index."""
        self.vector_store.save(filepath)
        print(f"💾 Saved RAG index to {filepath}")
        
    def load_index(self, filepath: str):
        """Load a previously saved index."""
        self.vector_store.load(filepath)
        self.chunk_count = len(self.vector_store.documents)
        print(f"📂 Loaded RAG index with {self.chunk_count} chunks")


def demonstrate_rag_pipeline():
    """Complete demonstration of RAG pipeline."""
    print("🚀 Complete RAG Pipeline Demonstration")
    print("=" * 60)
    
    # Create RAG pipeline
    rag = SimpleRAGPipeline(
        chunk_size=300,
        chunk_overlap=50,
        embedding_dim=100,  # Smaller for demo
        top_k=3
    )
    
    # Sample documents about AI
    documents = [
        {
            "content": """Machine learning is a method of data analysis that automates analytical 
            model building. It is a branch of artificial intelligence based on the idea that systems 
            can learn from data, identify patterns and make decisions with minimal human intervention. 
            Machine learning algorithms build a model based on sample data, known as training data, 
            in order to make predictions or decisions without being explicitly programmed to do so.""",
            "metadata": {"source": "ml_basics.txt", "title": "Machine Learning Basics"}
        },
        {
            "content": """Natural Language Processing (NLP) is a branch of artificial intelligence 
            that helps computers understand, interpret and manipulate human language. NLP draws from 
            many disciplines, including computer science and computational linguistics, in its pursuit 
            to fill the gap between human communication and computer understanding. Common NLP tasks 
            include text classification, named entity recognition, and sentiment analysis.""",
            "metadata": {"source": "nlp_guide.txt", "title": "NLP Overview"}
        },
        {
            "content": """Deep learning is part of a broader family of machine learning methods based 
            on artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, 
            deep belief networks, recurrent neural networks and convolutional neural networks have been 
            applied to fields including computer vision, speech recognition, and natural language processing.""",
            "metadata": {"source": "dl_intro.txt", "title": "Deep Learning Introduction"}
        }
    ]
    
    # Ingest documents
    rag.ingest_documents(documents)
    
    # Test queries
    print("\n🔍 Testing RAG Queries")
    print("=" * 60)
    
    test_queries = [
        "What is machine learning?",
        "How does NLP work?",
        "What are the applications of deep learning?",
        "What is the relationship between AI and machine learning?"
    ]
    
    for query in test_queries:
        response = rag.query(query)
        
        print(f"\nQ: {query}")
        print(f"A: {response.answer}")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
        
        # Show what was retrieved
        print("📄 Retrieved context:")
        for i, (chunk_preview, similarity) in enumerate(response.retrieved_chunks):
            print(f"  [{i+1}] ({similarity:.2f}) {chunk_preview}")
        
        print("-" * 60)
    
    # Save the index
    rag.save_index("data/rag_demo_index.json")
    
    print("\n✅ RAG pipeline demonstration complete!")


if __name__ == "__main__":
    demonstrate_rag_pipeline()