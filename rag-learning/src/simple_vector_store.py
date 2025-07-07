#!/usr/bin/env python3
"""
Simple Vector Store: Store and search document embeddings
Demonstrates how vector databases work for RAG
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class StoredDocument:
    """Document stored in vector database."""
    chunk_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, any]
    
    def to_dict(self):
        """Convert to dictionary (for storage)."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            chunk_id=data['chunk_id'],
            content=data['content'],
            embedding=np.array(data['embedding']),
            metadata=data['metadata']
        )


class SimpleVectorStore:
    """
    A simple in-memory vector store for RAG.
    In production, you'd use Pinecone, Weaviate, or ChromaDB.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize vector store."""
        self.embedding_dim = embedding_dim
        self.documents: List[StoredDocument] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        
    def add_documents(self, 
                     contents: List[str], 
                     embeddings: np.ndarray,
                     metadatas: List[Dict] = None,
                     chunk_ids: List[str] = None):
        """Add documents to the store."""
        if metadatas is None:
            metadatas = [{}] * len(contents)
        if chunk_ids is None:
            chunk_ids = [f"doc_{i}" for i in range(len(contents))]
            
        for i, (content, embedding, metadata, chunk_id) in enumerate(
            zip(contents, embeddings, metadatas, chunk_ids)
        ):
            doc = StoredDocument(
                chunk_id=chunk_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            self.documents.append(doc)
        
        # Rebuild embeddings matrix for fast search
        self._rebuild_embeddings_matrix()
        
    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix for efficient search."""
        if not self.documents:
            self.embeddings_matrix = None
            return
            
        self.embeddings_matrix = np.array([doc.embedding for doc in self.documents])
        
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 5,
               min_similarity: float = 0.0) -> List[Tuple[StoredDocument, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.documents or self.embeddings_matrix is None:
            return []
        
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize stored embeddings
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings_matrix / (norms + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(normalized_embeddings, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by minimum similarity and return results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= min_similarity:
                results.append((self.documents[idx], similarity))
                
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk."""
        data = {
            'embedding_dim': self.embedding_dim,
            'documents': [doc.to_dict() for doc in self.documents]
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, filepath: str):
        """Load vector store from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.embedding_dim = data['embedding_dim']
        self.documents = [StoredDocument.from_dict(doc) for doc in data['documents']]
        self._rebuild_embeddings_matrix()
        
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'memory_size_mb': self.embeddings_matrix.nbytes / 1024 / 1024 if self.embeddings_matrix is not None else 0,
            'unique_sources': len(set(doc.metadata.get('source', '') for doc in self.documents))
        }


# Production-ready vector store using ChromaDB
class ChromaVectorStore:
    """Production vector store using ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_docs"):
        """Initialize ChromaDB store."""
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            print("ChromaDB not installed. Using simple vector store.")
            self.client = None
            
    def add_documents(self, contents: List[str], embeddings: np.ndarray, 
                     metadatas: List[Dict], chunk_ids: List[str]):
        """Add to ChromaDB."""
        if not self.client:
            return
            
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search ChromaDB."""
        if not self.client:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        output = []
        for i in range(len(results['documents'][0])):
            doc = StoredDocument(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                embedding=None,  # ChromaDB doesn't return embeddings
                metadata=results['metadatas'][0][i]
            )
            similarity = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            output.append((doc, similarity))
            
        return output


def demonstrate_vector_store():
    """Show how vector stores work for RAG."""
    print("🗄️ Vector Store Demonstration")
    print("=" * 60)
    
    from simple_embedder import SimpleEmbedder
    
    # Create embedder and vector store
    embedder = SimpleEmbedder(embedding_dim=100)
    vector_store = SimpleVectorStore(embedding_dim=100)
    
    # Sample documents about AI topics
    documents = [
        {
            "content": "Machine learning is a subset of AI that enables systems to learn from data.",
            "metadata": {"source": "ai_basics.txt", "topic": "ML"}
        },
        {
            "content": "Deep learning uses neural networks with multiple layers to process complex patterns.",
            "metadata": {"source": "ai_basics.txt", "topic": "DL"}
        },
        {
            "content": "Natural language processing helps computers understand human language.",
            "metadata": {"source": "nlp_guide.txt", "topic": "NLP"}
        },
        {
            "content": "Computer vision enables machines to interpret and analyze visual information from images.",
            "metadata": {"source": "cv_intro.txt", "topic": "CV"}
        },
        {
            "content": "Reinforcement learning trains agents through rewards and penalties in an environment.",
            "metadata": {"source": "rl_basics.txt", "topic": "RL"}
        }
    ]
    
    # Process documents
    print("📝 Adding documents to vector store...")
    contents = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    # Generate embeddings
    embeddings = embedder.embed_text(contents)
    
    # Add to store
    vector_store.add_documents(
        contents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
        chunk_ids=[f"chunk_{i}" for i in range(len(documents))]
    )
    
    print(f"Added {len(documents)} documents to store")
    print(f"Store stats: {vector_store.get_stats()}")
    
    # Search examples
    print("\n🔍 Search Examples:")
    print("-" * 40)
    
    queries = [
        "How do neural networks work?",
        "What is NLP?",
        "Tell me about computer vision",
        "How do machines learn from experience?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Get query embedding
        query_embedding = embedder.embed_text(query)[0]
        
        # Search
        results = vector_store.search(query_embedding, top_k=3, min_similarity=0.3)
        
        print("Top results:")
        for doc, similarity in results:
            print(f"  [{similarity:.3f}] {doc.content[:60]}...")
            print(f"         Source: {doc.metadata.get('source', 'unknown')}")
    
    # Save and load demonstration
    print("\n💾 Save/Load Demonstration:")
    print("-" * 40)
    
    # Save
    save_path = "data/vector_store_demo.json"
    vector_store.save(save_path)
    print(f"Saved vector store to {save_path}")
    
    # Create new store and load
    new_store = SimpleVectorStore()
    new_store.load(save_path)
    print(f"Loaded store with {len(new_store.documents)} documents")
    
    # Verify search still works
    test_query = "machine learning"
    test_embedding = embedder.embed_text(test_query)[0]
    results = new_store.search(test_embedding, top_k=1)
    print(f"\nTest search for '{test_query}':")
    print(f"  Found: {results[0][0].content[:50]}...")


if __name__ == "__main__":
    demonstrate_vector_store()