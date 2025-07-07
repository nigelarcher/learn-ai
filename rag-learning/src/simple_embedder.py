#!/usr/bin/env python3
"""
Simple Embedder: Convert text to vector embeddings
This demonstrates how text becomes searchable numbers
"""

import numpy as np
from typing import List, Union
import hashlib


class SimpleEmbedder:
    """
    A basic embedder that creates vector representations of text.
    In production, you'd use OpenAI embeddings or sentence-transformers.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize with embedding dimension."""
        self.embedding_dim = embedding_dim
        self.vocabulary = {}  # Word to index mapping
        self.word_vectors = {}  # Pretend word embeddings
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on spaces and lowercase."""
        # Remove punctuation and lowercase
        text = text.lower()
        for punct in ".,!?;:":
            text = text.replace(punct, " ")
        return [word for word in text.split() if word]
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """
        Get a consistent vector for a word.
        In reality, this would come from Word2Vec, GloVe, etc.
        """
        if word not in self.word_vectors:
            # Create a deterministic "random" vector based on word hash
            seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            # Generate vector with some semantic properties
            base_vector = np.random.randn(self.embedding_dim) * 0.1
            
            # Add some "semantic" patterns (simplified)
            if any(tech in word for tech in ['ai', 'computer', 'software', 'data']):
                base_vector[0:50] += 0.3  # Tech cluster
            if any(fin in word for fin in ['money', 'price', 'cost', 'dollar']):
                base_vector[50:100] += 0.3  # Finance cluster
            if any(med in word for med in ['health', 'medical', 'doctor', 'patient']):
                base_vector[100:150] += 0.3  # Medical cluster
                
            self.word_vectors[word] = base_vector
            
        return self.word_vectors[word]
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to embeddings.
        Returns array of shape (n_texts, embedding_dim).
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        embeddings = []
        
        for t in texts:
            tokens = self._tokenize(t)
            if not tokens:
                # Empty text gets zero vector
                embeddings.append(np.zeros(self.embedding_dim))
                continue
                
            # Average word vectors (simple approach)
            word_vecs = [self._get_word_vector(token) for token in tokens]
            text_embedding = np.mean(word_vecs, axis=0)
            
            # Normalize to unit length
            norm = np.linalg.norm(text_embedding)
            if norm > 0:
                text_embedding = text_embedding / norm
                
            embeddings.append(text_embedding)
            
        return np.array(embeddings)
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)


# Production-ready embedder using OpenAI (requires API key)
class OpenAIEmbedder:
    """Production embedder using OpenAI's API."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """Initialize with OpenAI API."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.embedding_dim = 1536 if "ada" in model else 384
        except ImportError:
            print("OpenAI not installed. Using simple embedder instead.")
            self.client = None
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings from OpenAI."""
        if not self.client:
            return SimpleEmbedder().embed_text(text)
            
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        embeddings = [np.array(item.embedding) for item in response.data]
        return np.array(embeddings)


def demonstrate_embeddings():
    """Show how embeddings work with examples."""
    print("🧮 Demonstrating Text Embeddings")
    print("=" * 60)
    
    # Create embedder
    embedder = SimpleEmbedder(embedding_dim=100)  # Smaller for visualization
    
    # Example texts
    texts = [
        "Artificial intelligence and machine learning are transforming technology",
        "AI and ML are revolutionizing tech",  # Similar to first
        "The stock market saw significant gains today",  # Different topic
        "Machine learning algorithms process data efficiently",  # Related to first
        "Pizza is my favorite food"  # Completely different
    ]
    
    print("\n📝 Example Texts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    
    # Generate embeddings
    print("\n🔢 Generating Embeddings...")
    embeddings = embedder.embed_text(texts)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First text embedding (first 10 values): {embeddings[0][:10]}")
    
    # Compute similarities
    print("\n📊 Similarity Matrix:")
    print("(How similar each text is to every other text)")
    print("\n     ", end="")
    for i in range(len(texts)):
        print(f"  T{i+1}  ", end="")
    print()
    
    for i in range(len(texts)):
        print(f"T{i+1}:  ", end="")
        for j in range(len(texts)):
            sim = embedder.compute_similarity(embeddings[i], embeddings[j])
            print(f"{sim:5.2f} ", end="")
        print()
    
    print("\n💡 Insights:")
    print("- T1 and T2 are very similar (both about AI/ML)")
    print("- T1 and T4 are related (both mention ML)")
    print("- T3 (stock market) is different from AI texts")
    print("- T5 (pizza) is different from everything")
    
    # Find most similar to a query
    print("\n🔍 Query Example:")
    query = "How does artificial intelligence work?"
    query_embedding = embedder.embed_text(query)[0]
    
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = embedder.compute_similarity(query_embedding, emb)
        similarities.append((sim, i, texts[i]))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    print(f"Query: '{query}'")
    print("\nMost similar texts:")
    for sim, idx, text in similarities[:3]:
        print(f"  {sim:.3f} - {text[:50]}...")


if __name__ == "__main__":
    demonstrate_embeddings()