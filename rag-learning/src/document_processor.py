#!/usr/bin/env python3
"""
Document Processor: Chunk and prepare documents for RAG
Shows how to break documents into searchable pieces
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    metadata: Dict[str, any]
    chunk_id: str
    source_doc: str
    chunk_index: int
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk({self.chunk_id}: '{preview}')"


class DocumentProcessor:
    """Process documents into chunks for RAG."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize processor.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks to preserve context
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (production would use NLTK or spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks_by_sentences(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Create chunks trying to respect sentence boundaries.
        Better than splitting mid-sentence.
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_size = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        overlap_sentences.insert(0, sent)
                        if overlap_size >= self.chunk_overlap:
                            break
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        # Convert to DocumentChunk objects
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}_{i}_{chunk_text[:20]}".encode()).hexdigest()[:8]
            doc_chunks.append(DocumentChunk(
                content=chunk_text,
                metadata={
                    "source": source,
                    "chunk_method": "sentence",
                    "chunk_size": self.chunk_size
                },
                chunk_id=chunk_id,
                source_doc=source,
                chunk_index=i
            ))
        
        return doc_chunks
    
    def create_chunks_sliding_window(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Create chunks using sliding window (simple approach).
        Good for when sentence boundaries don't matter.
        """
        text = self.clean_text(text)
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # If we found space in last 20%
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = hashlib.md5(f"{source}_{len(chunks)}_{chunk_text[:20]}".encode()).hexdigest()[:8]
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_method": "sliding_window",
                        "chunk_size": self.chunk_size
                    },
                    chunk_id=chunk_id,
                    source_doc=source,
                    chunk_index=len(chunks)
                ))
            
            # Move forward (with overlap)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks
    
    def process_document(self, text: str, source: str, method: str = "sentence") -> List[DocumentChunk]:
        """
        Process a document into chunks.
        
        Args:
            text: Document text
            source: Source identifier (filename, URL, etc.)
            method: "sentence" or "sliding_window"
        """
        if method == "sentence":
            return self.create_chunks_by_sentences(text, source)
        else:
            return self.create_chunks_sliding_window(text, source)


def demonstrate_chunking():
    """Show different chunking strategies."""
    print("📄 Document Chunking Demonstration")
    print("=" * 60)
    
    # Sample document
    sample_text = """
    Artificial Intelligence (AI) is transforming the world. Machine learning algorithms 
    can now recognize images, understand speech, and even generate text. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers. These networks 
    can learn complex patterns from data.
    
    Natural Language Processing (NLP) is another important area of AI. It enables computers 
    to understand, interpret, and generate human language. Large Language Models like GPT 
    have revolutionized NLP by achieving human-like text generation. These models are trained 
    on vast amounts of text data.
    
    Computer vision is the field that enables machines to see and interpret visual information. 
    It uses deep learning models to analyze images and videos. Applications include facial 
    recognition, object detection, and autonomous vehicles. The combination of computer vision 
    and other AI technologies is creating powerful new applications.
    """
    
    # Create processor with small chunks for demo
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    
    print("\n1️⃣ Sentence-based Chunking")
    print("-" * 40)
    sentence_chunks = processor.process_document(sample_text, "ai_overview.txt", method="sentence")
    
    for chunk in sentence_chunks:
        print(f"\nChunk {chunk.chunk_index} (ID: {chunk.chunk_id}):")
        print(f"Size: {len(chunk.content)} chars")
        print(f"Content: {chunk.content}")
        print("-" * 40)
    
    print(f"\nTotal chunks: {len(sentence_chunks)}")
    
    # Show overlap
    if len(sentence_chunks) > 1:
        print("\n🔄 Overlap Demonstration:")
        print(f"End of Chunk 0: ...{sentence_chunks[0].content[-50:]}")
        print(f"Start of Chunk 1: {sentence_chunks[1].content[:50]}...")
    
    # Different chunk size
    print("\n\n2️⃣ Different Chunk Sizes Comparison")
    print("-" * 40)
    
    for size in [100, 300, 500]:
        processor = DocumentProcessor(chunk_size=size, chunk_overlap=20)
        chunks = processor.process_document(sample_text, f"doc_size_{size}.txt")
        print(f"Chunk size {size}: {len(chunks)} chunks created")
        
    print("\n💡 Insights:")
    print("- Smaller chunks = more precise retrieval but may lose context")
    print("- Larger chunks = more context but less precise retrieval")
    print("- Overlap helps preserve context across chunk boundaries")


if __name__ == "__main__":
    demonstrate_chunking()