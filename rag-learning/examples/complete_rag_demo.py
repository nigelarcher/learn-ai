#!/usr/bin/env python3
"""
Complete RAG Demo: From Web Scraping to Q&A
Shows the full pipeline from data collection to answering questions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_scraper import ResponsibleWebScraper, load_scraped_documents
from rag_pipeline import SimpleRAGPipeline


def run_complete_rag_demo():
    """Run a complete RAG demonstration from scraping to Q&A."""
    print("🎯 Complete RAG System Demo")
    print("=" * 60)
    print("This demo will:")
    print("1. Scrape AI-related content from Wikipedia")
    print("2. Process and store it in a vector database")
    print("3. Answer questions using the scraped knowledge")
    print("=" * 60)
    
    # Step 1: Check if we need to scrape data
    scraped_file = "data/scraped_ai_knowledge.json"
    
    if not os.path.exists(scraped_file):
        print("\n📊 Step 1: Scraping Wikipedia for AI knowledge...")
        scraper = ResponsibleWebScraper(delay_seconds=1.0, max_pages=5)
        
        ai_topics = [
            "Artificial intelligence",
            "Machine learning",
            "Natural language processing",
            "Computer vision",
            "Neural network"
        ]
        
        documents = scraper.scrape_topics(ai_topics)
    else:
        print("\n📊 Step 1: Loading previously scraped documents...")
        documents = load_scraped_documents(scraped_file)
        print(f"Loaded {len(documents)} documents")
    
    if not documents:
        print("❌ No documents available. Please run the scraper first.")
        return
    
    # Step 2: Initialize RAG system
    print("\n🔧 Step 2: Initializing RAG system...")
    rag = SimpleRAGPipeline(
        chunk_size=400,      # Larger chunks for Wikipedia content
        chunk_overlap=100,   # Good overlap for context
        embedding_dim=200,   # Reasonable size for demo
        top_k=4             # Retrieve top 4 chunks
    )
    
    # Step 3: Ingest scraped documents
    print("\n📥 Step 3: Processing and storing documents...")
    
    # Convert scraped format to RAG format
    rag_documents = []
    for doc in documents:
        rag_documents.append({
            "content": doc['content'],
            "metadata": {
                "source": doc['url'],
                "title": doc['title'],
                "sections": doc.get('sections', []),
                "scraped_at": doc['scraped_at']
            }
        })
    
    rag.ingest_documents(rag_documents)
    
    # Step 4: Interactive Q&A
    print("\n💬 Step 4: Interactive Q&A System")
    print("=" * 60)
    print("Ask questions about AI topics!")
    print("Type 'quit' to exit, 'stats' for system stats")
    print("=" * 60)
    
    # Example questions to get started
    example_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?",
        "What is computer vision used for?",
        "What is natural language processing?"
    ]
    
    print("\n📝 Example questions you can ask:")
    for q in example_questions[:3]:
        print(f"  - {q}")
    
    # Interactive loop
    while True:
        print("\n" + "-" * 60)
        query = input("Your question: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Thanks for using the RAG demo!")
            break
            
        elif query.lower() == 'stats':
            stats = rag.vector_store.get_stats()
            print("\n📊 System Statistics:")
            print(f"  Total chunks: {stats['total_documents']}")
            print(f"  Embedding dimension: {stats['embedding_dimension']}")
            print(f"  Memory usage: {stats['memory_size_mb']:.2f} MB")
            print(f"  Unique sources: {stats['unique_sources']}")
            continue
            
        elif not query:
            continue
            
        # Get RAG response
        response = rag.query(query)
        
        # Display answer
        print(f"\n🤖 Answer: {response.answer}")
        print(f"\n📊 Confidence: {response.confidence:.1%}")
        
        # Show sources
        if response.sources:
            print("\n📚 Sources:")
            unique_sources = {}
            for source in response.sources:
                title = source.get('title', 'Unknown')
                if title not in unique_sources:
                    unique_sources[title] = source.get('source', '')
            
            for title, url in unique_sources.items():
                print(f"  - {title}")
                if url:
                    print(f"    {url}")
        
        # Show what was retrieved (optional detail)
        show_chunks = input("\nShow retrieved chunks? (y/n): ").strip().lower()
        if show_chunks == 'y':
            print("\n📄 Retrieved chunks:")
            for i, (chunk, similarity) in enumerate(response.retrieved_chunks):
                print(f"\n[{i+1}] Similarity: {similarity:.2%}")
                print(f"{chunk}")
    
    # Save the index for future use
    print("\n💾 Saving RAG index...")
    rag.save_index("data/wikipedia_ai_rag_index.json")
    print("✅ Index saved! You can load it later for faster startup.")


def quick_demo():
    """Run a quick non-interactive demo."""
    print("🚀 Quick RAG Demo (Non-interactive)")
    print("=" * 60)
    
    # Load existing data or use sample
    try:
        documents = load_scraped_documents("data/scraped_ai_knowledge.json")
    except:
        # Use sample documents if no scraped data
        documents = [
            {
                "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
                "url": "sample_ai.txt",
                "title": "Artificial Intelligence"
            }
        ]
    
    # Create RAG system
    rag = SimpleRAGPipeline()
    
    # Ingest documents
    rag_docs = [{
        "content": doc.get('content', ''),
        "metadata": {"source": doc.get('url', ''), "title": doc.get('title', '')}
    } for doc in documents]
    
    rag.ingest_documents(rag_docs)
    
    # Test queries
    queries = [
        "What is artificial intelligence?",
        "How do machines demonstrate intelligence?",
        "What are intelligent agents?"
    ]
    
    for query in queries:
        response = rag.query(query)
        print(f"\nQ: {query}")
        print(f"A: {response.answer}")
        print(f"Confidence: {response.confidence:.1%}")
        print("-" * 40)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        run_complete_rag_demo()