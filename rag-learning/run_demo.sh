#!/bin/bash
# Run the RAG learning demos

echo "🎯 RAG Learning Project - Demo Runner"
echo "===================================="
echo ""

# Create data directory if it doesn't exist
mkdir -p data

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt --quiet

echo ""
echo "Choose a demo to run:"
echo "1) Simple Embeddings Demo - See how text becomes vectors"
echo "2) Document Chunking Demo - Learn about chunking strategies"
echo "3) Vector Store Demo - See how documents are stored and searched"
echo "4) Web Scraping Demo - Scrape Wikipedia for AI knowledge"
echo "5) Complete RAG Pipeline - See the full RAG system in action"
echo "6) Interactive Q&A Demo - Ask questions about scraped content"
echo "7) RAG Chatbot - Chat with a bot that knows about RAG"
echo "8) Quick Demo - Fast non-interactive demonstration"
echo ""

read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        echo "Running Embeddings Demo..."
        python src/simple_embedder.py
        ;;
    2)
        echo "Running Document Chunking Demo..."
        python src/document_processor.py
        ;;
    3)
        echo "Running Vector Store Demo..."
        python src/simple_vector_store.py
        ;;
    4)
        echo "Running Web Scraping Demo..."
        python src/web_scraper.py
        ;;
    5)
        echo "Running RAG Pipeline Demo..."
        python src/rag_pipeline.py
        ;;
    6)
        echo "Running Interactive Q&A Demo..."
        python examples/complete_rag_demo.py
        ;;
    7)
        echo "Running RAG Chatbot..."
        python examples/simple_chatbot.py
        ;;
    8)
        echo "Running Quick Demo..."
        python examples/complete_rag_demo.py --quick
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-8."
        ;;
esac

echo ""
echo "Demo complete! 🎉"