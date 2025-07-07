# RAG Learning Project 🔍📚

**Learn Retrieval-Augmented Generation (RAG) from scratch** - build a chatbot that can answer questions using your own documents and public data.

## 🎯 What We're Building

A complete RAG system that:
- Takes your documents and breaks them into searchable chunks
- Stores them in a vector database for semantic search
- Answers questions by finding relevant info and generating responses
- Provides a chatbot interface using Google Agent Kit

## 🧠 What is RAG? (ELI5)

**RAG = Retrieval-Augmented Generation**

Think of it like giving an AI assistant access to a library:

1. **Without RAG**: AI only knows what it learned during training (like a student taking a test from memory)
2. **With RAG**: AI can look up information in real-time (like a student with access to textbooks during an open-book test)

**The Process:**
```
Your Question → Find Relevant Documents → Send Both to AI → Get Answer with Sources
```

## 📁 Project Structure

```
rag-learning/
├── src/                    # Core RAG implementation
│   ├── embedder.py        # Convert text to vectors
│   ├── retriever.py       # Find relevant documents
│   ├── generator.py       # Generate answers using LLM
│   └── rag_pipeline.py    # Complete RAG workflow
├── data/                  # Knowledge base documents
│   ├── raw/              # Original documents
│   └── processed/        # Chunked and embedded
├── docs/                 # Documentation and guides
│   ├── rag_explained.md  # Deep dive into RAG concepts
│   └── architecture.md   # System design
├── examples/             # Usage examples and demos
└── chatbot/              # Google Agent Kit integration
```

## 🚀 Quick Start

```bash
# Setup the environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the basic RAG example
python examples/simple_rag.py

# Start the chatbot interface
python chatbot/app.py
```

## 🔧 Key Components

### 1. Document Processing
- **Text Chunking**: Break large documents into manageable pieces
- **Embedding**: Convert text chunks into vector representations
- **Storage**: Save vectors in a searchable database

### 2. Retrieval System
- **Semantic Search**: Find documents similar to user questions
- **Ranking**: Order results by relevance
- **Context Window**: Manage how much text to send to the LLM

### 3. Generation System
- **Prompt Engineering**: Combine question + retrieved docs
- **LLM Integration**: Use language models for response generation
- **Source Attribution**: Track which documents informed the answer

## 📚 Learning Path

1. **Understand RAG Basics** - What it is and why it's useful
2. **Build Simple RAG** - Basic retrieval + generation pipeline
3. **Add Vector Database** - Scalable document storage
4. **Improve Retrieval** - Better search and ranking
5. **Build Chatbot** - User-friendly interface
6. **Deploy and Scale** - Production considerations

## 🎓 What You'll Learn

- **Vector Embeddings**: How text becomes searchable numbers
- **Semantic Search**: Finding meaning, not just keywords
- **Prompt Engineering**: Crafting effective AI instructions
- **LLM Integration**: Working with language models
- **System Architecture**: Building scalable AI systems

## 🎯 Use Cases

- **Internal Knowledge Base**: Company docs, policies, procedures
- **Customer Support**: FAQ automation with real-time info
- **Research Assistant**: Query academic papers or reports
- **Personal Assistant**: Search your notes, emails, documents

---

*This project teaches both the theory and practice of RAG systems, from basic concepts to production deployment.*