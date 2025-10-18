#!/usr/bin/env python3
"""
Test script to validate rag_chatbot.ipynb requirements and functionality.
This script checks which components can run and documents what needs external connectivity.
"""

import sys
import os

def test_imports():
    """Test that all required libraries can be imported."""
    print("=" * 80)
    print("Testing Library Imports")
    print("=" * 80)
    
    try:
        import glob
        print("✅ glob imported")
    except ImportError as e:
        print(f"❌ glob import failed: {e}")
        return False
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ PyPDFLoader imported")
    except ImportError as e:
        print(f"❌ PyPDFLoader import failed: {e}")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ RecursiveCharacterTextSplitter imported")
    except ImportError as e:
        print(f"❌ RecursiveCharacterTextSplitter import failed: {e}")
        return False
    
    try:
        from langchain_community.vectorstores import FAISS
        print("✅ FAISS imported")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        print("✅ SentenceTransformerEmbeddings imported")
    except ImportError as e:
        print(f"❌ SentenceTransformerEmbeddings import failed: {e}")
        return False
    
    try:
        from langchain_community.llms import Ollama
        print("✅ Ollama imported (but may need external service)")
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False
    
    try:
        from langchain.chains import ConversationalRetrievalChain
        print("✅ ConversationalRetrievalChain imported")
    except ImportError as e:
        print(f"❌ ConversationalRetrievalChain import failed: {e}")
        return False
    
    try:
        from langchain.prompts import PromptTemplate
        print("✅ PromptTemplate imported")
    except ImportError as e:
        print(f"❌ PromptTemplate import failed: {e}")
        return False
    
    print("\n✅ All required libraries can be imported!\n")
    return True


def test_pdf_loading():
    """Test loading PDF files."""
    print("=" * 80)
    print("Testing PDF Loading")
    print("=" * 80)
    
    try:
        import glob
        from langchain_community.document_loaders import PyPDFLoader
        
        pdf_paths = glob.glob("data/Everstorm_*.pdf")
        print(f"Found {len(pdf_paths)} PDF files:")
        for p in pdf_paths:
            print(f"  - {p}")
        
        if not pdf_paths:
            print("❌ No PDF files found in data/ directory")
            return False
        
        raw_docs = []
        for p in pdf_paths:
            loader = PyPDFLoader(p)
            docs = loader.load()
            raw_docs.extend(docs)
            print(f"  Loaded {len(docs)} pages from {os.path.basename(p)}")
        
        print(f"\n✅ Successfully loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files\n")
        return True, raw_docs
        
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_chunking(raw_docs):
    """Test text chunking."""
    print("=" * 80)
    print("Testing Text Chunking")
    print("=" * 80)
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
        chunks = text_splitter.split_documents(raw_docs)
        
        print(f"✅ Successfully created {len(chunks)} chunks")
        if chunks:
            print(f"  First chunk preview: {chunks[0].page_content[:100]}...")
        print()
        return True, chunks
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_embeddings_offline():
    """Test if embeddings can work (requires internet for model download)."""
    print("=" * 80)
    print("Testing Embeddings (Offline Mode)")
    print("=" * 80)
    
    print("⚠️  Note: The notebook uses sentence-transformers/gte-small which requires")
    print("   downloading models from HuggingFace. This requires internet connectivity.")
    print()
    print("❌ Cannot test embeddings in offline mode without pre-cached models")
    print()
    return False


def test_ollama():
    """Test if Ollama is available."""
    print("=" * 80)
    print("Testing Ollama Availability")
    print("=" * 80)
    
    try:
        import subprocess
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama found at: {result.stdout.strip()}")
            
            # Check if ollama service is running
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Ollama service is running")
                print("Available models:")
                print(result.stdout)
                return True
            else:
                print("⚠️  Ollama is installed but service may not be running")
                return False
        else:
            print("❌ Ollama is not installed")
            print("   Installation required: curl -fsSL https://ollama.com/install.sh | sh")
            return False
            
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False


def print_summary(results):
    """Print a summary of test results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nComponents that work:")
    for component, status in results.items():
        if status:
            print(f"  ✅ {component}")
    
    print("\nComponents that need external resources:")
    for component, status in results.items():
        if not status:
            print(f"  ❌ {component}")
    
    print("\n" + "=" * 80)
    print("REQUIREMENTS FOR FULL FUNCTIONALITY")
    print("=" * 80)
    print("""
The notebook requires the following external resources:

1. Internet connectivity to HuggingFace (huggingface.co):
   - Required for downloading the sentence-transformers/gte-small model
   - First-time setup only; models are cached locally after download
   
2. Ollama installation and service:
   - Install: curl -fsSL https://ollama.com/install.sh | sh
   - Download model: ollama pull gemma3:1b
   - Service must be running: ollama serve (runs in background)

3. Optional: Internet connectivity for web scraping (UnstructuredURLLoader)
   - Used to load additional documentation from web pages
   - Can be skipped if only using local PDF files

WORKAROUND for offline environments:
- Pre-download and cache the embedding model on a connected machine
- Transfer the cache directory (~/.cache/huggingface/) to the offline machine
- Install Ollama and pull the gemma3:1b model while connected
- Ollama models are stored locally and don't need internet after download
""")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 80)
    print("RAG CHATBOT NOTEBOOK - REQUIREMENTS TEST")
    print("*" * 80)
    print()
    
    results = {}
    
    # Test imports
    results['Library Imports'] = test_imports()
    
    # Test PDF loading
    pdf_success, raw_docs = test_pdf_loading()
    results['PDF Loading'] = pdf_success
    
    # Test chunking
    if pdf_success:
        chunk_success, chunks = test_chunking(raw_docs)
        results['Text Chunking'] = chunk_success
    else:
        results['Text Chunking'] = False
    
    # Test embeddings (will fail in offline mode)
    results['Embeddings (Online)'] = test_embeddings_offline()
    
    # Test Ollama
    results['Ollama'] = test_ollama()
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    all_critical_passed = results['Library Imports'] and results['PDF Loading'] and results['Text Chunking']
    if all_critical_passed:
        print("\n✅ Core notebook components are functional (PDF loading, chunking)")
        print("⚠️  External services (HuggingFace, Ollama) are required for full functionality")
        return 0
    else:
        print("\n❌ Some core components failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
