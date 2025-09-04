#!/usr/bin/env python3
"""
debug_imports.py - Debug script to identify import issues

This script helps identify what's causing the import error in the RAG pipeline.
"""

import sys
import os
from pathlib import Path

def check_file_exists():
    """Check if required files exist"""
    print("=== FILE EXISTENCE CHECK ===")
    
    files_to_check = [
        "rag_pipeline.py",
        "app.py", 
        "faiss_index",
        "raw/dataset/",
        ".env"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        exists = path.exists()
        file_type = "Directory" if path.is_dir() else "File"
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{file_type:10} {file_path:20} {status}")

def check_python_path():
    """Check Python path and working directory"""
    print("\n=== PYTHON PATH CHECK ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path:")
    for path in sys.path:
        print(f"  - {path}")

def test_basic_imports():
    """Test importing required packages one by one"""
    print("\n=== BASIC IMPORTS TEST ===")
    
    packages = [
        "streamlit",
        "json", 
        "logging",
        "pathlib",
        "typing",
        "datetime",
        "re",
        "os",
        "dataclasses",
        "openai",
        "langchain_huggingface",
        "langchain_community.vectorstores",
        "langchain_core.documents", 
        "dotenv"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
        except Exception as e:
            print(f"⚠️  {package}: {e}")

def test_rag_pipeline_import():
    """Test importing from rag_pipeline.py step by step"""
    print("\n=== RAG PIPELINE IMPORT TEST ===")
    
    try:
        # First, try to import the module
        print("1. Attempting to import rag_pipeline module...")
        import rag_pipeline
        print("✅ rag_pipeline module imported successfully")
        
        # Check what's available in the module
        print("\n2. Checking available attributes in rag_pipeline:")
        attributes = dir(rag_pipeline)
        for attr in attributes:
            if not attr.startswith('_'):
                print(f"   - {attr}")
        
        # Try to import specific classes
        print("\n3. Attempting to import specific classes...")
        
        try:
            from rag_pipeline import RAGResponse
            print("✅ RAGResponse imported successfully")
        except ImportError as e:
            print(f"❌ RAGResponse import failed: {e}")
        
        try:
            from rag_pipeline import GutHealthRAG
            print("✅ GutHealthRAG imported successfully")
        except ImportError as e:
            print(f"❌ GutHealthRAG import failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to import rag_pipeline module: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to read the file and check for syntax errors
        try:
            print("\n4. Checking rag_pipeline.py for syntax errors...")
            with open('rag_pipeline.py', 'r') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, 'rag_pipeline.py', 'exec')
            print("✅ No syntax errors found in rag_pipeline.py")
            
        except FileNotFoundError:
            print("❌ rag_pipeline.py file not found")
        except SyntaxError as se:
            print(f"❌ Syntax error in rag_pipeline.py: {se}")
        except Exception as ce:
            print(f"❌ Compilation error: {ce}")

def check_environment_variables():
    """Check for required environment variables"""
    print("\n=== ENVIRONMENT VARIABLES CHECK ===")
    
    env_vars = [
        "NVIDIA_API_KEY",
        "PYTHONPATH"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Don't print the full API key for security
            if "KEY" in var or "TOKEN" in var:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

def check_faiss_index():
    """Check FAISS index structure"""
    print("\n=== FAISS INDEX CHECK ===")
    
    index_path = Path("faiss_index")
    if not index_path.exists():
        print("❌ faiss_index directory not found")
        print("   You need to run build_index.py first")
        return
    
    # Check for required FAISS files
    required_files = ["index.faiss", "index.pkl"]
    for file in required_files:
        file_path = index_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file}: {size:,} bytes")
        else:
            print(f"❌ {file}: Missing")

def main():
    """Run all diagnostic checks"""
    print("🔍 RAG PIPELINE IMPORT DIAGNOSTICS")
    print("=" * 50)
    
    check_file_exists()
    check_python_path()
    test_basic_imports()
    check_environment_variables()
    check_faiss_index()
    test_rag_pipeline_import()
    
    print("\n" + "=" * 50)
    print("🏁 DIAGNOSTICS COMPLETE")
    print("\nIf you see any ❌ errors above, those need to be fixed first.")
    print("Common solutions:")
    print("1. Install missing packages: pip install package_name")
    print("2. Set environment variables in .env file")  
    print("3. Run build_index.py to create FAISS index")
    print("4. Check file paths and working directory")

if __name__ == "__main__":
    main()