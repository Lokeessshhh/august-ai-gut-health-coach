# Gut Health RAG Pipeline Setup Guide

## Overview
This RAG pipeline provides intelligent question-answering for gut health topics using:
- **Retrieval**: FAISS vector store with sentence-transformers embeddings
- **Generation**: NVIDIA Llama3-8B model via OpenAI-compatible API
- **Tone Guidance**: Examples from `tone_examples.jsonl` for consistent friendly responses

## Prerequisites

### 1. Install Dependencies
```bash
pip install langchain sentence-transformers faiss-cpu openai pathlib
```

For GPU support (optional but faster):
```bash
pip install faiss-gpu
```

### 2. Dataset Structure
Ensure your dataset follows this structure:
```
raw/dataset/
├── train.jsonl
├── val.jsonl  
├── test.jsonl
├── tone_examples.jsonl
└── negatives.jsonl
```

### 3. Environment Setup
Set your NVIDIA API key:
```bash
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

Or create a `.env` file:
```bash
echo "NVIDIA_API_KEY=your_nvidia_api_key_here" > .env
```

## Usage Instructions

### Step 1: Build the FAISS Index

First, create the vector index from your dataset:

```bash
python build_index.py
```

**What this does:**
- Reads all JSONL files (train/val/test)
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Builds FAISS vector store
- Saves index to `faiss_index/` directory

**Expected output:**
```
2024-09-04 10:15:23 - INFO - Loading sentence-transformers/all-MiniLM-L6-v2 embeddings...
2024-09-04 10:15:25 - INFO - Loaded 150 records from raw/dataset/train.jsonl
2024-09-04 10:15:25 - INFO - Loaded 30 records from raw/dataset/val.jsonl
2024-09-04 10:15:25 - INFO - Loaded 25 records from raw/dataset/test.jsonl
2024-09-04 10:15:25 - INFO - Total records loaded: 205
2024-09-04 10:15:25 - INFO - Created 205 documents
2024-09-04 10:15:26 - INFO - Building FAISS vector store...
2024-09-04 10:15:28 - INFO - Saving index to faiss_index...
2024-09-04 10:15:28 - INFO - Index built and saved successfully!
```

### Step 2: Run the RAG Pipeline

Launch the interactive question-answering system:

```bash
python rag_pipeline.py
```

**What this does:**
- Loads the FAISS index and embeddings
- Connects to NVIDIA API
- Loads tone examples for response guidance
- Starts interactive Q&A session

**Example interaction:**
```
🚀 RAG Pipeline Ready!

💭 Example: What type of carb supports healthy gut bacteria?
💭 Example: How does fiber affect gut health?
💭 Example: Can probiotics help with IBS?

❓ Your question: What is sulfoquinovose and how does it help gut bacteria?

================================================================================
🦠 GUT HEALTH ASSISTANT RESPONSE
================================================================================

📝 ANSWER:
I totally understand your curiosity about sulfoquinovose! It's actually a really 
fascinating compound. Sulfoquinovose is a special type of carbohydrate that's 
found in certain leafy greens and other plant foods. What makes it so interesting 
is that it acts as a prebiotic - meaning it serves as food for the beneficial 
bacteria in your gut.

Research has shown that sulfoquinovose specifically supports the growth of 
healthy gut bacteria, which can help improve your overall digestive health and 
potentially boost your immune system. It's one of those "good carbs" that your 
gut microbiome really appreciates!

🔗 SOURCES:
1. https://mcpress.mayoclinic.org/...
================================================================================
```

### Step 3: Programmatic Usage

You can also use the RAG pipeline programmatically:

```python
from rag_pipeline import GutHealthRAG

# Initialize the pipeline
rag = GutHealthRAG(api_key="your_api_key")

# Ask a question
response = rag.query("How does fiber benefit gut health?")

# Access the results
print("Answer:", response.answer)
print("Sources:", response.sources)
print("Retrieved docs:", len(response.retrieved_docs))
```

## Configuration Options

### GutHealthRAG Parameters

```python
rag = GutHealthRAG(
    index_path="faiss_index",           # Path to FAISS index
    dataset_path="raw/dataset/",        # Path to dataset folder
    api_key=None,                       # NVIDIA API key (or use env var)
    top_k=5                            # Number of documents to retrieve
)
```

### DatasetIndexBuilder Parameters

```python
builder = DatasetIndexBuilder(
    dataset_path="raw/dataset/",        # Dataset folder path
    index_path="faiss_index"           # Where to save the index
)
```

## File Structure After Setup

```
your_project/
├── build_index.py                 # Index building script
├── rag_pipeline.py               # RAG pipeline implementation
├── faiss_index/                  # Generated FAISS index (after build_index.py)
│   ├── index.faiss
│   └── index.pkl
└── raw/dataset/                  # Your dataset
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── tone_examples.jsonl
    └── negatives.jsonl
```

## Troubleshooting

### Common Issues

**1. "FAISS index not found" error**
```bash
# Solution: Build the index first
python build_index.py
```

**2. "NVIDIA API key not provided" error**
```bash
# Solution: Set environment variable
export NVIDIA_API_KEY="your_key_here"
```

**3. Import errors**
```bash
# Solution: Install missing dependencies
pip install langchain sentence-transformers faiss-cpu openai
```

**4. GPU memory issues**
```python
# In build_index.py, change device to CPU
model_kwargs={'device': 'cpu'}
```

### Performance Tips

1. **GPU Acceleration**: Install `faiss-gpu` and use `device='cuda'` in embeddings
2. **Batch Processing**: For many queries, use batch processing instead of individual calls
3. **Index Optimization**: Use `faiss.IndexIVFFlat` for larger datasets (>10k documents)

## API Reference

### RAGResponse Class
```python
@dataclass
class RAGResponse:
    answer: str                    # Generated answer
    sources: List[str]            # Source URLs from dataset
    retrieved_docs: List[Dict]    # Retrieved document metadata
```

### Main Methods

#### GutHealthRAG.query(question: str) -> RAGResponse
Main query method that:
1. Retrieves relevant documents using similarity search
2. Generates response using NVIDIA Llama3-8B
3. Extracts and returns sources

#### DatasetIndexBuilder.build_index() -> None
Builds FAISS index from dataset files:
1. Reads JSONL files
2. Creates embeddings
3. Builds and saves FAISS index

## Next Steps

- The `negatives.jsonl` file is ready for evaluation/testing implementations
- Consider adding response caching for frequently asked questions  
- Implement batch processing for handling multiple queries efficiently
- Add support for additional embedding models or reranking