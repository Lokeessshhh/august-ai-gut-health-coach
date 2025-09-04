#!/usr/bin/env python3
"""
build_index.py - Creates FAISS vector index from gut health dataset

This script reads train/val/test JSONL files, creates embeddings using 
sentence-transformers, and builds a FAISS index for RAG retrieval.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


import os
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetIndexBuilder:
    """Builds FAISS index from gut health dataset"""
    
    def __init__(self, dataset_path: str = "raw/dataset/", index_path: str = "faiss_index"):
        """
        Initialize the index builder
        
        Args:
            dataset_path: Path to dataset folder containing JSONL files
            index_path: Path where FAISS index will be saved
        """
        self.dataset_path = Path(dataset_path)
        self.index_path = Path(index_path)
        self.embeddings = None
        self.vector_store = None
        
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load sentence-transformers embedding model"""
        try:
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 embeddings...")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    
    def _read_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSONL file and return list of records"""
        records = []
        try:
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist, skipping...")
                return records
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path} line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(records)} records from {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return records
    
    def _create_documents(self, records: List[Dict[str, Any]]) -> List[Document]:
        """Convert dataset records to LangChain documents"""
        documents = []
        
        for i, record in enumerate(records):
            try:
                # Extract required fields
                instruction = record.get('instruction', '').strip()
                response = record.get('response', '').strip()
                sources = record.get('sources', [])
                tone = record.get('tone', '')
                
                if not instruction or not response:
                    logger.warning(f"Record {i} missing instruction or response, skipping...")
                    continue
                
                # Create document content combining instruction and response
                content = f"Question: {instruction}\nAnswer: {response}"
                
                # Create metadata
                metadata = {
                    'instruction': instruction,
                    'response': response,
                    'sources': sources,
                    'tone': tone,
                    'doc_id': i
                }
                
                # Create LangChain document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                continue
        
        logger.info(f"Created {len(documents)} documents")
        return documents
    
    def build_index(self) -> None:
        """Build FAISS index from dataset files"""
        try:
            # Load embedding model
            self.embeddings = self._load_embeddings()
            
            # Read all dataset files
            all_records = []
            for filename in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
                file_path = self.dataset_path / filename
                records = self._read_jsonl(file_path)
                all_records.extend(records)
            
            if not all_records:
                raise ValueError("No records found in dataset files")
            
            logger.info(f"Total records loaded: {len(all_records)}")
            
            # Convert to documents
            documents = self._create_documents(all_records)
            
            if not documents:
                raise ValueError("No valid documents created from records")
            
            # Build FAISS index
            logger.info("Building FAISS vector store...")
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save index
            logger.info(f"Saving index to {self.index_path}...")
            self.index_path.mkdir(exist_ok=True)
            self.vector_store.save_local(str(self.index_path))
            
            logger.info("Index built and saved successfully!")
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def load_existing_index(self) -> FAISS:
        """Load existing FAISS index"""
        try:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index path {self.index_path} does not exist")
            
            if not self.embeddings:
                self.embeddings = self._load_embeddings()
            
            logger.info(f"Loading existing index from {self.index_path}...")
            self.vector_store = FAISS.load_local(
                str(self.index_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Index loaded successfully!")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

def main():
    """Main function to build the index"""
    try:
        # Initialize builder
        builder = DatasetIndexBuilder()
        
        # Check if dataset directory exists
        if not builder.dataset_path.exists():
            logger.error(f"Dataset path {builder.dataset_path} does not exist!")
            logger.error("Please ensure the dataset folder structure is correct:")
            logger.error("raw/dataset/")
            logger.error("├── train.jsonl")
            logger.error("├── val.jsonl")
            logger.error("├── test.jsonl")
            logger.error("├── tone_examples.jsonl")
            logger.error("└── negatives.jsonl")
            return
        
        # Build index
        builder.build_index()
        
        # Verify index was created
        vector_store = builder.load_existing_index()
        logger.info(f"Index verification successful! Vector store contains {vector_store.index.ntotal} vectors")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()