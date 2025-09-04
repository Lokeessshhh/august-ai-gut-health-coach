#!/usr/bin/env python3
"""
rag_pipeline.py - RAG pipeline for gut health question answering

This script implements a RAG pipeline that retrieves relevant documents from FAISS
and generates answers using NVIDIA's Llama3-8B model with appropriate tone guidance.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structure for RAG pipeline response"""
    answer: str
    sources: List[str]
    retrieved_docs: List[Dict[str, Any]]

class GutHealthRAG:
    """RAG pipeline for gut health question answering"""
    
    def __init__(
        self, 
        index_path: str = "faiss_index",
        dataset_path: str = "raw/dataset/",
        api_key: str = None,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            index_path: Path to FAISS index
            dataset_path: Path to dataset folder
            api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            top_k: Number of documents to retrieve
        """
        self.index_path = Path(index_path)
        self.dataset_path = Path(dataset_path)
        self.top_k = top_k
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm_client = None
        self.tone_examples = []
        
        # Setup API client
        self._setup_llm_client(api_key)
        
        # Load components
        self._load_embeddings()
        self._load_vector_store()
        self._load_tone_examples()
    
    def _setup_llm_client(self, api_key: str = None) -> None:
        """Setup NVIDIA API client"""
        try:
            # Get API key from parameter or environment
            if not api_key:
                api_key = os.getenv("NVIDIA_API_KEY")
            
            if not api_key:
                raise ValueError("NVIDIA API key not provided. Set NVIDIA_API_KEY env var or pass api_key parameter")
            
            self.llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            
            logger.info("NVIDIA API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
            raise
    
    def _load_embeddings(self) -> None:
        """Load sentence-transformers embedding model"""
        try:
            logger.info("Loading embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def _load_vector_store(self) -> None:
        """Load FAISS vector store"""
        try:
            if not self.index_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Run build_index.py first.")
            
            logger.info(f"Loading FAISS index from {self.index_path}...")
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"Vector store loaded with {self.vector_store.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def _load_tone_examples(self) -> None:
        """Load tone examples from tone_examples.jsonl"""
        try:
            tone_file = self.dataset_path / "tone_examples.jsonl"
            
            if not tone_file.exists():
                logger.warning(f"Tone examples file not found at {tone_file}")
                return
            
            with open(tone_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        example = json.loads(line.strip())
                        self.tone_examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tone example: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.tone_examples)} tone examples")
            
        except Exception as e:
            logger.error(f"Failed to load tone examples: {e}")
            # Continue without tone examples
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with tone examples"""
        base_prompt = """You are a helpful and knowledgeable gut health assistant. Your goal is to provide accurate, supportive, and friendly responses to questions about gut health, microbiome, digestive issues, and related topics.

Guidelines:
- Use a warm, supportive, and friendly tone
- Be empathetic and understanding
- Provide accurate information based on the retrieved context
- If sources are available, include them in your response
- Keep responses conversational but informative
- Acknowledge when someone might be struggling with health issues
"""
        
        # Add tone examples if available
        if self.tone_examples:
            base_prompt += "\n\nHere are examples of the tone and style you should use:\n\n"
            
            for i, example in enumerate(self.tone_examples[:3], 1):  # Limit to 3 examples
                instruction = example.get('instruction', '')
                response = example.get('response', '')
                base_prompt += f"Example {i}:\nQuestion: {instruction}\nResponse: {response}\n\n"
        
        base_prompt += """
Based on the retrieved context below, answer the user's question in a similar supportive and friendly tone. If the context includes sources, make sure to include them in your response.

Retrieved Context:
{context}

User Question: {question}

Provide your response now:"""
        
        return base_prompt
    
    def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from FAISS"""
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            
            retrieved_docs = []
            for doc, score in docs:
                doc_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                }
                retrieved_docs.append(doc_info)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate response using NVIDIA LLM"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                content = doc['content']
                metadata = doc['metadata']
                sources = metadata.get('sources', [])
                
                context_part = f"Document {i}:\n{content}"
                if sources:
                    context_part += f"\nSources: {', '.join(sources)}"
                context_parts.append(context_part)
            
            context = "\n\n".join(context_parts)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Format prompt with context and question
            formatted_prompt = system_prompt.format(context=context, question=query)
            
            # Call NVIDIA API
            response = self.llm_client.chat.completions.create(
                model="meta/llama3-8b-instruct",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Generated response successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while generating a response. Please try again later."
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sources from retrieved documents"""
        sources = set()
        
        for doc in retrieved_docs:
            doc_sources = doc['metadata'].get('sources', [])
            sources.update(doc_sources)
        
        return list(sources)
    
    def query(self, question: str) -> RAGResponse:
        """
        Main query function for the RAG pipeline
        
        Args:
            question: User's gut health question
            
        Returns:
            RAGResponse with answer, sources, and retrieved documents
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(question)
            
            if not retrieved_docs:
                return RAGResponse(
                    answer="I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or ask about a different gut health topic.",
                    sources=[],
                    retrieved_docs=[]
                )
            
            # Step 2: Generate response
            answer = self._generate_response(question, retrieved_docs)
            
            # Step 3: Extract sources
            sources = self._extract_sources(retrieved_docs)
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                retrieved_docs=retrieved_docs
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again later.",
                sources=[],
                retrieved_docs=[]
            )

def print_response(response: RAGResponse) -> None:
    """Print RAG response in a clean format"""
    print("\n" + "="*80)
    print("🦠 GUT HEALTH ASSISTANT RESPONSE")
    print("="*80)
    
    print("\n📝 ANSWER:")
    print("-" * 40)
    print(response.answer)
    
    if response.sources:
        print("\n🔗 SOURCES:")
        print("-" * 40)
        for i, source in enumerate(response.sources, 1):
            print(f"{i}. {source}")
    
    print("\n" + "="*80)

def main():
    """Demo function"""
    try:
        # Initialize RAG pipeline
        print("Initializing Gut Health RAG Pipeline...")
        rag = GutHealthRAG()
        
        # Example queries
        example_questions = [
            "What type of carb supports healthy gut bacteria?",
            "How does fiber affect gut health?",
            "Can probiotics help with IBS?"
        ]
        
        print("\n🚀 RAG Pipeline Ready!")
        print("\nTry some example questions or enter your own:")
        
        for question in example_questions:
            print(f"\n💭 Example: {question}")
        
        # Interactive loop
        print("\n" + "-"*60)
        print("Enter your gut health questions (type 'quit' to exit):")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Gut Health Assistant!")
                    break
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                # Process query
                response = rag.query(user_input)
                print_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using Gut Health Assistant!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Failed to initialize RAG pipeline: {e}")
        print("\nMake sure you have:")
        print("1. Built the FAISS index (run build_index.py)")
        print("2. Set your NVIDIA_API_KEY environment variable")
        print("3. Installed all required dependencies")

if __name__ == "__main__":
    main()