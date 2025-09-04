#!/usr/bin/env python3
"""
app.py - Optimized Streamlit chatbot interface for the Gut Health RAG pipeline

This Streamlit app provides a fast, responsive chatbot interface that wraps
the RAG pipeline for gut health question answering with significant performance
improvements.
"""
import os
import streamlit as st
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import our RAG pipeline
from gut_rag import GutHealthRAG, RAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Page configuration (only set once)
if 'page_configured' not in st.session_state:
    st.set_page_config(
        page_title="🦠 Gut Health Assistant",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_configured = True

# Custom CSS for better styling (optimized)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #2E7D32;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-container {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E7D32;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #555;
    }
    
    .stChatMessage {
        margin: 0.5rem 0;
    }
    
    .source-item {
        font-size: 0.9rem;
        color: #1976D2;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_evaluation_results_cached(eval_file: str = "eval_results.jsonl") -> Dict[str, Any]:
    """Cached version of evaluation results loading"""
    try:
        eval_path = Path(eval_file)
        if not eval_path.exists():
            return {}
        
        results = []
        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                except json.JSONDecodeError:
                    continue
        
        if not results:
            return {}
        
        # Calculate summary statistics
        total_questions = len(results)
        avg_response_time = sum(r.get('response_time', 0) for r in results) / total_questions
        avg_retrieval_score = sum(r.get('retrieval_relevance', 0) for r in results) / total_questions
        avg_answer_quality = sum(r.get('answer_quality', 0) for r in results) / total_questions
        
        return {
            'total_questions': total_questions,
            'avg_response_time': avg_response_time,
            'avg_retrieval_score': avg_retrieval_score,
            'avg_answer_quality': avg_answer_quality,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        return {}

@st.cache_resource
def get_rag_pipeline():
    """Singleton pattern for RAG pipeline with caching"""
    try:
        return GutHealthRAG()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def initialize_session_state():
    """Initialize all session state variables once"""
    if 'initialized' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.rag_pipeline = None
        st.session_state.system_ready = False
        st.session_state.initialized = True
        st.session_state.message_count = 0

def cleanup_chat_history():
    """Clean up old chat messages to prevent memory bloat"""
    max_history = 100
    if len(st.session_state.chat_history) > max_history:
        st.session_state.chat_history = st.session_state.chat_history[-max_history:]
        logger.info(f"Cleaned up chat history, kept last {max_history} messages")

@monitor_performance
def load_rag_pipeline_async():
    """Load RAG pipeline with proper error handling and progress"""
    if 'rag_pipeline' not in st.session_state or st.session_state.rag_pipeline is None:
        try:
            with st.spinner("🚀 Loading knowledge base..."):
                pipeline = get_rag_pipeline()
                if pipeline is not None:
                    st.session_state.rag_pipeline = pipeline
                    st.session_state.system_ready = True
                    return True
                else:
                    st.session_state.system_ready = False
                    return False
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)[:100]}...")
            st.session_state.system_ready = False
            return False
    return st.session_state.system_ready

def render_sidebar_optimized():
    """Optimized sidebar with conditional updates"""
    with st.sidebar:
        st.header("📊 Performance Monitor")
        
        # Load evaluation results (cached)
        eval_results = load_evaluation_results_cached()
        
        if eval_results:
            # Compact metric display
            metrics = [
                ("Questions", eval_results['total_questions'], "🔢"),
                ("Avg Time", f"{eval_results['avg_response_time']:.1f}s", "⏱️"),
                ("Relevance", f"{eval_results['avg_retrieval_score']*100:.0f}%", "🎯"),
                ("Quality", f"{eval_results['avg_answer_quality']*100:.0f}%", "⭐")
            ]
            
            # Display metrics in a compact grid
            for i in range(0, len(metrics), 2):
                col1, col2 = st.columns(2)
                
                # First metric
                label1, value1, icon1 = metrics[i]
                col1.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{icon1} {value1}</div>
                    <div class="metric-label">{label1}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Second metric (if exists)
                if i + 1 < len(metrics):
                    label2, value2, icon2 = metrics[i + 1]
                    col2.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{icon2} {value2}</div>
                        <div class="metric-label">{label2}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📊 No evaluation data available")
        
        st.divider()
        
        # System controls
        st.subheader("🔧 Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.success("Chat cleared!")
            st.rerun()
        
        # System status
        if st.session_state.get('system_ready', False):
            st.success("✅ System Ready")
            
            # Show system info
            try:
                if hasattr(st.session_state.rag_pipeline, 'vector_store'):
                    if hasattr(st.session_state.rag_pipeline.vector_store, 'index'):
                        doc_count = st.session_state.rag_pipeline.vector_store.index.ntotal
                        st.info(f"📚 {doc_count:,} documents indexed")
                    else:
                        st.info("📚 Vector store ready")
            except Exception as e:
                logger.warning(f"Could not get document count: {e}")
                st.info("📚 Knowledge base loaded")
                
            # Chat stats
            if st.session_state.chat_history:
                chat_count = len([msg for msg in st.session_state.chat_history if msg.get('is_user', False)])
                st.info(f"💬 {chat_count} questions asked")
        else:
            st.error("❌ System Not Ready")
            if st.button("🔄 Retry Setup", use_container_width=True):
                st.session_state.rag_pipeline = None
                st.rerun()

def display_welcome_message():
    """Display optimized welcome message"""
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🦠"):
            st.markdown("""
            👋 **Welcome to your Gut Health Assistant!**
            
            I can help you with:
            • 🦠 Gut microbiome and digestive health
            • 💊 Probiotics and prebiotics
            • 🥗 Diet and nutrition for gut health
            • 🩺 Digestive issues and symptoms
            • 🏃‍♀️ Lifestyle factors affecting gut health
            
            **Try asking:** *"What foods improve gut health?"* or *"How do probiotics work?"*
            """)

def display_chat_messages_optimized():
    """Optimized chat message display with pagination"""
    # Limit visible messages for performance
    max_visible_messages = 20
    
    if st.session_state.chat_history:
        # Show pagination info if needed
        total_messages = len(st.session_state.chat_history)
        if total_messages > max_visible_messages:
            st.info(f"Showing last {max_visible_messages} of {total_messages} messages")
        
        # Display recent messages
        visible_messages = st.session_state.chat_history[-max_visible_messages:]
        
        for message in visible_messages:
            if message.get('is_user'):
                with st.chat_message("user", avatar="👤"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant", avatar="🦠"):
                    st.write(message['content'])
                    
                    # Display sources in an expander
                    if message.get('sources'):
                        with st.expander(f"📖 Sources ({len(message['sources'])})", expanded=False):
                            for i, source in enumerate(message['sources'][:5], 1):  # Limit to 5 sources
                                if source.startswith('http'):
                                    st.markdown(f"{i}. 🔗 [{source}]({source})")
                                else:
                                    st.markdown(f"{i}. 📄 {source}")

@monitor_performance
def process_user_query(user_input: str):
    """Process user query with optimized error handling"""
    if not user_input.strip():
        return
    
    # Clean up history if needed
    cleanup_chat_history()
    
    # Add user message immediately
    user_message = {
        'content': user_input.strip(),
        'is_user': True,
        'timestamp': datetime.now()
    }
    st.session_state.chat_history.append(user_message)
    st.session_state.message_count += 1
    
    try:
        # Process query with timeout handling
        start_time = time.time()
        response = st.session_state.rag_pipeline.query(user_input.strip())
        processing_time = time.time() - start_time
        
        # Add successful response
        assistant_message = {
            'content': response.answer,
            'sources': response.sources[:5] if hasattr(response, 'sources') else [],
            'is_user': False,
            'timestamp': datetime.now(),
            'processing_time': processing_time
        }
        st.session_state.chat_history.append(assistant_message)
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        
        # Add error message
        error_message = {
            'content': "I'm sorry, but I encountered an issue processing your question. Please try rephrasing or ask a different question.",
            'sources': [],
            'is_user': False,
            'timestamp': datetime.now(),
            'error': True
        }
        st.session_state.chat_history.append(error_message)

def main():
    """Optimized main Streamlit app"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header with better styling
    st.markdown('<h1 class="main-header">🦠 Gut Health Assistant</h1>', unsafe_allow_html=True)
    st.caption("*Fast, evidence-based answers to your gut health questions*")
    
    # Load RAG pipeline with progress indicator
    system_ready = load_rag_pipeline_async()
    
    if not system_ready:
        st.error("⚠️ **System Initialization Failed**")
        st.markdown("""
        **Please check:**
        1. ✅ FAISS index built (run `build_index.py`)
        2. ✅ NVIDIA_API_KEY environment variable set
        3. ✅ All dependencies installed
        4. ✅ Vector store files present
        """)
        
        if st.button("🔄 Retry Initialization", type="primary"):
            st.session_state.rag_pipeline = None
            st.session_state.system_ready = False
            st.rerun()
        return
    
    # Render optimized sidebar
    render_sidebar_optimized()
    
    # Main chat interface
    st.subheader("💬 Chat Interface")
    
    # Welcome message
    display_welcome_message()
    
    # Display chat messages (optimized)
    display_chat_messages_optimized()
    
    # Chat input (moved to bottom for better UX)
    user_input = st.chat_input(
        placeholder="Ask about gut health, probiotics, microbiome, digestion...",
        key="main_chat_input"
    )
    
    # Process input
    if user_input:
        with st.spinner("🤔 Analyzing your question..."):
            process_user_query(user_input)
        st.rerun()
    
    # Quick suggestions
    if not st.session_state.chat_history:
        st.markdown("**💡 Quick Start Questions:**")
        suggestions = [
            "What foods promote a healthy gut microbiome?",
            "How do antibiotics affect gut bacteria?",
            "What's the difference between probiotics and prebiotics?",
            "Can stress affect digestive health?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💭 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    with st.spinner("🤔 Processing..."):
                        process_user_query(suggestion)
                    st.rerun()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🔬 <em>Educational information only - consult healthcare professionals for medical advice</em>"
            "</div>", 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        logger.error(f"Application error: {e}")
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()
