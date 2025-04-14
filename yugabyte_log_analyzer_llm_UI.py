"""
YugabyteDB Log Analyzer with ChromaDB and Ollama

Objective:
- Analyze YugabyteDB log files using semantic search and LLM (Llama 3)
- Enable natural language queries about log contents
- Provide structured visualization of log entries

High-Level Workflow:
1. Upload log file → Parse and structure log entries
2. Store processed logs in ChromaDB vector database
3. Query logs using natural language
4. Display analyzed results with original log context

Requirements:
- Python 3.10+
- Required packages: chromadb, ollama, streamlit, sentence-transformers
- Ollama installed with Llama 3 model (ollama pull llama3)
- 4GB+ RAM (8GB recommended for larger log files)

How to Run:
1. Install requirements: pip install chromadb ollama streamlit sentence-transformers
2. Start Ollama: ollama serve (in separate terminal)
3. Run app: streamlit run yugabyte_log_analyzer.py

Usage:
1. Upload YugabyteDB log file via sidebar
2. Wait for processing to complete
3. Enter natural language questions in main interface
4. View analysis and explore individual log entries
"""

import re
import chromadb
from datetime import datetime
import ollama
import streamlit as st
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import os
import time

# --------------------------
# Constants
# --------------------------
CHROMA_PATH = "./chroma_yugabyte"  # Directory to store ChromaDB data
LOG_PATTERN = re.compile(
    r'(?P<level>[IWEF])(?P<timestamp>\d{4} \d{2}:\d{2}:\d{2}\.\d{6})\s+'
    r'(?P<thread>\d+)\s+(?P<file>.+?)\] '
    r'(?:\[(?P<node_id>[\w-]+)\])?\s*(?:T (?P<tablet_id>\w+):?)?\s*(?P<message>.*)'
)
SYSTEM_PROMPT = """You are an expert YugabyteDB administrator analyzing logs. Follow these rules:
1. Be concise but precise
2. Always reference specific node/tablet IDs when available
3. Highlight critical errors first
4. Suggest potential solutions when possible
5. Include relevant timestamps"""

# --------------------------
# Log Parsing Functions
# --------------------------
def parse_log_line(line: str) -> Optional[Dict]:
    """
    Parse a single YugabyteDB log line into structured components
    
    Args:
        line: Raw log line text
        
    Returns:
        Dictionary containing parsed components:
        - level: Log severity (I/W/E/F)
        - timestamp: ISO formatted timestamp
        - thread_id: Thread identifier
        - file: Source file
        - node_id: Cluster node ID
        - tablet_id: Tablet identifier  
        - message: Log message content
        - raw: Original log line
        - is_structured: Whether parsing succeeded
        - stacktrace: List of stacktrace lines
    """
    line = line.strip()
    if not line:
        return None
    
    match = LOG_PATTERN.match(line)
    if not match:
        return {
            "raw": line,
            "is_structured": False,
            "message": line,
            "level": "U",  # 'U' for Unknown
            "timestamp": "",
            "thread_id": "",
            "file": "",
            "node_id": "",
            "tablet_id": ""
        }
    
    log_data = match.groupdict()
    try:
        timestamp = datetime.strptime(
            log_data["timestamp"], 
            "%m%d %H:%M:%S.%f"
        ).isoformat()
    except ValueError:
        timestamp = ""
    
    return {
        "level": log_data["level"] or "U",
        "timestamp": timestamp or "",
        "thread_id": log_data["thread"] or "",
        "file": log_data["file"] or "",
        "node_id": log_data.get("node_id", "") or "",
        "tablet_id": log_data.get("tablet_id", "") or "",
        "message": log_data["message"] or "",
        "raw": line,
        "is_structured": True,
        "stacktrace": []
    }

def parse_logs(file_path: str) -> List[Dict]:
    """
    Parse complete log file with multi-line and stack trace support
    
    Args:
        file_path: Path to log file
        
    Returns:
        List of parsed log entries (dictionaries)
        Handles multi-line stack traces by attaching them to previous log entry
    """
    logs = []
    current_entry = None
    
    with open(file_path, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            
            if parsed and parsed["is_structured"]:
                if current_entry:
                    logs.append(current_entry)
                current_entry = parsed
            elif current_entry:
                if line.strip():
                    current_entry["stacktrace"].append(line.strip())
    
    if current_entry:
        logs.append(current_entry)
    
    return logs

# --------------------------
# ChromaDB Vector Database Functions
# --------------------------
def setup_chroma() -> chromadb.Collection:
    """
    Initialize ChromaDB collection with error handling
    
    Returns:
        ChromaDB collection object
        - Creates new collection if none exists
        - Connects to existing collection if available
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection(
            name="yugabyte_logs",
            embedding_function=embedder
        )
        st.session_state.collection_loaded = True
    except ValueError:
        collection = client.create_collection(
            name="yugabyte_logs",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"}
        )
        st.session_state.collection_loaded = False
    
    return collection

def clean_metadata(log: Dict) -> Dict:
    """
    Ensure all metadata values are ChromaDB compatible
    
    Args:
        log: Parsed log entry dictionary
        
    Returns:
        Dictionary with guaranteed valid types:
        - None → empty string
        - All values converted to str except bool
    """
    return {
        "level": str(log["level"]),
        "timestamp": str(log["timestamp"]),
        "node_id": str(log["node_id"]),
        "tablet_id": str(log["tablet_id"]),
        "file": str(log.get("file", "")),
        "has_stacktrace": bool(log.get("stacktrace", []))
    }

def ingest_logs(collection: chromadb.Collection, logs: List[Dict]):
    """
    Safely ingest logs into ChromaDB with batch processing
    
    Args:
        collection: ChromaDB collection object
        logs: List of parsed log entries
        
    Processes in batches for memory efficiency
    Includes progress bar visualization
    """
    batch_size = 1000
    total_batches = (len(logs) // batch_size + 1)
    progress_bar = st.progress(0)
    
    for i in range(0, len(logs), batch_size):
        batch = logs[i:i+batch_size]
        
        documents = []
        metadatas = []
        ids = []
        
        for j, log in enumerate(batch):
            documents.append(log["message"])
            metadatas.append(clean_metadata(log))
            ids.append(f"log_{i+j}")
        
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            progress_bar.progress(min((i + batch_size) / len(logs), 1.0))
        except Exception as e:
            st.error(f"Batch {i//batch_size + 1}/{total_batches} failed: {str(e)}")
            st.error(f"Problematic metadata: {metadatas[j]}")
            break

# --------------------------
# UI Components
# --------------------------
def display_log_entry(log: Dict):
    """
    Render a single log entry in expandable UI component
    
    Args:
        log: Parsed log entry dictionary
        
    Displays:
        - Metadata in JSON format
        - Main log message
        - Stack trace (if exists)
        - Raw log line
    """
    with st.expander(f"{log['timestamp']} [{log['level']}] {log['message'][:100]}..."):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("**Metadata**")
            st.json({
                "Level": log["level"],
                "Timestamp": log["timestamp"],
                "Node": log["node_id"],
                "Tablet": log["tablet_id"],
                "Thread": log["thread_id"],
                "File": log["file"]
            })
        
        with col2:
            st.markdown("**Message**")
            st.code(log["message"], language="text")
            
            if log.get("stacktrace"):
                st.markdown("**Stack Trace**")
                st.code("\n".join(log["stacktrace"]), language="text")
        
        st.markdown("**Raw Log**")
        st.code(log["raw"], language="text")

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'collection_loaded' not in st.session_state:
        st.session_state.collection_loaded = False

# --------------------------
# Main Application
# --------------------------
def main():
    """Main Streamlit application workflow"""
    st.set_page_config(page_title="YugabyteDB Log Analyzer", layout="wide")
    st.title("YugabyteDB Log Analyzer")
    init_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])
        
        if uploaded_file:
            with st.spinner("Processing..."):
                # Save to temp file and parse
                temp_file = "temp.log"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.logs = parse_logs(temp_file)
                os.remove(temp_file)
                
                # Initialize ChromaDB and ingest logs
                st.session_state.collection = setup_chroma()
                ingest_logs(st.session_state.collection, st.session_state.logs)
                
                st.success(f"Processed {len(st.session_state.logs)} entries")
    
    # Main Query Interface
    query = st.text_input("Search logs", help="Example: 'Find timeout errors'")
    
    if query and st.session_state.collection:
        with st.spinner("Analyzing..."):
            # 1. Query ChromaDB vector store
            results = st.session_state.collection.query(
                query_texts=[query],
                n_results=10,
                include=["documents", "metadatas"]
            )
            
            # 2. Retrieve matching full log entries
            retrieved_logs = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                for log in st.session_state.logs:
                    if (log["message"] == doc and 
                        log["timestamp"] == meta["timestamp"]):
                        retrieved_logs.append(log)
                        break
            
            # 3. Generate LLM analysis
            st.subheader("Analysis")
            context = "\n".join([log["raw"] for log in retrieved_logs])
            response = ollama.generate(
                model="llama3",
                system=SYSTEM_PROMPT,
                prompt=f"Question: {query}\n\nLogs:\n{context}"
            )
            st.markdown(response["response"])
            
            # 4. Display matching logs
            st.subheader("Matching Logs")
            for log in retrieved_logs:
                display_log_entry(log)

if __name__ == "__main__":
    main()
