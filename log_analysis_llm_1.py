"""
## ChromaDB + Ollama API for Log Analysis

### Overview
This script sets up an API to analyze PostgreSQL/YugabyteDB logs using ChromaDB as a vector store and an Ollama-based LLM for retrieval. The API supports querying logs for issue analysis.

### Features
- Uses **ChromaDB** for efficient vector storage and retrieval.
- Leverages **Ollama** to generate embeddings and run LLM-based queries.
- Exposes a **FastAPI** endpoint to query logs.
- Implements a **Retrieval-Augmented Generation (RAG)** approach to enhance log analysis accuracy.

### High-Level Steps to Build the YB Log Analyzer
1. **Set up a vector database**: Use ChromaDB to store and retrieve log embeddings.
2. **Generate embeddings**: Convert log messages into embeddings using Ollama’s embedding model.
3. **Store logs**: Persist embeddings and log messages in ChromaDB for retrieval.
4. **Use a retriever**: Configure LangChain’s `Chroma` retriever to fetch relevant logs based on queries.
5. **Integrate an LLM**: Use Ollama’s local model (e.g., `mistral`) to process queries and retrieved logs.
6. **Expose an API**: Implement a FastAPI-based interface to accept log-related queries and return responses.
7. **Run the API and test queries**: Start the API and use `cURL` or a browser to query the log database.

### Dependencies
Make sure you have the following installed:
```sh
pip install ollama chromadb fastapi langchain uvicorn
```

### How to Run
1. **Run the script directly**:
   ```sh
   python <script_name>.py
   ```
2. **Alternatively, start the API using uvicorn**:
   ```sh
   uvicorn main:app --reload
   ```
3. **Query logs using cURL**:
   ```sh
   curl "http://localhost:8000/query?query=Why is my database connection failing?"
   ```

### Code Implementation
"""

import ollama
import chromadb
from fastapi import FastAPI
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import asyncio

# Step 1: Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="log_embeddings")

# Step 2: Custom function to generate embeddings using Ollama
def get_ollama_embedding(text: str):
    response = ollama.embeddings(model="all-minilm", prompt=text)
    return response["embedding"]

# Step 3: Load Sample Data (Placeholder for YugabyteDB/PostgreSQL Logs)
log_entries = [
    "ERROR 42601: Syntax error in SQL statement",
    "FATAL: connection to database failed",
    "Replication slot not found"
]

# Convert logs to embeddings and store in ChromaDB
for i, log in enumerate(log_entries):
    embedding = get_ollama_embedding(log)
    chroma_collection.add(ids=[str(i)], embeddings=[embedding], documents=[log])

# Step 4: Define an embedding function class
class OllamaEmbeddings:
    def embed_query(self, text: str):
        return get_ollama_embedding(text)

    def embed_documents(self, texts):
        return [get_ollama_embedding(text) for text in texts]

embedding_function = OllamaEmbeddings()

# Step 5: Initialize Chroma Vector Store with embedding function
vector_store = Chroma(
    client=chroma_client,
    collection_name="log_embeddings",
    embedding_function=embedding_function  # ✅ Fix: Provide an embedding function
)
retriever = vector_store.as_retriever()

# Step 6: Initialize Ollama LLM Locally
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 7: Set Up FastAPI for Querying Logs
app = FastAPI()

@app.get("/query")
async def query_logs(query: str):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, qa_chain.run, query)
    return {"response": response}

# Step 8: Run the API (uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
