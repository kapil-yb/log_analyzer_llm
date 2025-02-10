"""
## ChromaDB + Ollama API for Log Analysis with YugabyteDB Docs

### Overview
This script sets up an API to analyze PostgreSQL/YugabyteDB logs using ChromaDB as a vector store and an Ollama-based LLM for retrieval. It also integrates **YugabyteDB documentation (`docs.yugabyte.com`)** into the RAG pipeline for better insights.

### Features
- Uses **ChromaDB** for efficient vector storage and retrieval.
- Leverages **Ollama** to generate embeddings and run LLM-based queries.
- Scrapes **YugabyteDB documentation** for real-time reference.
- Exposes a **FastAPI** endpoint to query logs and documentation.
- Implements a **Retrieval-Augmented Generation (RAG)** approach to enhance log analysis accuracy.

### High-Level Steps to Build the YB Log Analyzer
1. **Set up a vector database**: Use ChromaDB to store and retrieve log and documentation embeddings.
2. **Generate embeddings**: Convert log messages and documentation into embeddings using Ollama’s embedding model.
3. **Store logs and docs**: Persist embeddings and content in ChromaDB for retrieval.
4. **Use a retriever**: Configure LangChain’s `Chroma` retriever to fetch relevant logs and docs based on queries.
5. **Integrate an LLM**: Use Ollama’s local model (e.g., `mistral`) to process queries with retrieved logs and docs.
6. **Expose an API**: Implement a FastAPI-based interface to accept log-related queries and return responses.
7. **Run the API and test queries**: Start the API and use `cURL` or a browser to query the log database.

### Dependencies
Make sure you have the following installed:
```sh
pip install ollama chromadb fastapi langchain uvicorn requests beautifulsoup4
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
import requests
from bs4 import BeautifulSoup
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

# Step 4: Scrape YugabyteDB Documentation

def scrape_yugabyte_docs():
    url = "https://docs.yugabyte.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    docs = []
    
    for link in soup.find_all("a", href=True):
        full_url = requests.compat.urljoin(url, link["href"])
        if full_url.startswith("https://docs.yugabyte.com/"):
            print(f"Fetching: {full_url}")  # Debug output
            try:
                page = requests.get(full_url)
                page_soup = BeautifulSoup(page.text, "html.parser")
                text = page_soup.get_text()
                docs.append((full_url, text))
            except Exception as e:
                print(f"Error fetching {full_url}: {e}")
    
    print(f"Total documents scraped: {len(docs)}")
    return docs

# Store documentation embeddings
yb_docs = scrape_yugabyte_docs()
for i, (url, text) in enumerate(yb_docs):
    print(f"Embedding doc {i+1}/{len(yb_docs)}: {url}")
    embedding = get_ollama_embedding(text)
    chroma_collection.add(ids=[f"doc_{i}"], embeddings=[embedding], documents=[text], metadatas=[{"url": url}])

# Step 5: Define an embedding function class
class OllamaEmbeddings:
    def embed_query(self, text: str):
        return get_ollama_embedding(text)

    def embed_documents(self, texts):
        return [get_ollama_embedding(text) for text in texts]

embedding_function = OllamaEmbeddings()

# Step 6: Initialize Chroma Vector Store with embedding function
vector_store = Chroma(
    client=chroma_client,
    collection_name="log_embeddings",
    embedding_function=embedding_function  # ✅ Fix: Provide an embedding function
)
retriever = vector_store.as_retriever()

# Step 7: Initialize Ollama LLM Locally
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 8: Set Up FastAPI for Querying Logs and Docs
app = FastAPI()

@app.get("/query")
async def query_logs(query: str):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, qa_chain.run, query)
    return {"response": response}

# Step 9: Run the API (uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Sample query run:

% curl -s "http://localhost:8000/query?query=$(echo "What are the different yugabyte deployment options?" | sed 's/ /%20/g')" | jq -r '.response'
 Yugabyte offers several deployment options to suit various use cases and environments. Here are some of them:

1. Cloud Native: Yugabyte can be deployed on popular cloud platforms like AWS, Azure, Google Cloud Platform (GCP), Alibaba Cloud, and others. This allows you to leverage the scalability, reliability, and security benefits provided by these platforms.

2. On-premises: If you prefer to keep your data center or have specific compliance requirements, Yugabyte can be deployed on-premises.

3. Kubernetes: As a cloud-native database, YugabyteDB is highly compatible with Kubernetes environments, making it easier to scale and manage your database alongside other applications in a containerized environment.

4. Geo-distributed deployments: For low latency and high availability across multiple regions, Yugabyte allows you to deploy clusters spanning multiple data centers. This improves performance and reliability for geographically dispersed applications.

5. Edge computing: In edge computing scenarios, where IoT devices or remote locations require local processing, Yugabyte can be deployed to manage data at the edge.

6. Hybrid deployments: A hybrid deployment strategy combines on-premises and cloud resources for greater flexibility and scalability in a single topology. You can seamlessly extend your on-premises deployment to the cloud or vice versa, as needed.
""""
