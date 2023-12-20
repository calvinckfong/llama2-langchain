# Simple RAG with LangChain + Ollama + ChromaDB

## Prerequisite
- Python >=3.10
- LangChain
- ChromaDB
- Ollama
- BeautifulSoup4
- sentence-transformers

## Setup virtual envirnoment (recommended)
```
# Create virtual envirnoment
python3 -m venv langchain-venv

# Activate 
source langchain-venv/bin/activate
```

## Install Ollama
```
curl https://ollama.ai/install.sh | sh

# Start ollama server
ollama serve
```
More information: 
- https://github.com/jmorganca/ollama/blob/main/README.md
- https://github.com/jmorganca/ollama/blob/main/docs/linux.md


## Install other packages
```
pip3 install -r requirements.txt
```

## Start chromadb docker
```
cd chromadb && docker-compose up -d
```

## Import documents to chromaDB
- Place documents to be imported in folder `KB`
- Run:
  ```
  python3 import_doc.py
  ```
- Documents are read by dedicated loader
- Documents are splitted into chunks
- Chunks are encoded into embeddings (using `sentence-transformers` with `all-MiniLM-L6-v2`)
- embeddings are inserted into chromaDB

## Query document
- Run:
  ```
  python3 query.py [question]
  ```
- [question] is encoded into embedding
- Query chromaDB via embedding 

## Simple RAG (Retrieval-Augmented Generation)
- Run:
  ```
  python3 simple-rag.py [question]
  ```
- [question] is encoded into embedding
- Query chromaDB via embedding 
- Rephase query results using LLM (llama2)

## Todo
* [X] Documents import to chromaDB
* [X] Simple document query
* [X] Simple RAG