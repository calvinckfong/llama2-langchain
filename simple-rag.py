import chromadb, sys
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.llms import ollama
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from pprint import pprint

def RAG(query:str):
    # setup llm
    llm = ollama.Ollama(model="llama2", temperature=0.2)
    # setup reiever for external data
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.HttpClient()
    db = Chroma(client=client, embedding_function=embedding_function, collection_name="KnowledgeBase")
    # setup prompt
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    # setup QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_llama},
    )
    
    return qa_chain({"query": query})


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error. Too few arguments.')
        exit()    
        
    query = sys.argv[1]
    answer = RAG(query)
    
    pprint(answer) 