import chromadb, sys
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from pprint import pprint

def QueryDoc(query:str):
    # # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    client = chromadb.HttpClient()
    db = Chroma(client=client, embedding_function=embedding_function, collection_name="KnowledgeBase")
    
    return db.similarity_search(query)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error. Too few arguments.')
        exit()
        
    query = sys.argv[1]
    docs = QueryDoc(query)
    
    pprint(docs)