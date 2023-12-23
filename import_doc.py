import chromadb, os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

def ReadDocs(pathname:str):
    documents = []
    for file in os.listdir(pathname):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pathname, file)
            print(pdf_path)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(pathname, file)
            print(doc_path)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = os.path.join(pathname, file)
            print(text_path)
            loader = TextLoader(text_path)
            documents.extend(loader.load())
        print(f'{len(documents)} docs accumulated')
    print(f'{len(documents)} documents loaded.')
    return documents

def SplitDocs(docs):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    #chunks = splitter.split_text(docs)
    print(f'{len(chunks)} chunks')
    print(chunks[10])
    return chunks 

def InsertDB(host:str, port:str, collection:str, documents):
    assert len(collection)>2 and len(collection)<64
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # # load it into Chroma
    client = chromadb.HttpClient(host=host, port=port)
    print(f'Insert to {host}:{port} {collection}')
    client.delete_collection(collection)
    Chroma.from_documents(documents, embedding_function, client=client, collection_name=collection)   
    
    print(f'{client.get_collection(collection).count()} embeddings in collect {collection}')

if __name__ == '__main__':
    doc_path = './kb'
    documents = ReadDocs(doc_path)
    chunks = SplitDocs(documents)
    InsertDB(host='localhost', port='8000', collection='KnowledgeBase', documents=chunks)
    
