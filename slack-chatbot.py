import chromadb, json, logging, os

from langchain import hub
from langchain.chains import RetrievalQA
from langchain.llms import ollama
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers import WikipediaRetriever

from pprint import pprint

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Add SLACK_BOT_TOKEN and SLACK_APP_TOKEN to .env file
from dotenv import load_dotenv
load_dotenv()

# Ref: https://api.slack.com/tutorials/tracks/responding-to-app-mentions

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='/tmp/slack-chatbot.log',
                    filemode='a')

class RAG:
    def __init__(self):
        # setup llm
        self.llm = ollama.Ollama(model="llama2", temperature=0.2)
        # setup reiever for external data
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        client = chromadb.HttpClient()
        self.db = Chroma(client=client, 
                         embedding_function=embedding_function, 
                         collection_name="KnowledgeBase",
                         collection_metadata={"hnsw:space": "cosine"})
        
        # setup prompt
        self.rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
        #print('RAG initialized.')
        logging.info('RAG initialized')
    
    def LLM(self, query:str):
        # qa_chain = RetrievalQA.from_chain_type(
        #     self.llm,
        #     retriever=self.db.as_retriever(
        #         search_type="similarity", 
        #         search_kwargs={"k": 1}),
        #     chain_type_kwargs={"prompt": self.rag_prompt_llama},
        # )
        # response = qa_chain({"query": query})["result"]
        logging.info(f'LLM: {query}')
        response = self.llm(query)
        return response

    def Wiki(self, query:str):
        retriever = WikipediaRetriever(load_all_available_meta=True)
        qa_chain = RetrievalQA.from_llm(self.llm, retriever=retriever)
        logging.info(f'Wiki: {query}')
        result = qa_chain({"query": query})
        logging.info(json.dumps(result))
        return result["result"]        

rag = RAG()

# Install the Slack app and get xoxb- token in advance
app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.command("/chatbot")
def llm_command(ack, body, respond):
    user_id = body["user_id"]
    query = body['text']
    ack(f"*Question: {query}*")
    response = rag.LLM(query)
    #pprint(response)
    respond(f'<@{user_id}> {response}')

@app.command("/wiki")
def wiki_command(ack, body, respond):
    user_id = body["user_id"]
    query = body['text']
    ack(f"*Question: {query}*")
    result = rag.Wiki(query)
    #pprint(response)
    respond(f'<@{user_id}> {result}')


@app.event("app_mention")
def event_test(event, say):
    #say("Try `/chatbot [question]`")
    #pprint(event)
    query = event['text']
    response = rag.Query(query)
    #say(f"<@{user_id}>'s Question: {query} \nAnswer: {response['result']}")
    say(response)

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()