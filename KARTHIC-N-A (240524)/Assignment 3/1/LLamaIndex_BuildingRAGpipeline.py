from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pydantic import *
import os
import dotenv
dotenv.load_dotenv()
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import download_loader
from llama_index.core.node_parser import SentenceSplitter

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
embed = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=GOOGLE_API_KEY)
llm = GoogleGenAI(model="gemini-2.0-flash", api_key = GOOGLE_API_KEY)
Settings.llm = llm
Settings.embed_model = embed

def RAGGing():
    documents = SimpleDirectoryReader("./data").load_data()    
    text_splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=1500)
    Settings.text_splitter = text_splitter

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)    
    query_engine = index.as_query_engine()
    
    response = query_engine.query("Who killed Sirius Black?")
    print(response)

RAGGing()