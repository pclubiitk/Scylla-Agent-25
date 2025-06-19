from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

documents = SimpleDirectoryReader(
    input_files=[r"D:\RAG(research).pdf"]).load_data()

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2")

dimension = len(embed_model.get_text_embedding("pipeline"))
faiss_index = faiss.IndexFlatIP(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

response = query_engine.query("Summarise the main points of the document.")
print(response)
