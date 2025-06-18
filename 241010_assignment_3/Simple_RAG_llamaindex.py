import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext
import faiss

# Load API Key from .env
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

documents = SimpleDirectoryReader(input_files=[r"D:\vs_code_python\SCYLLAAGENT\final_draft.pdf"]).load_data()
print(len(documents))


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

dimension = len(embed_model.get_text_embedding("test"))
faiss_index = faiss.IndexFlatIP(dimension)
vector_store = FaissVectorStore(faiss_index = faiss_index)

storage_context = StorageContext.from_defaults(vector_store = vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)
#print(index._index_struct)  # Shows doc-to-node mapping

#retriever = index.as_retriever(similarity_top_k=10)
#nodes = retriever.retrieve("List names of all mentors")

#for i, node in enumerate(nodes):
    #print(f"\n--- Retrieved Node {i} ---\n")
    #print(node.text[:500])  # print first 500 chars of each

llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_key)

query_engine = index.as_query_engine(llm=llm,similarity_top_k=10)

response = query_engine.query("Mentors: What names are listed under mentors?")
print(response)