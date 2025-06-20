# main.py

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

documents = SimpleDirectoryReader("data").load_data()
parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = parser.get_nodes_from_documents(documents)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceLLM(
    model_name="tiiuae/falcon-7b-instruct", 
    tokenizer_name="tiiuae/falcon-7b-instruct",
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    device_map="auto"
)

vector_store = FaissVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
synthesizer = ResponseSynthesizer.from_defaults(llm=llm)

query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

query = "What are the rules of f1"
response = query_engine.query(query)

print(f"Question: {query}\nAnswer: {response}")