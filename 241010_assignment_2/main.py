import os
from dotenv import load_dotenv
import weaviate

# Load .env variables
load_dotenv() 

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


os.environ["GROQ_API_KEY"] = GROQ_API_KEY


loader = TextLoader("state_of_the_union.txt", encoding="utf-8")
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=25)
docs = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(docs, embeddings)


groq_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    max_tokens=200,
)


rag_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    chain_type="stuff",  # Stuff the context into the prompt
    retriever=vectorstore.as_retriever(),
    return_source_documents=False  # Optional: set to True to also get the source docs
)


query = "Give a summary of the state of the union address?"
result = rag_chain.run(query)

print("\nAnswer:")
print(result)
