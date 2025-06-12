import os
import dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Weaviate

import weaviate
from weaviate.embedded import EmbeddedOptions

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def irrelevant(context, q, thresh=0.2):
    if not context:
        return True
    text = " ".join([data.page_content for data in context])
    return len(text.strip()) < 1

dotenv.load_dotenv()

NOMIC_KEY = os.getenv("NOMIC_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
file_path = './state_of_the_union.txt'

loader = TextLoader(file_path)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

embed = NomicEmbeddings(model="nomic-embed-text-v1.5")

store = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = embed,
    by_text = False
)

retriever = store.as_retriever()

print("Vectorstore created successfully with NomicEmbeddings!")

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7, groq_api_key=GROQ_KEY)

prompt = ChatPromptTemplate.from_template("""
Answer the question below. If the information is in the context, use it (BUT PLEASE DON'T SAY 'According to context'); 
otherwise, answer using your own knowledge (offer disclaimer in this case though).
Be elaborate (but don't have the answer very LONG) and end the conversation warmly
Context: {context}

Question: {question}
""")

rag = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
os.system('clear')

query = input("Enter query: ")
context = retriever.invoke(query)
if not context or irrelevant(context, query):
    resp = llm.invoke(query)
else:
    resp = rag.invoke(query)

print("\n--- RAG Chain Response ---")
print(resp)
print("\n\n")
