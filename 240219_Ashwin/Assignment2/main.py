import os
from dotenv import load_dotenv

load_dotenv()

NOMIC_KEY = os.getenv("NOMIC_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

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

# Function to check if the context is empty or irrelevant
def irrelevant(context, q):
    if not context:
        return True
    text = " ".join([doc.page_content for doc in context])
    return len(text.strip()) < 10

file_path = "state_of_the_union.txt"

loader = TextLoader(file_path)
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

client = weaviate.Client(embedded_options=EmbeddedOptions())

embedding = NomicEmbeddings(model="nomic-embed-text-v1.5")

vectorstore = Weaviate.from_documents(
    client=client,
    documents=chunks,
    embedding=embedding,
    by_text=False
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7, groq_api_key=GROQ_KEY)

prompt = ChatPromptTemplate.from_template("""
Answer the question below. If the information is in the context, use it. 
If not, answer using your own knowledge and mention that you are doing so.
Be detailed but not too long, and end politely.

Context: {context}

Question: {question}
""")


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print("You can now ask a question.")
query = input("Your question: ")

# Get relevant context for the question
context = retriever.invoke(query)

# Decide how to respond
if not context or irrelevant(context, query):
    response = llm.invoke(query)
else:
    response = rag_chain.invoke(query)

# Show the answer
print("\n Answer:")
print(response)
