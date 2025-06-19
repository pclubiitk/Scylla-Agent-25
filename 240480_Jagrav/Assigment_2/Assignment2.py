import dotenv
dotenv.load_dotenv()

import requests
from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
client = weaviate.Client(embedded_options=EmbeddedOptions())
vectorstore = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY), 
    by_text = False
)

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """Answer the question below. If the information is in the context, use it.
If you don't know the answer, just say that you don't know.
Answer in a sarcastic tone.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.5, groq_api_key=GROQ_KEY)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query = "What was told about Justice Breyer by the President?"
print(rag_chain.invoke(query))
