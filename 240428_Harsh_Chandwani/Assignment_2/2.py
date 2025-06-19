import dotenv
dotenv.load_dotenv()

import os
import requests
from langchain_community.document_loaders import TextLoader 

url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/master/state_of_the_union.txt"
res = requests.get(url)
text = res.text

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text) 

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma  

embedding = CohereEmbeddings(
    cohere_api_key=os.getenv("cohere_API_key"),  
    model="embed-english-v3.0"
)

db = Chroma.from_texts(chunks, embedding=embedding)
retriever = db.as_retriever(search_kwargs={"k": 5})

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

# LLM setup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("google_API_key"), 
    temperature=0
)

# RAG chain
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

query = "What did the president say about Justice Breyer"
r = rag_chain.invoke(query)
print(r)
