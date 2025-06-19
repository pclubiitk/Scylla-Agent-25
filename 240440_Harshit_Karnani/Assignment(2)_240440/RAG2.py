import kagglehub
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

paths = kagglehub.dataset_download(
    "whegedusich/president-bidens-state-of-the-union-2023")
with open(os.path.join(paths, "biden-sotu-2023-planned-official.txt"), "r", encoding="utf-8") as f:
    text = f.read()
with open("state_of_the_union.txt", "w") as f:
    f.write(text)
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

embedding = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=COHERE_API_KEY,
    truncate="END"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering in sarcastic ways. 
Use the following pieces of retrieved context to answer the question.
Answer in a sarcastic tone.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    temperature=0,
    model="command-light"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "Summarize the main points of Biden's state of union address 2023 speech."
print(rag_chain.invoke(query))
