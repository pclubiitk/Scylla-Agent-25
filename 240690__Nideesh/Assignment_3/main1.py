import os
import sys
from dotenv import load_dotenv
from google.colab import userdata

GOOGLE_API_KEY=userdata.get("GOOGLE_API_KEY")
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def replacetabwithspace(documentslist):
  for doc in documentslist:
    doc.page_content=doc.page_content.replace('/t', ' ')
  return documentslist

def textwrapping(text, width=120):
  return textwrap.fill(text,width=width)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):

    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replacetabwithspace(texts)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY) # Specify the Gemini embedding model
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def retrieve_context_per_question(question, chunks_query_retriever):
    docs = chunks_query_retriever.get_relevant_documents(question)

    context = [doc.page_content for doc in docs]

    return context

import os
os.makedirs('data', exist_ok=True)

# Download the PDF document used in this notebook
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = "data/Understanding_Climate_Change.pdf"

chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

def show_context(context):
    for i, text in enumerate(context):
        print(f"--- Context Snippet {i+1} ---")
        print(text)
        print("\n")

test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": chunks_query_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke(test_query)

print(response)
