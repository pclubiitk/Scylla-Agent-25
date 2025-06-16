"""
This is based on 'Query Transformation' Rag technique as based on 
'https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb'

So basically what we'll do is 'Query-Rewriting' were we'll give another query which will give us a query that is
enriched to include specific aspects of the original prompt
"""

import os
import dotenv
from groq import Groq

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

def rewrittenquery(original_query, query_rewriter):
    return query_rewriter.invoke(original_query).content

dotenv.load_dotenv()

NOMIC_KEY = os.getenv("NOMIC_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
file_path = './HarryPotter.txt'

loader = TextLoader(file_path)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=1500)
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

llm = ChatGroq(model_name="llama3-8b-8192", temperature=1, groq_api_key=GROQ_KEY)

query_converter = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=GROQ_KEY)

query_rewrite_prompt=ChatPromptTemplate.from_template("""
You are an excellent teacher - very good at explaining things in details. Your job is to take this original query:
Query: {query}
and rewrite it to be more general, detailed, and likely to retrieve relevant information.
The format should just be like a query. When there's more than one context possible - list every context in the query
""")

main_prompt = ChatPromptTemplate.from_template("""
Answer the question below. If the information is in the context, use it (BUT PLEASE DON'T SAY 'According to context'); 
otherwise, answer using your own knowledge (offer disclaimer in this case though).
When there's more than one context give every possible answer.
Be elaborate (but don't have the answer very LONG) and end the conversation warmly
Context: {context}

Question: {question}
""")

rag = (
    {"context": retriever, "question": RunnablePassthrough()}
    | main_prompt
    | llm
    | StrOutputParser()
)
os.system('clear')
print("Press Ctrl+C to stop execution at anytime\n")
while(1):
    query = input("Enter query: ")
    better_query = rewrittenquery(query, query_rewrite_prompt|query_converter)
    context = retriever.invoke(better_query)
    llm_invoked = llm.invoke(better_query)
    rag_invoked = rag.invoke(better_query)

    resp = Groq(api_key=GROQ_KEY).chat.completions.create(
        messages=[
            {"role":"user",
            "content": f'''Below attached are LLM invoked reply to query and RAG retrieved (from physical trusted docs) reply to query.
                Both of them need to be fact-checked to avoid hallucinations. Give higher weightage/importance to the RAG reply.
                Summarize, Fact check and give your reply. By the way talk like 'Hagrid' and yap a little before answering.
                When there's more than one context give every possible answer.
                Be elaborate (but don't have the answer very LONG) and end the conversation warmly
                RAG response: {rag_invoked}
                LLM reponse: {llm_invoked}
                Please don't comment on which is RAG, which is LLM. Just give one complete answer - Hagrid style
                '''
            }
        ],
        model = "llama3-8b-8192",
        stream=False,
        temperature=1,
        max_completion_tokens=8000,
    )

    print(resp.choices[0].message.content)
    print("\n\n")
