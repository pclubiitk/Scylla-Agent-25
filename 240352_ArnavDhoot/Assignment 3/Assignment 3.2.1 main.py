from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq  
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=200,
    groq_api_key=os.environ["GROQ_API_KEY"],
)
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0"
)   
pdf_path = "Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)
persist_directory = os.path.join(os.getcwd(), "chroma_db", "stock_market")
collection_name = "stock_market"

os.makedirs(persist_directory, exist_ok=True)
vectorstore = FAISS.from_documents(
    documents=pages_split,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns information from the Stock Market Performance 2024 document.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Document {i+1}:{doc.page_content}" for i, doc in enumerate(docs)])

llm = llm.bind_tools([retriever_tool])

system_prompt = """
You are an intelligent assistant helping with questions about Stock Market Performance in 2024 based on the PDF document.
Use the retriever tool when necessary and cite the document snippets.
"""

def run_rag_agent():
    print("\nRAG AGENT")
    while True:
        user_input = input("\nWhat is your question? (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

        # First LLM response
        response = llm.invoke(messages)
        messages.append(response)

        # Handle tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "retriever_tool":
                    result = retriever_tool.invoke(tool_call["args"].get("query", ""))
                    tool_msg = ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                        content=result
                    )
                    messages.append(tool_msg)
            response = llm.invoke(messages)

        print("\n=== ANSWER ===")
        print(response.content)
run_rag_agent()