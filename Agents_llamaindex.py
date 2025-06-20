from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from pydantic import BaseModel, Field
import asyncio
import dotenv
import os

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)

def build_rag_tool():
    documents = SimpleDirectoryReader("data").load_data()
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = parser.get_nodes_from_documents(documents)

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FaissVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    query_engine = index.as_query_engine()

    return QueryEngineTool.from_defaults(
        name="DocumentQA",
        description="Answer questions from the local documents.",
        query_engine=query_engine,
    )

class MultiplyToolSchema(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults(
    name="multiply",
    fn=multiply,
    description="Multiply two numbers",
    fn_schema=MultiplyToolSchema
)

async def main():
    rag_tool = build_rag_tool()
    agent = FunctionAgent(
        name="GeminiRAGAgent",
        tools=[rag_tool, multiply_tool],
        llm=llm,
        system_prompt="You are a helpful assistant that can answer document-based questions and do math.",
    )

    print("\n Q1: What is 25 * 8?")
    resp1 = await agent.arun("What is 25 * 8?")
    print(" A1:", resp1)

    print("\n Q2: What is the document about?")
    resp2 = await agent.arun("What is the document about?")
    print("A2:", resp2)

if __name__ == "__main__":
    asyncio.run(main())