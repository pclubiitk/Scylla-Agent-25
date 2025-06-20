import os
import dotenv
import asyncio

dotenv.load_dotenv()
TAVILY_API_KEY = os.environ["Tavily_API_key"]

from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
)

tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

from llama_index.llms.groq import Groq
llm = Groq(
    model="llama-3.1-8b-instant",  
    api_key=os.environ["GROQ_API_KEY"],
    stream=False
)


def multiply(a: float, b: float) -> float:
    return a * b

def add(a: float, b: float) -> float:
    return a + b

def create_combined_agent():
    finance_tools = YahooFinanceToolSpec().to_tool_list()
    tavily_tools = tavily_tool.to_tool_list()
    basic_math_tools = [
        FunctionTool.from_defaults(fn=multiply, name="multiply"),
        FunctionTool.from_defaults(fn=add, name="add"),
    ]
    all_tools = finance_tools + basic_math_tools + tavily_tools
    agent = FunctionAgent(
        name="UniversalAgent",
         description="A multi-skill assistant capable of performing math operations, finance lookups, and summarizing web search results.",
        tools=all_tools,
        llm=llm,
        system_prompt=(
            "You are UniversalAgent, an intelligent assistant that can:\n"
            "- Answer general questions.\n"
            "- Perform math operations using available functions.\n"
            "- Retrieve stock and financial data using finance tools.\n"
            "- Search the web and summarize relevant information, such as news or weather.\n"
            "Respond concisely and helpfully using available tools when applicable."
        ),
    )
    return agent

async def main():
    agent = create_combined_agent()
    queries = [
        "Hi, my name is Albus Percival Wulfric Brian Dumbledore!",
        "Wait. Do you remember my name?",
        "What is 20 + (2 * 4)?",
        "What is the current stock price of NVIDIA?",
        "What's the weather like in Coimbatore?"
    ]
    for q in queries:
        print(f"\n User: {q}")
        response = await agent.run(user_msg=q)
        print(f" Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())

