from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
import os
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key=os.environ["OPENAI_API_KEY"]
)


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


tool_multiply = FunctionTool.from_defaults(fn=multiply)
tool_add = FunctionTool.from_defaults(fn=add)
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([tool_multiply, tool_add])

financial_agent = AgentRunner.from_llm(
    tools=finance_tools,
    llm=llm,
    system_prompt="""You are a financial analyst assistant. 
    Use available tools to get accurate financial data and perform calculations.""",
    verbose=True
)

tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

research_agent = AgentRunner.from_llm(
    tools=[tavily_tool],
    llm=llm,
    system_prompt="""You are a helpful assistant that can search the web for information."""
)


async def main():

    stock_response = await financial_agent.chat("What's NVIDIA's current stock price?")
    print("\nStock Price Response:")
    print(stock_response)

    research_response = await research_agent.chat("What's the weather like in San Francisco?")
    print("\nResearch Response:")
    print(research_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
