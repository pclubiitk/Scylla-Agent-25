from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.workflow import Context
from llama_index.tools.yahoo_finance import *
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
    FunctionAgent,
)
from pydantic import *
import os
import dotenv
import asyncio
dotenv.load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
llm = GoogleGenAI(model="gemini-2.0-flash", api_key = GOOGLE_API_KEY)
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

#LlamaHub has a lot of inbuilt tools which we can use rightaway
def BasicTools():
    def multiply(a: float, b:float)->float:
        return a*b
    def add(a: float, b:float)->float:
        return a+b
    finance_tools = YahooFinanceToolSpec().to_tool_list().extend([multiply, add])
    workflow = FunctionAgent(
        name="Agent",
        description="Useful for performing financial operations.",
        tools=finance_tools,
        llm=llm,
        system_prompt="You are a helpful assistant.",
    )
    ctx = Context(workflow) #Context can also be serialized as serializable (so we can later de-serialize it and use it)
    async def main1():
        response2 = await workflow.run("What's the current stock price of NVIDIA?")
        response1 = await workflow.run(user_msg="What is 20+(2*4)?")
        print(response1, "\n", response2)
    async def intros():
        print("User: Hi, my name is Albus Percival Wulfric Brian Dumbledore!")
        resp1 = await workflow.run(user_msg="Hi, my name is Albus Percival Wulfric Brian Dumbledore!", ctx=ctx)
        print("System:", resp1)
        print("User: Wait. Do you remember my name?")
        resp2 = await workflow.run(user_msg="Wait. Do you remember my name?", ctx=ctx)
        print("System:", resp2)
    asyncio.run(intros())

def UsingTavilyTool():
    workflow = FunctionAgent(
        tools=tavily_tool.to_tool_list(),
        llm=llm,
        system_prompt="You're a helpful assistant that can search the web for information.",
    )
    async def main1():
        response = workflow.run(user_msg="What's the weather like in Coimbatore?")
        async for event in response.stream_events():
            if isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
    asyncio.run(main1())

UsingTavilyTool()