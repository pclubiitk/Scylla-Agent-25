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
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from llama_index.utils.workflow import draw_all_possible_flows
from pydantic import *
import os
import dotenv
import asyncio
dotenv.load_dotenv()
import random
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
llm = GoogleGenAI(model="gemini-2.0-flash", api_key = GOOGLE_API_KEY)
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

'''
Application is divided into sections called Steps which are triggered by Events, and themselves emit Events which trigger further steps. 
By combining steps and events, we can create arbitrarily complex flows that encapsulate logic and 
make your application more maintainable and easier to understand
'''
'''
JUDGE EVENT -> (Judge Query <-> Improve Query) -> NaiveRAGvent, HighTopKEvent, Rerank Event -> Response Event -> JUDGE RESPONSE
Event Based Pattern of Workflows aim to resolve the limitations of DAG
'''

class FirstEvent(Event):
    first_output: str
class SecondEvent(Event):
    second_output: str
class LoopEvent(Event):
    loop_output: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")
    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")
async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run(first_input = "Start the Workflow")
    print(result)
    draw_all_possible_flows(MyWorkflow, filename="multi_step_workflow.html")
asyncio.run(main())


"""
Similarly for Branching we can do this-
class BranchA1Event(Event):
    payload: str
class BranchA2Event(Event):
    payload: str
class BranchB1Event(Event):
    payload: str
class BranchB2Event(Event):
    payload: str

class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchA1Event | BranchB1Event:
        if random.randint(0, 1) == 0:
            print("Go to branch A")
            return BranchA1Event(payload="Branch A")
        else:
            print("Go to branch B")
            return BranchB1Event(payload="Branch B")

    @step
    async def step_a1(self, ev: BranchA1Event) -> BranchA2Event:
        print(ev.payload)
        return BranchA2Event(payload=ev.payload)

    @step
    async def step_b1(self, ev: BranchB1Event) -> BranchB2Event:
        print(ev.payload)
        return BranchB2Event(payload=ev.payload)

    @step
    async def step_a2(self, ev: BranchA2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch A complete.")

    @step
    async def step_b2(self, ev: BranchB2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch B complete.")
"""

'''
For maintaining context, similar to BuildingAgents we do this-
class SetupEvent(Event):
    query: str
class StepTwoEvent(Event):
    query: str
class StatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> SetupEvent | StepTwoEvent:
        db = await ctx.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)
        return StepTwoEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        await ctx.set("some_database", [1, 2, 3])
        return StartEvent(query=ev.query)
    
    @step
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        print("Data is ", await ctx.get("some_database"))
        return StopEvent(result=await ctx.get("some_database"))
async def main():
    w = StatefulFlow(timeout=10, verbose=False)
    result = await w.run(query="Some query")
    print(result)

'''